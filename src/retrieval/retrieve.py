import pickle
import re
from pathlib import Path

import numpy as np



# CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_index.pkl"
)



# TEXT CLEANING


def clean_text(text: str) -> str:
    """
    Clean text in the same way as the retrieval index.
    """

    if not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()



# LOAD INDEX


def load_index(index_path=INDEX_PATH):
    """
    Load the TF-IDF retrieval index from disk.
    """

    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {index_path}\n"
            "Run: python src/retrieval/index.py"
        )

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    return index



# RETRIEVER


class AmazonHelpRetriever:
    """
    TF-IDF based retriever for historical AmazonHelp conversations.
    """

    def __init__(self, index_path=INDEX_PATH):

        # Load saved index
        self.index = load_index(index_path)

        # Validate required keys
        required_keys = {
            "vectorizer",
            "matrix",
            "metadata",
        }

        missing_keys = required_keys - set(self.index.keys())

        if missing_keys:
            raise ValueError(
                f"Invalid retrieval index. Missing keys: {missing_keys}"
            )

        # Load components
        self.vectorizer = self.index["vectorizer"]
        self.matrix = self.index["matrix"]
        self.metadata = self.index["metadata"]

        # The index does not store a separate "documents" key.
        # Customer messages are stored inside metadata.
        self.documents = [
            item.get("customer_message", "")
            for item in self.metadata
        ]

        # Validate dimensions
        if len(self.documents) != self.matrix.shape[0]:
            raise ValueError(
                "Number of documents does not match "
                "TF-IDF matrix rows."
            )

        if len(self.metadata) != len(self.documents):
            raise ValueError(
                "Number of metadata records does not "
                "match documents."
            )


    # ========================================================
    # SEARCH
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        predicted_intent: str = None,
    ):
        """
        Retrieve the most similar historical AmazonHelp
        conversations.

        Parameters
        ----------
        query : str
            New customer message.

        top_k : int
            Number of historical examples to return.

        min_score : float
            Minimum cosine similarity score.

        Returns
        -------
        list[dict]
            Ranked historical examples.
        """

        # Handle empty query
        if not query or not query.strip():
            return []

        # Clean query
        query = clean_text(query)

        if not query:
            return []

        # ----------------------------------------------------
        # Convert query into TF-IDF vector
        # ----------------------------------------------------

        query_vector = self.vectorizer.transform([query])

        # ----------------------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------------------
        #
        # TF-IDF vectors are L2-normalized by default.
        # Therefore:
        #
        # cosine_similarity(A, B) = A dot B
        #
        # ----------------------------------------------------

        scores = self.matrix @ query_vector.T

        scores = np.asarray(
            scores.toarray()
        ).ravel()

        # Preserve raw TF-IDF cosine similarity for diagnostics. The public
        # `score` below is an intent-adjusted retrieval score, not a claim of
        # semantic similarity from an embedding model.
        lexical_scores = scores.copy()

        # ----------------------------------------------------
        # Intent-Aware Re-ranking (Stage 2)
        # ----------------------------------------------------

        candidate_intents = {}

        if predicted_intent:
            # We don't want to classify all 200,000 items.
            # Get top N based on raw TF-IDF score
            top_n = max(top_k * 4, 20)

            candidate_indices = np.where(
                scores >= min_score
            )[0]

            if len(candidate_indices) > 0:
                raw_ranked = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
                re_rank_candidates = raw_ranked[:top_n]

                if not hasattr(self, "classifier"):
                    from src.intent.classifier import IntentClassifier
                    self.classifier = IntentClassifier()

                for idx in re_rank_candidates:
                    candidate_msg = self.metadata[idx].get("customer_message", "")
                    if candidate_msg:
                        candidate_intent = self.classifier.predict(candidate_msg)
                        candidate_intents[idx] = candidate_intent
                        if candidate_intent != predicted_intent:
                            scores[idx] *= 0.3

        # ----------------------------------------------------
        # Filter by minimum similarity
        # ----------------------------------------------------

        candidate_indices = np.where(
            scores >= min_score
        )[0]

        if len(candidate_indices) == 0:
            return []

        # ----------------------------------------------------
        # Sort by similarity
        # Highest score first
        # ----------------------------------------------------

        ranked_indices = candidate_indices[
            np.argsort(
                scores[candidate_indices]
            )[::-1]
        ]

        # Keep only top K
        ranked_indices = ranked_indices[:top_k]

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results = []

        for rank, idx in enumerate(
            ranked_indices,
            start=1
        ):

            metadata = self.metadata[idx]

            result = {
                "rank": rank,

                "score": float(
                    scores[idx]
                ),

                "lexical_score": float(
                    lexical_scores[idx]
                ),

                "intent_adjusted_score": float(
                    scores[idx]
                ),

                "conversation_id": metadata.get(
                    "conversation_id"
                ),

                "root_tweet_id": metadata.get(
                    "root_tweet_id"
                ),

                "customer_message": metadata.get(
                    "customer_message",
                    ""
                ),

                "historical_amazonhelp_response": metadata.get(
                    "historical_amazonhelp_response",
                    ""
                ),

                "all_amazonhelp_responses": metadata.get(
                    "all_amazonhelp_responses",
                    []
                ),

                "num_messages": metadata.get(
                    "num_messages"
                ),

                # Diagnostics make the intent-aware reranking auditable and
                # become part of the LLM-judge cache input when available.
                "candidate_intent": candidate_intents.get(idx),

                "intent_compatible": (
                    candidate_intents.get(idx) == predicted_intent
                    if candidate_intents.get(idx) is not None
                    and predicted_intent
                    else None
                ),
            }

            results.append(result)

        return results



# SIMPLE FUNCTION API


def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    index_path=INDEX_PATH,
    predicted_intent: str = None,
):
    """
    Convenience function.

    Example
    -------
    results = retrieve(
        "My package says delivered but I haven't received it",
        top_k=5
    )
    """

    retriever = AmazonHelpRetriever(
        index_path
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        predicted_intent=predicted_intent,
    )



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP RETRIEVAL TEST")
    print("=" * 70)

    print("\nLoading retrieval index...")

    retriever = AmazonHelpRetriever()

    print(
        f"Documents loaded : {len(retriever.documents):,}"
    )

    print(
        f"TF-IDF features  : {retriever.matrix.shape[1]:,}"
    )


    # Test queries


    test_queries = [

        "My package says delivered but I haven't received it",

        "My Prime Video is not working",

        "I want a refund for my order",

    ]


    # Run retrieval


    for query in test_queries:

        print("\n" + "-" * 70)

        print("QUERY:")
        print(query)

        results = retriever.retrieve(
            query=query,
            top_k=3,
        )

        if not results:
            print("\nNo results found.")
            continue

        print("\nTOP RESULTS:")

        for result in results:

            print("\n" + "-" * 50)

            print(
                f"Rank       : {result['rank']}"
            )

            print(
                f"Similarity : {result['score']:.4f}"
            )

            print(
                f"Conversation: "
                f"{result['conversation_id']}"
            )

            print("\nCustomer:")

            print(
                result["customer_message"]
            )

            print("\nAmazonHelp:")

            print(
                result[
                    "historical_amazonhelp_response"
                ]
            )

    print("\n" + "=" * 70)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 70)



# ENTRY POINT


if __name__ == "__main__":
    main()
