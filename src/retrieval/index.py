"""
Build a TF-IDF retrieval index over historical AmazonHelp conversations.

Input:
    data/processed/amazon_help_threads.jsonl

Output:
    data/processed/retrieval_index.pkl

The index stores:
    - TF-IDF vectorizer
    - TF-IDF matrix
    - searchable customer messages
    - corresponding AmazonHelp responses
    - thread metadata
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer



# PATHS


ROOT_DIR = Path(__file__).resolve().parents[2]

THREADS_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "amazon_help_threads.jsonl"
)

INDEX_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "retrieval_index.pkl"
)



# CONFIG


MAX_FEATURES = 30000
MIN_DF = 2
MAX_DF = 0.95

NGRAM_RANGE = (1, 2)



# TEXT CLEANING


def clean_text(text):
    """
    Basic normalization for retrieval.

    We intentionally keep most words because customer messages
    contain useful product, issue and support terminology.
    """

    if not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Remove HTML entities
    text = re.sub(
        r"&\w+;",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()



# LOAD THREADS


def load_threads():
    if not THREADS_FILE.exists():
        raise FileNotFoundError(
            f"Threads file not found:\n{THREADS_FILE}"
        )

    threads = []

    with open(
        THREADS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                thread = json.loads(line)
                threads.append(thread)

            except json.JSONDecodeError:
                continue

    return threads



# EXTRACT RETRIEVAL DOCUMENTS


def build_documents(threads):
    """
    Build retrieval pairs from reconstructed AmazonHelp threads.

    For each thread:
        - first customer message = query/document
        - first AmazonHelp response after it = historical resolution

    The complete AmazonHelp response history is also preserved
    in metadata for later inspection.
    """

    documents = []
    metadata = []

    skipped = 0

    for thread in threads:

        messages = thread.get("messages", [])

        if not messages:
            skipped += 1
            continue

        customer_message = None
        amazonhelp_messages = []

        for message in messages:

            text = str(
                message.get("text", "")
            ).strip()

            if not text:
                continue

            # Your JSONL uses is_amazon=True for AmazonHelp
            # and is_amazon=False for customers.
            is_amazon = message.get(
                "is_amazon",
                False
            )

            if is_amazon:
                amazonhelp_messages.append(text)

            else:
                # First customer message in the thread
                if customer_message is None:
                    customer_message = text

        # Need both sides of the conversation
        if not customer_message:
            skipped += 1
            continue

        if not amazonhelp_messages:
            skipped += 1
            continue

        cleaned_customer = clean_text(
            customer_message
        )

        if not cleaned_customer:
            skipped += 1
            continue

        # First AmazonHelp response is the historical
        # response associated with the initial customer issue.
        historical_response = amazonhelp_messages[0]

        documents.append(cleaned_customer)

        metadata.append(
            {
                "conversation_id": thread.get(
                    "conversation_id",
                    str(len(metadata)),
                ),
                "root_tweet_id": thread.get(
                    "root_tweet_id"
                ),
                "customer_message": customer_message,
                "historical_amazonhelp_response": historical_response,
                "all_amazonhelp_responses": amazonhelp_messages,
                "num_messages": len(messages),
            }
        )

    return documents, metadata, skipped



# BUILD TF-IDF INDEX


def build_index(documents):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    return vectorizer, matrix



# SAVE INDEX


def save_index(
    vectorizer,
    matrix,
    metadata,
):
    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": metadata,
        "config": {
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "ngram_range": NGRAM_RANGE,
        },
    }

    with open(
        INDEX_FILE,
        "wb",
    ) as f:

        pickle.dump(
            index,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )



# MAIN


def main():

    print("=" * 70)
    print("AMAZONHELP RETRIEVAL INDEX")
    print("=" * 70)


    # Load


    print("\nLoading historical threads...")

    threads = load_threads()

    print(
        f"Threads loaded : {len(threads):,}"
    )


    # Extract documents


    print(
        "\nExtracting customer/resolution pairs..."
    )

    documents, metadata, skipped = build_documents(
        threads
    )

    print(
        f"Retrieval documents : {len(documents):,}"
    )

    print(
        f"Skipped threads     : {skipped:,}"
    )

    if not documents:
        raise RuntimeError(
            "No retrieval documents were created."
        )


    # Build TF-IDF


    print(
        "\nBuilding TF-IDF index..."
    )

    vectorizer, matrix = build_index(
        documents
    )

    print(
        f"Matrix shape : {matrix.shape}"
    )

    print(
        f"Vocabulary   : {len(vectorizer.vocabulary_):,}"
    )


    # Save


    print(
        "\nSaving index..."
    )

    save_index(
        vectorizer,
        matrix,
        metadata,
    )

    print(
        f"Saved to: {INDEX_FILE}"
    )


    # Summary


    print("\n")
    print("=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Documents : {matrix.shape[0]:,}"
    )

    print(
        f"Features  : {matrix.shape[1]:,}"
    )

    print(
        f"File      : {INDEX_FILE}"
    )


if __name__ == "__main__":
    main()