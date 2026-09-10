import json
import os
import re

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer



# CONFIGURATION


INPUT_FILE = "data/processed/amazon_help_threads.jsonl"
OUTPUT_FILE = "data/processed/amazonhelp_intent_clusters.csv"

N_CLUSTERS = 15
MAX_FEATURES = 20_000
MIN_DF = 5
MAX_DF = 0.90

TOP_TERMS = 12
EXAMPLES_PER_CLUSTER = 8



# TEXT CLEANING


def clean_text(text: str) -> str:
    """
    Remove common Twitter-specific noise while preserving
    the words that are useful for intent discovery.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove HTML entities such as &amp; and &gt;
    text = re.sub(
        r"&\w+;",
        " ",
        text
    )

    # Remove standalone numbers
    text = re.sub(
        r"\b\d+\b",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



# LOAD THREADS


def load_threads(input_file: str) -> list:
    """
    Load reconstructed conversation threads from JSONL.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    threads = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            threads.append(
                json.loads(line)
            )

    return threads



# EXTRACT CUSTOMER ISSUES


def extract_customer_issues(threads: list) -> pd.DataFrame:
    """
    Extract the first customer message from every thread.

    The first customer message is treated as the initial
    support issue that started the conversation.
    """

    records = []

    for thread in threads:

        first_customer_message = None

        for message in thread.get("messages", []):

            if not message.get("is_amazon", False):

                text = message.get("text", "")

                if text and text.strip():

                    first_customer_message = text.strip()
                    break

        if first_customer_message:

            records.append(
                {
                    "conversation_id": thread.get(
                        "conversation_id"
                    ),
                    "root_tweet_id": thread.get(
                        "root_tweet_id"
                    ),
                    "text": first_customer_message
                }
            )

    return pd.DataFrame(records)



# CREATE TF-IDF REPRESENTATION


def create_tfidf(texts):

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix



# CLUSTER


def cluster_messages(matrix):

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(matrix)

    return model, labels



# TOP TERMS


def get_cluster_terms(model, vectorizer):

    terms = vectorizer.get_feature_names_out()

    order = model.cluster_centers_.argsort(
        axis=1
    )[:, ::-1]

    cluster_terms = {}

    for cluster_id in range(N_CLUSTERS):

        indices = order[
            cluster_id
        ][:TOP_TERMS]

        cluster_terms[cluster_id] = [
            terms[index]
            for index in indices
        ]

    return cluster_terms



# print cluster information
def print_cluster_analysis(
    df,
    cluster_terms
):

    print("\n" + "=" * 60)
    print("CLUSTER SIZES")
    print("=" * 60)

    counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster_id, count in counts.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"Cluster {cluster_id:>2}: "
            f"{count:>6,} "
            f"({percentage:>5.2f}%)"
        )

    print("\n" + "=" * 60)
    print("TOP TERMS + REPRESENTATIVE EXAMPLES")
    print("=" * 60)

    for cluster_id in range(N_CLUSTERS):

        cluster_df = df[
            df["cluster"] == cluster_id
        ].copy()

        print("\n" + "-" * 60)

        print(
            f"CLUSTER {cluster_id}"
        )

        print(
            f"Size: {len(cluster_df):,}"
        )

        print(
            "Top terms: "
            + ", ".join(
                cluster_terms[cluster_id]
            )
        )

        print("\nExamples:")

        # Shorter examples are easier to inspect manually.
        examples = (
            cluster_df
            .assign(
                text_length=cluster_df[
                    "clean_text"
                ].str.len()
            )
            .sort_values("text_length")
            .head(EXAMPLES_PER_CLUSTER)
        )

        for _, row in examples.iterrows():

            text = row["text"]

            if len(text) > 300:
                text = text[:300] + "..."

            print(
                f"\n[{row['conversation_id']}]"
            )

            print(text)



# MAIN


def main():

    print("=" * 60)
    print("INTENT DISCOVERY")
    print("=" * 60)


    # Step 1: Load threads


    print("\n" + "=" * 60)
    print("STEP 1: Loading reconstructed threads")
    print("=" * 60)

    threads = load_threads(
        INPUT_FILE
    )

    print(
        f"Threads loaded: {len(threads):,}"
    )


    # Step 2: Extract customer issues


    print("\n" + "=" * 60)
    print("STEP 2: Extracting customer issues")
    print("=" * 60)

    df = extract_customer_issues(
        threads
    )

    print(
        f"Customer issue messages: "
        f"{len(df):,}"
    )


    # Step 3: Clean text


    print("\n" + "=" * 60)
    print("STEP 3: Cleaning customer messages")
    print("=" * 60)

    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )

    df = df[
        df["clean_text"].str.len() >= 3
    ].reset_index(drop=True)

    print(
        f"Usable customer messages: "
        f"{len(df):,}"
    )


    # Step 4: TF-IDF


    print("\n" + "=" * 60)
    print("STEP 4: Creating TF-IDF representation")
    print("=" * 60)

    vectorizer, matrix = create_tfidf(
        df["clean_text"]
    )

    print(
        f"TF-IDF matrix shape: "
        f"{matrix.shape}"
    )


    # Step 5: K-Means


    print("\n" + "=" * 60)
    print("STEP 5: Clustering customer issues")
    print("=" * 60)

    model, labels = cluster_messages(
        matrix
    )

    df["cluster"] = labels

    print(
        f"Created {N_CLUSTERS} clusters."
    )


    # Step 6: Top terms


    cluster_terms = get_cluster_terms(
        model,
        vectorizer
    )


    # Step 7: Analysis


    print_cluster_analysis(
        df,
        cluster_terms
    )


    # Step 8: Save results


    print("\n" + "=" * 60)
    print("STEP 8: Saving clustering results")
    print("=" * 60)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df[
        [
            "conversation_id",
            "root_tweet_id",
            "text",
            "clean_text",
            "cluster"
        ]
    ].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


    # Final


    print("\n" + "=" * 60)
    print("INTENT DISCOVERY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()