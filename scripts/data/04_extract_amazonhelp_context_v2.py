"""Second-generation AmazonHelp context extraction used by reconstruction."""

import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
AMAZON_FILE = "data/processed/amazon_help_tweets.csv"
OUTPUT_FILE = "data/processed/amazon_help_context.csv"

CHUNK_SIZE = 100_000

DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "inbound": "string",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string"
}

os.makedirs("data/processed", exist_ok=True)

print("=" * 60)
print("STEP 1: Loading AmazonHelp tweets")
print("=" * 60)

amazon_df = pd.read_csv(
    AMAZON_FILE,
    dtype=DTYPES,
    low_memory=False
)

amazon_df["tweet_id"] = amazon_df["tweet_id"].str.strip()
amazon_df["in_response_to_tweet_id"] = (
    amazon_df["in_response_to_tweet_id"].str.strip()
)

amazon_ids = set(
    amazon_df["tweet_id"].dropna()
)

parent_target_ids = set(
    amazon_df["in_response_to_tweet_id"].dropna()
)

print(f"AmazonHelp tweets: {len(amazon_ids):,}")
print(f"Potential customer parent tweets: {len(parent_target_ids):,}")


# ------------------------------------------------------------
# STEP 2
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: Finding customer parent tweets")
print("=" * 60)

parent_ids = set()
chunk_number = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype=DTYPES,
    low_memory=False
):

    chunk_number += 1

    chunk["tweet_id"] = chunk["tweet_id"].str.strip()

    matches = chunk[
        chunk["tweet_id"].isin(parent_target_ids)
    ]

    if not matches.empty:
        parent_ids.update(
            matches["tweet_id"].dropna()
        )

    print(
        f"Processed chunk {chunk_number} | "
        f"Customer parents found: {len(parent_ids):,}"
    )

print(
    f"\nDirect customer parent tweets found: "
    f"{len(parent_ids):,}"
)


# ------------------------------------------------------------
# STEP 3
# ------------------------------------------------------------

required_ids = amazon_ids | parent_ids

print("\n" + "=" * 60)
print("STEP 3: Preparing extraction")
print("=" * 60)

print(f"AmazonHelp tweets:          {len(amazon_ids):,}")
print(f"Customer parent tweets:      {len(parent_ids):,}")
print(f"Total unique tweets needed:  {len(required_ids):,}")


# ------------------------------------------------------------
# STEP 4
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Extracting relevant tweets")
print("=" * 60)

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

first_write = True
total_extracted = 0
chunk_number = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype=DTYPES,
    low_memory=False
):

    chunk_number += 1

    chunk["tweet_id"] = chunk["tweet_id"].str.strip()

    matches = chunk[
        chunk["tweet_id"].isin(required_ids)
    ].copy()

    if not matches.empty:

        matches = matches.drop_duplicates(
            subset=["tweet_id"]
        )

        matches.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

        total_extracted += len(matches)

    print(
        f"Processed chunk {chunk_number} | "
        f"Extracted: {total_extracted:,}"
    )


# ------------------------------------------------------------
# STEP 5
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: Extraction summary")
print("=" * 60)

print(f"Expected tweets:   {len(required_ids):,}")
print(f"Extracted tweets:  {total_extracted:,}")

if total_extracted == len(required_ids):
    print("✅ Extraction count matches expected.")
else:
    print(
        "⚠️ Extraction count does NOT match expected."
    )

print(f"\nSaved to:")
print(OUTPUT_FILE)
