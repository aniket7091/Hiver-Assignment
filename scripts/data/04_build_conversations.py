"""Historical context-extraction attempt retained for reproducibility."""

import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/raw/twcs.csv"
AMAZON_FILE = "data/processed/amazon_help_tweets.csv"
OUTPUT_FILE = "data/processed/amazon_help_context.csv"

CHUNK_SIZE = 100_000

os.makedirs("data/processed", exist_ok=True)


# ============================================================
# DATA TYPES
# ============================================================

DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "inbound": "boolean",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string"
}


# ============================================================
# STEP 1: LOAD AMAZONHELP TWEETS
# ============================================================

print("=" * 60)
print("STEP 1: Loading AmazonHelp tweets")
print("=" * 60)

amazon_df = pd.read_csv(
    AMAZON_FILE,
    dtype=DTYPES,
    low_memory=False
)

# Normalize tweet IDs
amazon_df["tweet_id"] = (
    amazon_df["tweet_id"]
    .str.strip()
)

amazon_df["in_response_to_tweet_id"] = (
    amazon_df["in_response_to_tweet_id"]
    .str.strip()
)

# AmazonHelp tweet IDs
amazon_ids = set(
    amazon_df["tweet_id"]
    .dropna()
    .tolist()
)

# IDs of tweets that AmazonHelp replied to
parent_target_ids = set(
    amazon_df["in_response_to_tweet_id"]
    .dropna()
    .tolist()
)

print(f"AmazonHelp tweets: {len(amazon_ids):,}")
print(f"Potential customer parent tweets: {len(parent_target_ids):,}")


# ============================================================
# STEP 2: FIND CUSTOMER PARENT TWEETS
# ============================================================

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

    # Normalize tweet IDs
    chunk["tweet_id"] = (
        chunk["tweet_id"]
        .str.strip()
    )

    # Find tweets that AmazonHelp directly replied to
    matches = chunk[
        chunk["tweet_id"].isin(parent_target_ids)
    ]

    if not matches.empty:

        parent_ids.update(
            matches["tweet_id"]
            .dropna()
            .tolist()
        )

    print(
        f"Processed chunk {chunk_number} | "
        f"Customer parents found: {len(parent_ids):,}"
    )


print(
    f"\nDirect customer parent tweets found: "
    f"{len(parent_ids):,}"
)


# ============================================================
# STEP 3: COMBINE REQUIRED TWEET IDs
# ============================================================

required_ids = amazon_ids | parent_ids

print("\n" + "=" * 60)
print("STEP 3: Preparing extraction")
print("=" * 60)

print(
    f"AmazonHelp tweets:        {len(amazon_ids):,}"
)

print(
    f"Customer parent tweets:    {len(parent_ids):,}"
)

print(
    f"Total unique tweets needed: {len(required_ids):,}"
)


# ============================================================
# STEP 4: EXTRACT RELEVANT TWEETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Extracting relevant tweets")
print("=" * 60)

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

    # Normalize tweet IDs
    chunk["tweet_id"] = (
        chunk["tweet_id"]
        .str.strip()
    )

    matches = chunk[
        chunk["tweet_id"].isin(required_ids)
    ]

    if not matches.empty:

        # Remove duplicate IDs inside this chunk
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


# ============================================================
# STEP 5: VALIDATE OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Validating extracted data")
print("=" * 60)

if not os.path.exists(OUTPUT_FILE):

    print("ERROR: Output file was not created.")

    exit(1)


df = pd.read_csv(
    OUTPUT_FILE,
    dtype=DTYPES,
    low_memory=False
)

# Normalize again
df["tweet_id"] = (
    df["tweet_id"]
    .str.strip()
)

df["author_id"] = (
    df["author_id"]
    .str.strip()
)


# Remove duplicates
before = len(df)

df = df.drop_duplicates(
    subset=["tweet_id"]
)

duplicates_removed = before - len(df)

# Save cleaned result
df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

amazon_count = (
    df["author_id"]
    .str.lower()
    .eq("amazonhelp")
    .sum()
)

customer_count = len(df) - amazon_count

unique_tweets = df["tweet_id"].nunique()

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Total extracted tweets:      {len(df):,}"
)

print(
    f"AmazonHelp tweets:            {amazon_count:,}"
)

print(
    f"Customer tweets:              {customer_count:,}"
)

print(
    f"Unique tweet IDs:             {unique_tweets:,}"
)

print(
    f"Duplicates removed:           {duplicates_removed:,}"
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)


# ============================================================
# INBOUND DISTRIBUTION
# ============================================================

print("\nInbound distribution:")

print(
    df["inbound"]
    .value_counts(dropna=False)
)


# ============================================================
# SAMPLE
# ============================================================

print("\nSample tweets:")

print(
    df[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id"
        ]
    ]
    .head(15)
    .to_string(index=False)
)
