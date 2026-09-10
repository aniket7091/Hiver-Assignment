"""Extract AmazonHelp-authored tweets from the raw TWCS data."""

import pandas as pd
import os

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/amazon_help_tweets.csv"

os.makedirs("data/processed", exist_ok=True)

print("Extracting AmazonHelp tweets...")

total = 0
first_write = True

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=100_000,
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "inbound": "boolean",
        "created_at": "string",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string"
    },
    low_memory=False
):

    mask = (
        chunk["author_id"]
        .str.strip()
        .str.lower()
        .eq("amazonhelp")
    )

    matches = chunk[mask]

    if len(matches) > 0:

        matches.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False
        total += len(matches)

        print(
            f"Extracted so far: {total:,}"
        )

print("\nExtraction complete!")
print(f"AmazonHelp tweets: {total:,}")
print(f"Saved to: {OUTPUT_FILE}")
