"""Exploratory scan of the raw TWCS data for AmazonHelp mentions."""

import pandas as pd

FILE_PATH = "data/raw/twcs.csv"

print("Scanning dataset for AmazonHelp...")

amazon_rows = []

for chunk in pd.read_csv(FILE_PATH, chunksize=100_000):
    # Search text for AmazonHelp mentions
    mask = chunk["text"].str.contains(
        "@AmazonHelp",
        case=False,
        na=False
    )

    matches = chunk[mask]

    if not matches.empty:
        amazon_rows.append(matches)

if amazon_rows:
    amazon_df = pd.concat(amazon_rows, ignore_index=True)

    print("\nTweets mentioning @AmazonHelp:")
    print("Total:", len(amazon_df))

    print("\nSample tweets:")
    print(
        amazon_df[
            [
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nAuthor IDs:")
    print(amazon_df["author_id"].value_counts().head(20))

else:
    print("No tweets containing @AmazonHelp were found.")
