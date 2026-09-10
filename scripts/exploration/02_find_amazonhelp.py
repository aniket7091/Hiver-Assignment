"""Exploratory identification of the AmazonHelp author ID."""

import pandas as pd

FILE_PATH = "data/raw/twcs.csv"

print("Finding AmazonHelp author ID...")

# First collect tweets that mention @AmazonHelp
amazon_mentions = []

for chunk in pd.read_csv(FILE_PATH, chunksize=100_000):
    mask = chunk["text"].str.contains(
        "@AmazonHelp",
        case=False,
        na=False
    )

    matches = chunk[mask]

    if not matches.empty:
        amazon_mentions.append(matches)

df = pd.concat(amazon_mentions, ignore_index=True)

print("\nTotal tweets mentioning @AmazonHelp:", len(df))

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nAuthor IDs grouped by inbound:")
print(
    df.groupby(["inbound", "author_id"])
      .size()
      .sort_values(ascending=False)
      .head(30)
)

print("\nTweets with inbound=False:")
print(
    df[df["inbound"] == False][
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text"
        ]
    ]
    .head(30)
    .to_string(index=False)
)
