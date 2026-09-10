import pandas as pd
import os

INPUT_FILE = "data/processed/amazon_help_context.csv"

CHUNK_SIZE = 50_000

DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "inbound": "string",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string"
}

print("=" * 60)
print("VALIDATING AMAZONHELP CONTEXT")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    exit(1)

total_rows = 0
amazon_count = 0
customer_count = 0

unique_ids = set()
duplicate_ids = set()

inbound_true = 0
inbound_false = 0
inbound_other = 0
inbound_null = 0

chunk_number = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype=DTYPES,
    engine="python"
):
    chunk_number += 1

    chunk["tweet_id"] = chunk["tweet_id"].str.strip()
    chunk["author_id"] = chunk["author_id"].str.strip()

    total_rows += len(chunk)

    # --------------------------------------------------
    # AmazonHelp / Customer count
    # --------------------------------------------------

    amazon_mask = (
        chunk["author_id"]
        .str.lower()
        .eq("amazonhelp")
    )

    amazon_count += int(amazon_mask.sum())
    customer_count += int((~amazon_mask).sum())

    # --------------------------------------------------
    # Duplicate tweet IDs
    # --------------------------------------------------

    ids = chunk["tweet_id"].dropna().tolist()

    for tweet_id in ids:
        if tweet_id in unique_ids:
            duplicate_ids.add(tweet_id)
        else:
            unique_ids.add(tweet_id)

    # --------------------------------------------------
    # Inbound validation
    # --------------------------------------------------

    inbound = chunk["inbound"].astype("string").str.strip().str.lower()

    inbound_true += int((inbound == "true").sum())
    inbound_false += int((inbound == "false").sum())

    null_mask = inbound.isna() | (inbound == "")
    inbound_null += int(null_mask.sum())

    valid_mask = (
        (inbound == "true") |
        (inbound == "false") |
        null_mask
    )

    inbound_other += int((~valid_mask).sum())

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print(f"Total rows:              {total_rows:,}")
print(f"AmazonHelp tweets:       {amazon_count:,}")
print(f"Customer tweets:         {customer_count:,}")
print(f"Unique tweet IDs:        {len(unique_ids):,}")
print(f"Duplicate tweet IDs:     {len(duplicate_ids):,}")

print("\nInbound distribution:")
print(f"True:                    {inbound_true:,}")
print(f"False:                   {inbound_false:,}")
print(f"Null:                    {inbound_null:,}")
print(f"Other values:            {inbound_other:,}")

print("\n" + "=" * 60)
print("EXPECTED VALUES")
print("=" * 60)

print("Total rows:              324,816")
print("AmazonHelp tweets:       169,840")
print("Customer tweets:         154,976")

print("\n" + "=" * 60)
print("CHECKS")
print("=" * 60)

if total_rows == 324816:
    print("✅ Row count matches expected value.")
else:
    print(
        f"⚠️ Row count mismatch: "
        f"expected 324,816, got {total_rows:,}"
    )

if amazon_count == 169840:
    print("✅ AmazonHelp count matches expected value.")
else:
    print(
        f"⚠️ AmazonHelp count mismatch: "
        f"expected 169,840, got {amazon_count:,}"
    )

if customer_count == 154976:
    print("✅ Customer count matches expected value.")
else:
    print(
        f"⚠️ Customer count mismatch: "
        f"expected 154,976, got {customer_count:,}"
    )

if len(duplicate_ids) == 0:
    print("✅ No duplicate tweet IDs found.")
else:
    print(
        f"⚠️ Duplicate tweet IDs found: "
        f"{len(duplicate_ids):,}"
    )

if inbound_other == 0:
    print("✅ Inbound column contains only True/False/Null.")
else:
    print(
        f"⚠️ Found {inbound_other:,} unexpected inbound values."
    )

print("\nValidation finished.")