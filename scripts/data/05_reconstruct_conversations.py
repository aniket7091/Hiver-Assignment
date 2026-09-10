"""Reconstruct ordered AmazonHelp conversation threads."""

import pandas as pd
import json
import os
from collections import defaultdict

INPUT_FILE = "data/processed/amazon_help_context.csv"
OUTPUT_FILE = "data/processed/amazon_help_threads.jsonl"

CHUNK_SIZE = 50_000


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading AmazonHelp context")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    exit(1)

# Use strings first.
# We will normalize inbound ourselves.
DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "inbound": "string",
    "created_at": "string",
    "text": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string"
}

chunks = []

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    dtype=DTYPES,
    engine="python"
):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

print(f"Loaded rows: {len(df):,}")


# ============================================================
# STEP 2: CLEAN BASIC FIELDS
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning fields")
print("=" * 60)

for col in [
    "tweet_id",
    "author_id",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]:
    df[col] = df[col].astype("string").str.strip()


# Normalize inbound

df["inbound"] = (
    df["inbound"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df["is_amazon"] = (
    df["author_id"]
    .str.lower()
    .eq("amazonhelp")
)

df["is_customer"] = ~df["is_amazon"]

print(
    f"AmazonHelp messages: {df['is_amazon'].sum():,}"
)

print(
    f"Customer messages:    {df['is_customer'].sum():,}"
)


# ============================================================
# STEP 3: REMOVE DUPLICATE TWEETS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Removing duplicate tweet IDs")
print("=" * 60)

before = len(df)

df = df.drop_duplicates(
    subset=["tweet_id"],
    keep="first"
).copy()

duplicates_removed = before - len(df)

print(f"Rows before:           {before:,}")
print(f"Duplicates removed:    {duplicates_removed:,}")
print(f"Rows after:            {len(df):,}")


# ============================================================
# STEP 4: CREATE TWEET LOOKUP
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Building tweet lookup")
print("=" * 60)

tweet_lookup = {}

for row in df.itertuples(index=False):

    tweet_id = str(row.tweet_id)

    tweet_lookup[tweet_id] = {
        "tweet_id": tweet_id,
        "author_id": str(row.author_id),
        "inbound": str(row.inbound),
        "created_at": str(row.created_at),
        "text": str(row.text),
        "response_tweet_id": (
            None
            if pd.isna(row.response_tweet_id)
            else str(row.response_tweet_id)
        ),
        "in_response_to_tweet_id": (
            None
            if pd.isna(row.in_response_to_tweet_id)
            else str(row.in_response_to_tweet_id)
        ),
        "is_amazon": bool(
            str(row.author_id).lower() == "amazonhelp"
        )
    }

print(
    f"Tweet lookup created: {len(tweet_lookup):,}"
)


# ============================================================
# STEP 5: BUILD CHILD RELATIONSHIPS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Building reply relationships")
print("=" * 60)

children = defaultdict(list)

for tweet_id, tweet in tweet_lookup.items():

    parent_id = tweet["in_response_to_tweet_id"]

    if parent_id and parent_id in tweet_lookup:
        children[parent_id].append(tweet_id)

print(
    f"Tweets with known parent: {sum(len(v) for v in children.values()):,}"
)


# ============================================================
# STEP 6: FIND AMAZONHELP → CUSTOMER CONNECTIONS
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: Identifying conversation entry points")
print("=" * 60)

# Every AmazonHelp tweet that directly replies to a customer
# can serve as an anchor for a conversation.

anchors = []

for tweet_id, tweet in tweet_lookup.items():

    if not tweet["is_amazon"]:
        continue

    parent_id = tweet["in_response_to_tweet_id"]

    if not parent_id:
        continue

    if parent_id not in tweet_lookup:
        continue

    parent = tweet_lookup[parent_id]

    if parent["is_amazon"]:
        continue

    anchors.append(tweet_id)

print(
    f"AmazonHelp replies with customer parent: {len(anchors):,}"
)


# ============================================================
# STEP 7: FIND ROOT OF EACH THREAD
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Finding conversation roots")
print("=" * 60)


def find_root(tweet_id):
    """
    Follow in_response_to_tweet_id backwards until:
      - there is no parent
      - parent is not available
      - a cycle is detected
    """

    current_id = tweet_id
    visited = set()

    while True:

        if current_id in visited:
            return current_id

        visited.add(current_id)

        current = tweet_lookup.get(current_id)

        if current is None:
            return current_id

        parent_id = current["in_response_to_tweet_id"]

        if not parent_id:
            return current_id

        if parent_id not in tweet_lookup:
            return current_id

        current_id = parent_id


root_to_messages = defaultdict(set)

for anchor_id in anchors:

    root_id = find_root(anchor_id)

    root_to_messages[root_id].add(anchor_id)

print(
    f"Unique conversation roots discovered: "
    f"{len(root_to_messages):,}"
)


# ============================================================
# STEP 8: EXPAND THREADS FORWARD
# ============================================================

print("\n" + "=" * 60)
print("STEP 8: Expanding conversation threads")
print("=" * 60)


def collect_thread(root_id):
    """
    Starting from root, follow all known child replies.
    """

    collected = []
    visited = set()
    queue = [root_id]

    while queue:

        current_id = queue.pop(0)

        if current_id in visited:
            continue

        visited.add(current_id)

        tweet = tweet_lookup.get(current_id)

        if tweet is None:
            continue

        collected.append(tweet)

        for child_id in children.get(current_id, []):
            if child_id not in visited:
                queue.append(child_id)

    return collected


threads = []

for index, root_id in enumerate(root_to_messages.keys(), start=1):

    messages = collect_thread(root_id)

    # Keep only threads that contain AmazonHelp.
    has_amazon = any(
        message["is_amazon"]
        for message in messages
    )

    if not has_amazon:
        continue

    # Sort chronologically.
    messages.sort(
        key=lambda x: x["created_at"]
    )

    threads.append({
        "conversation_id": f"amazonhelp_{index}",
        "root_tweet_id": root_id,
        "messages": messages
    })

    if index % 10_000 == 0:
        print(
            f"Processed roots: {index:,} | "
            f"Threads: {len(threads):,}"
        )


print(
    f"\nFinal threads: {len(threads):,}"
)


# ============================================================
# STEP 9: THREAD STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("STEP 9: Thread statistics")
print("=" * 60)

single_message = 0
multi_message = 0
total_messages = 0

threads_with_customer = 0
threads_with_multiple_amazon = 0

for thread in threads:

    messages = thread["messages"]

    total_messages += len(messages)

    if len(messages) == 1:
        single_message += 1
    else:
        multi_message += 1

    customer_messages = sum(
        not message["is_amazon"]
        for message in messages
    )

    amazon_messages = sum(
        message["is_amazon"]
        for message in messages
    )

    if customer_messages > 0:
        threads_with_customer += 1

    if amazon_messages > 1:
        threads_with_multiple_amazon += 1


print(f"Threads:                         {len(threads):,}")
print(f"Total messages in threads:       {total_messages:,}")
print(f"Single-message threads:          {single_message:,}")
print(f"Multi-message threads:           {multi_message:,}")
print(f"Threads with customer messages:  {threads_with_customer:,}")
print(
    f"Threads with multiple AmazonHelp "
    f"messages:                       {threads_with_multiple_amazon:,}"
)


if len(threads) > 0:

    avg_messages = total_messages / len(threads)

    print(
        f"Average messages per thread:    "
        f"{avg_messages:.2f}"
    )


# ============================================================
# STEP 10: SAVE JSONL
# ============================================================

print("\n" + "=" * 60)
print("STEP 10: Saving reconstructed threads")
print("=" * 60)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for thread in threads:

        f.write(
            json.dumps(
                thread,
                ensure_ascii=False
            )
            + "\n"
        )

print(f"Saved to: {OUTPUT_FILE}")


# ============================================================
# STEP 11: SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 60)
print("STEP 11: Example conversations")
print("=" * 60)

shown = 0

for thread in threads:

    messages = thread["messages"]

    if len(messages) < 2:
        continue

    print("\n" + "-" * 60)
    print(
        f"Conversation: {thread['conversation_id']}"
    )
    print(
        f"Root tweet: {thread['root_tweet_id']}"
    )

    for message in messages[:10]:

        speaker = (
            "AmazonHelp"
            if message["is_amazon"]
            else "Customer"
        )

        text = message["text"]

        if len(text) > 250:
            text = text[:250] + "..."

        print(
            f"\n[{speaker}] "
            f"{message['tweet_id']}"
        )

        print(text)

    shown += 1

    if shown >= 5:
        break


print("\n" + "=" * 60)
print("RECONSTRUCTION COMPLETE")
print("=" * 60)
