"""Validate the generated AmazonHelp thread artifact."""

import json
import os
from collections import Counter

INPUT_FILE = "data/processed/amazon_help_threads.jsonl"

print("=" * 60)
print("VALIDATING RECONSTRUCTED THREADS")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    exit(1)

threads = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        threads.append(json.loads(line))

print(f"Threads loaded: {len(threads):,}")


# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

message_counts = Counter()
amazon_counts = Counter()
customer_counts = Counter()

invalid_threads = 0
ordering_errors = 0
threads_without_customer = 0
threads_without_amazon = 0

for thread in threads:

    messages = thread.get("messages", [])

    message_counts[len(messages)] += 1

    amazon = [
        m for m in messages
        if m.get("is_amazon") is True
    ]

    customer = [
        m for m in messages
        if m.get("is_amazon") is False
    ]

    amazon_counts[len(amazon)] += 1
    customer_counts[len(customer)] += 1

    if len(messages) == 0:
        invalid_threads += 1

    if len(amazon) == 0:
        threads_without_amazon += 1

    if len(customer) == 0:
        threads_without_customer += 1

    # Check chronological ordering
    dates = [
        m.get("created_at")
        for m in messages
        if m.get("created_at")
    ]

    if dates != sorted(dates):
        ordering_errors += 1


# ------------------------------------------------------------
# Print statistics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("THREAD QUALITY")
print("=" * 60)

print(f"Invalid empty threads:       {invalid_threads:,}")
print(f"Threads without customer:    {threads_without_customer:,}")
print(f"Threads without AmazonHelp:  {threads_without_amazon:,}")
print(f"Chronological errors:        {ordering_errors:,}")


print("\n" + "=" * 60)
print("THREAD LENGTH DISTRIBUTION")
print("=" * 60)

for length, count in sorted(message_counts.items()):

    print(
        f"{length:>3} messages: "
        f"{count:,} threads"
    )


print("\n" + "=" * 60)
print("AMAZONHELP MESSAGES PER THREAD")
print("=" * 60)

for count, number_threads in sorted(amazon_counts.items()):

    print(
        f"{count:>3} AmazonHelp messages: "
        f"{number_threads:,} threads"
    )


print("\n" + "=" * 60)
print("CUSTOMER MESSAGES PER THREAD")
print("=" * 60)

for count, number_threads in sorted(customer_counts.items()):

    print(
        f"{count:>3} customer messages: "
        f"{number_threads:,} threads"
    )


# ------------------------------------------------------------
# Show representative examples
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("REPRESENTATIVE THREADS")
print("=" * 60)

# Pick threads of different lengths
selected = {}

for thread in threads:

    length = len(thread["messages"])

    if length not in selected:
        selected[length] = thread

    if len(selected) >= 5:
        break


for length, thread in sorted(selected.items()):

    print("\n" + "-" * 60)
    print(
        f"Thread: {thread['conversation_id']} "
        f"({length} messages)"
    )

    for message in thread["messages"]:

        speaker = (
            "AmazonHelp"
            if message["is_amazon"]
            else "Customer"
        )

        text = message.get("text", "")

        if len(text) > 180:
            text = text[:180] + "..."

        print(f"\n[{speaker}]")
        print(text)


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)

if (
    invalid_threads == 0
    and threads_without_customer == 0
    and threads_without_amazon == 0
    and ordering_errors == 0
):
    print("✅ Thread structure looks valid.")
else:
    print("⚠️ Some thread-quality issues were detected.")

print("\nValidation complete.")
