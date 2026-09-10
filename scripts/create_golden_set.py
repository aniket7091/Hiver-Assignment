import json
import random
import re
from pathlib import Path
from collections import Counter



# Configuration


RANDOM_SEED = 42
TARGET_SIZE = 200

INPUT_FILE = Path("data/processed/amazon_help_threads.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.jsonl")



# Text cleaning


def clean_text(text):
    """Light cleaning for sampling/search only."""
    if not text:
        return ""

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



# Heuristic intent signals

# These are NOT the final labels.
# They are only used to make the golden set more diverse.
# The final intent must be assigned manually.

INTENT_KEYWORDS = {
    "delivery_issue": [
        "delivered",
        "delivery",
        "package",
        "parcel",
        "not received",
        "didn't receive",
        "did not receive",
        "where is my package",
        "where's my package",
        "missing package",
        "missing parcel",
        "delivery date",
        "delivery delayed",
        "late delivery",
    ],
    "order_issue": [
        "order",
        "order number",
        "cancel my order",
        "cancel order",
        "wrong order",
        "order status",
        "ordered",
    ],
    "shipping_issue": [
        "shipping",
        "ship",
        "shipped",
        "prime shipping",
        "two day",
        "2 day",
        "next day",
        "shipping speed",
        "shipping date",
    ],
    "payment_issue": [
        "payment",
        "paid",
        "charge",
        "charged",
        "billing",
        "credit card",
        "debit card",
        "payment method",
        "transaction",
        "money charged",
    ],
    "refund_issue": [
        "refund",
        "refunded",
        "money back",
        "refund status",
        "return my money",
        "reimbursement",
    ],
    "account_issue": [
        "account",
        "login",
        "log in",
        "password",
        "locked",
        "account closed",
        "close my account",
        "account access",
    ],
    "prime_video_issue": [
        "prime video",
        "primevideo",
        "video",
        "streaming",
        "stream",
        "movie",
        "episode",
        "playback",
        "buffering",
        "buffer",
        "video error",
    ],
    "device_issue": [
        "fire tv",
        "firetv",
        "kindle",
        "echo",
        "alexa",
        "device",
        "tablet",
        "tv",
        "remote",
    ],
    "product_or_content_issue": [
        "product",
        "item",
        "listing",
        "seller",
        "book",
        "movie",
        "show",
        "content",
        "quality",
        "defective",
        "broken item",
    ],
    "support_complaint": [
        "customer service",
        "customer support",
        "support",
        "worst service",
        "terrible service",
        "bad service",
        "poor service",
        "no help",
        "nobody helped",
        "disappointed",
        "frustrated",
    ],
}



# Extract first customer message


def get_first_customer_message(thread):
    """
    Returns the first customer-authored message in a thread.
    """
    for message in thread.get("messages", []):
        author = str(message.get("author_id", "")).lower()

        # AmazonHelp's author_id is "amazonhelp"
        if author != "amazonhelp":
            text = message.get("text", "")
            if text:
                return text

    return ""



# Find historical AmazonHelp response


def get_first_amazon_response(thread):
    """
    Returns the first AmazonHelp response after the customer's
    initial message.

    This is useful later for historical-resolution grounding.
    """
    customer_seen = False

    for message in thread.get("messages", []):
        author = str(message.get("author_id", "")).lower()

        if author != "amazonhelp":
            if message.get("text"):
                customer_seen = True

        elif customer_seen:
            text = message.get("text", "")
            if text:
                return text

    return ""



# Heuristic category


def heuristic_categories(text):
    """
    Returns possible categories based on keyword matches.

    IMPORTANT:
    These are only sampling hints.
    They must NOT be treated as gold labels.
    """
    normalized = clean_text(text).lower()

    matches = []

    for category, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in normalized:
                matches.append(category)
                break

    return matches



# Difficulty estimation


def estimate_difficulty(text):
    """
    Rough difficulty estimate for sampling.

    Short/generic messages are often harder to classify,
    while highly specific messages are generally easier.
    """
    cleaned = clean_text(text)
    words = cleaned.split()

    if len(words) <= 4:
        return "hard"

    if len(words) <= 10:
        return "medium"

    return "easy"



# Load threads


def load_threads():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run the conversation reconstruction step first."
        )

    threads = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            threads.append(json.loads(line))

    return threads



# Build candidate examples


def build_candidates(threads):
    candidates = []

    for thread_index, thread in enumerate(threads):
        customer_text = get_first_customer_message(thread)

        if not customer_text:
            continue

        cleaned = clean_text(customer_text)

        if not cleaned:
            continue

        # Ignore extremely long messages for the golden set.
        # They make manual annotation unnecessarily expensive.
        if len(cleaned) > 1000:
            continue

        messages = thread.get("messages", [])

        categories = heuristic_categories(cleaned)

        difficulty = estimate_difficulty(cleaned)

        first_response = get_first_amazon_response(thread)

        # Keep a compact version of the thread context.
        context = []

        for message in messages[:8]:
            author = str(message.get("author_id", "")).lower()

            if author == "amazonhelp":
                role = "amazonhelp"
            else:
                role = "customer"

            text = clean_text(message.get("text", ""))

            if text:
                context.append({
                    "role": role,
                    "text": text,
                })

        candidates.append({
            "source_thread_index": thread_index,
            "thread_id": thread.get("thread_id"),
            "customer_message": cleaned,
            "historical_amazonhelp_response": clean_text(first_response),
            "thread_context": context,
            "heuristic_categories": categories,
            "difficulty": difficulty,
            "word_count": len(cleaned.split()),
        })

    return candidates



# Stratified sampling


def sample_candidates(candidates, target_size=200):
    """
    Create a diverse sample.

    We intentionally mix:
      - keyword-matched business issues
      - ambiguous/other examples
      - short hard examples
      - medium examples
      - longer examples

    The heuristic categories are never used as gold labels.
    """

    random.seed(RANDOM_SEED)

    selected = []
    selected_ids = set()


    # 1. Try to cover each discovered business category.


    categories = list(INTENT_KEYWORDS.keys())

    per_category = 12

    for category in categories:
        pool = [
            x
            for x in candidates
            if category in x["heuristic_categories"]
        ]

        random.shuffle(pool)

        count = 0

        for item in pool:
            if item["source_thread_index"] in selected_ids:
                continue

            selected.append(item)
            selected_ids.add(item["source_thread_index"])

            count += 1

            if count >= per_category:
                break


    # 2. Add hard/ambiguous examples.


    hard_pool = [
        x
        for x in candidates
        if x["difficulty"] == "hard"
        and x["source_thread_index"] not in selected_ids
    ]

    random.shuffle(hard_pool)

    for item in hard_pool[:40]:
        selected.append(item)
        selected_ids.add(item["source_thread_index"])


    # 3. Add examples with no heuristic category.


    other_pool = [
        x
        for x in candidates
        if not x["heuristic_categories"]
        and x["source_thread_index"] not in selected_ids
    ]

    random.shuffle(other_pool)

    for item in other_pool[:40]:
        selected.append(item)
        selected_ids.add(item["source_thread_index"])


    # 4. Fill remaining slots randomly.


    remaining = [
        x
        for x in candidates
        if x["source_thread_index"] not in selected_ids
    ]

    random.shuffle(remaining)

    needed = target_size - len(selected)

    if needed > 0:
        for item in remaining[:needed]:
            selected.append(item)
            selected_ids.add(item["source_thread_index"])

    # Safety
    selected = selected[:target_size]

    # Shuffle final ordering so categories are not grouped.
    random.shuffle(selected)

    return selected



# Convert to annotation format


def create_annotation_record(item, index):
    """
    Creates the actual golden-set record.

    Gold fields intentionally start empty.
    """

    return {
        "example_id": f"golden_{index:04d}",

        "customer_message": item["customer_message"],

        "thread_context": item["thread_context"],

        "historical_amazonhelp_response": (
            item["historical_amazonhelp_response"]
        ),


        # Human annotation fields


        "gold_intent": "",

        "gold_action": "",

        "gold_escalation_reason": "",


        # Annotation metadata


        "annotator_notes": "",


        # Sampling metadata

        # heuristic_categories are deliberately retained
        # separately from gold_intent.
        #
        # DO NOT use heuristic_categories as labels.


        "sampling_metadata": {
            "difficulty": item["difficulty"],
            "word_count": item["word_count"],
            "heuristic_categories": item["heuristic_categories"],
            "source_thread_index": item["source_thread_index"],
        },
    }



# Save JSONL


def save_jsonl(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )



# Main


def main():

    print("=" * 60)
    print("Creating AmazonHelp Golden Evaluation Set")
    print("=" * 60)

    print(f"\nInput : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Target size: {TARGET_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")

    # Load
    threads = load_threads()

    print(f"\nThreads loaded: {len(threads):,}")

    # Candidates
    candidates = build_candidates(threads)

    print(f"Candidate examples: {len(candidates):,}")

    if len(candidates) < TARGET_SIZE:
        raise ValueError(
            f"Only {len(candidates)} usable examples found, "
            f"but {TARGET_SIZE} are required."
        )

    # Sample
    sampled = sample_candidates(
        candidates,
        target_size=TARGET_SIZE
    )

    print(f"Sampled examples: {len(sampled):,}")

    # Annotation records
    records = []

    for index, item in enumerate(sampled, start=1):
        records.append(
            create_annotation_record(item, index)
        )

    # Save
    save_jsonl(records)


    # Summary


    difficulty_counts = Counter(
        r["sampling_metadata"]["difficulty"]
        for r in records
    )

    category_counts = Counter()

    for r in records:
        for category in r["sampling_metadata"]["heuristic_categories"]:
            category_counts[category] += 1

    print("\n" + "=" * 60)
    print("Golden set created successfully")
    print("=" * 60)

    print(f"\nSaved: {OUTPUT_FILE}")

    print("\nDifficulty distribution:")

    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty:10s}: {count}")

    print("\nHeuristic category coverage:")

    for category in INTENT_KEYWORDS:
        print(
            f"  {category:28s}: "
            f"{category_counts.get(category, 0)}"
        )

    print("\nIMPORTANT:")
    print(
        "heuristic_categories are sampling hints only."
    )

    print(
        "You must manually assign gold_intent, "
        "gold_action and gold_escalation_reason."
    )


if __name__ == "__main__":
    main()