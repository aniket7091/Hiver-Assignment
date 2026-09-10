import json
from pathlib import Path
from collections import Counter


GOLDEN_FILE = Path("data/golden/golden_set.jsonl")

VALID_INTENTS = {
    "delivery_issue",
    "order_issue",
    "shipping_issue",
    "payment_issue",
    "refund_issue",
    "account_issue",
    "prime_video_issue",
    "device_issue",
    "product_or_content_issue",
    "support_complaint",
    "other",
}

VALID_ACTIONS = {
    "auto_handle",
    "escalate",
}


def is_labelled(record):
    intent = str(record.get("gold_intent", "")).strip()
    action = str(record.get("gold_action", "")).strip()

    return bool(intent and action)


def load_golden_set():
    records = []

    if not GOLDEN_FILE.exists():
        print(f"File not found: {GOLDEN_FILE}")
        return None

    with GOLDEN_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"Invalid JSON at line {line_number}")
                print(error)
                return None

    return records


def main():
    print("=" * 70)
    print("GOLDEN SET VERIFICATION")
    print("=" * 70)

    records = load_golden_set()

    if records is None:
        return

    total = len(records)

    labelled = [
        record
        for record in records
        if is_labelled(record)
    ]

    unlabelled = [
        record
        for record in records
        if not is_labelled(record)
    ]

    print()
    print("LABELING STATUS")
    print("-" * 70)

    print(f"Total examples : {total}")
    print(f"Labelled       : {len(labelled)}")
    print(f"Unlabelled     : {len(unlabelled)}")

    if total > 0:
        coverage = len(labelled) / total * 100
        print(f"Coverage       : {coverage:.2f}%")

    if unlabelled:
        print()
        print("Unlabelled examples:")
        print("-" * 70)

        for record in unlabelled:
            example_id = record.get("example_id", "UNKNOWN")
            print(example_id)


    # Intent distribution


    intent_counts = Counter()

    for record in labelled:
        intent = str(
            record.get("gold_intent", "")
        ).strip()

        intent_counts[intent] += 1

    print()
    print("INTENT DISTRIBUTION")
    print("-" * 70)

    if intent_counts:
        for intent, count in intent_counts.most_common():
            percentage = count / len(labelled) * 100

            print(
                f"{intent:<35}"
                f"{count:>5} "
                f"({percentage:>6.2f}%)"
            )
    else:
        print("No labelled examples.")


    # Action distribution


    action_counts = Counter()

    for record in labelled:
        action = str(
            record.get("gold_action", "")
        ).strip()

        action_counts[action] += 1

    print()
    print("ACTION DISTRIBUTION")
    print("-" * 70)

    if action_counts:
        for action, count in action_counts.most_common():
            percentage = count / len(labelled) * 100

            print(
                f"{action:<35}"
                f"{count:>5} "
                f"({percentage:>6.2f}%)"
            )
    else:
        print("No labelled examples.")


    # Invalid intent labels


    invalid_intents = []

    for record in labelled:
        intent = str(
            record.get("gold_intent", "")
        ).strip()

        if intent not in VALID_INTENTS:
            invalid_intents.append(
                (
                    record.get("example_id", "UNKNOWN"),
                    intent,
                )
            )


    # Invalid action labels


    invalid_actions = []

    for record in labelled:
        action = str(
            record.get("gold_action", "")
        ).strip()

        if action not in VALID_ACTIONS:
            invalid_actions.append(
                (
                    record.get("example_id", "UNKNOWN"),
                    action,
                )
            )


    # Invalid label report


    print()
    print("INVALID LABEL CHECK")
    print("-" * 70)

    if not invalid_intents:
        print("No invalid intent labels.")
    else:
        print(
            f"Invalid intent labels: "
            f"{len(invalid_intents)}"
        )

        for example_id, intent in invalid_intents:
            print(
                f"{example_id}: {intent!r}"
            )

    print()

    if not invalid_actions:
        print("No invalid action labels.")
    else:
        print(
            f"Invalid action labels: "
            f"{len(invalid_actions)}"
        )

        for example_id, action in invalid_actions:
            print(
                f"{example_id}: {action!r}"
            )


    # Final summary


    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Total examples  : {total}")
    print(f"Labelled        : {len(labelled)}")
    print(f"Unlabelled      : {len(unlabelled)}")
    print(f"Invalid intents : {len(invalid_intents)}")
    print(f"Invalid actions : {len(invalid_actions)}")

    if (
        len(unlabelled) == 0
        and not invalid_intents
        and not invalid_actions
    ):
        print()
        print("GOLDEN SET IS CLEAN")
    else:
        print()
        print("GOLDEN SET NEEDS ATTENTION")


if __name__ == "__main__":
    main()