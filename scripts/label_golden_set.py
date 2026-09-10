import json
from pathlib import Path



# Configuration


INPUT_FILE = Path("data/golden/golden_set.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.jsonl")



# Allowed intents


INTENTS = {
    "1": "delivery_issue",
    "2": "order_issue",
    "3": "shipping_issue",
    "4": "payment_issue",
    "5": "refund_issue",
    "6": "account_issue",
    "7": "prime_video_issue",
    "8": "device_issue",
    "9": "product_or_content_issue",
    "10": "support_complaint",
    "11": "other",
}


INTENT_DESCRIPTIONS = {
    "delivery_issue":
        "Package missing, late, delivered-but-not-received, delivery problem",

    "order_issue":
        "Order status/details/cancellation/order-specific problem",

    "shipping_issue":
        "Shipping method, shipping speed, Prime shipping, promised shipping",

    "payment_issue":
        "Payment, charge, billing, card, transaction problem",

    "refund_issue":
        "Refund requested, missing refund, refund status, money back",

    "account_issue":
        "Login, account access, account settings, account closure",

    "prime_video_issue":
        "Prime Video playback, streaming, buffering, video errors, video quality",

    "device_issue":
        "Fire TV, Kindle, Echo, Alexa, remote, or other Amazon device",

    "product_or_content_issue":
        "Product/item/listing/seller/content/movie/show problem or feedback",

    "support_complaint":
        "Complaint about Amazon customer support/service",

    "other":
        "Greeting, thanks, reaction, unrelated, unclear, or insufficient information",
}



# Allowed actions


ACTIONS = {
    "1": "auto_handle",
    "2": "escalate",
}



# Escalation reasons


ESCALATION_REASONS = {
    "1": "requires_account_access",
    "2": "requires_order_or_transaction_action",
    "3": "requires_sensitive_information",
    "4": "complex_or_unresolved_issue",
    "5": "customer_requests_human",
    "6": "policy_or_exception_case",
    "7": "insufficient_information",
    "8": "not_suitable_for_automation",
    "9": "none",
}



# Helpers


def load_records():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Golden set not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def save_records(records):
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def show_intents():
    print("\n" + "-" * 70)
    print("INTENT")
    print("-" * 70)

    for number, intent in INTENTS.items():
        print(f"{number:>2}. {intent:<28} - {INTENT_DESCRIPTIONS[intent]}")


def show_actions():
    print("\n" + "-" * 70)
    print("ACTION")
    print("-" * 70)

    print("1. auto_handle")
    print("   Agent can safely handle the request automatically.")

    print("2. escalate")
    print("   Request should be sent to a human/support specialist.")


def show_escalation_reasons():
    print("\n" + "-" * 70)
    print("ESCALATION REASON")
    print("-" * 70)

    for number, reason in ESCALATION_REASONS.items():
        print(f"{number:>2}. {reason}")


def get_intent():
    while True:
        show_intents()

        choice = input("\nSelect intent number: ").strip()

        if choice in INTENTS:
            return INTENTS[choice]

        print("\nInvalid choice. Please enter a number from 1-11.")


def get_action():
    while True:
        show_actions()

        choice = input("\nSelect action number: ").strip()

        if choice in ACTIONS:
            return ACTIONS[choice]

        print("\nInvalid choice. Please enter 1 or 2.")


def get_escalation_reason(action):
    # If automatically handled, escalation reason is none.
    if action == "auto_handle":
        return "none"

    while True:
        show_escalation_reasons()

        choice = input(
            "\nSelect escalation reason number: "
        ).strip()

        if choice in ESCALATION_REASONS:
            return ESCALATION_REASONS[choice]

        print("\nInvalid choice. Please enter a number from 1-9.")


def get_notes():
    print("\nOptional annotator note.")
    print("Press Enter without typing anything to skip.")

    return input("Note: ").strip()


def print_context(context):
    if not context:
        print("No thread context available.")
        return

    print("\n" + "-" * 70)
    print("THREAD CONTEXT")
    print("-" * 70)

    for i, message in enumerate(context, start=1):
        role = message.get("role", "unknown")
        text = message.get("text", "")

        print(f"\n[{i}] {role.upper()}:")
        print(text)


def print_previous_progress(records):
    total = len(records)

    labelled = 0

    for record in records:
        if (
            record.get("gold_intent")
            and record.get("gold_action")
            and record.get("gold_escalation_reason")
        ):
            labelled += 1

    print("\n" + "=" * 70)
    print("GOLDEN SET LABELING")
    print("=" * 70)

    print(f"Total examples : {total}")
    print(f"Already labeled: {labelled}")
    print(f"Remaining      : {total - labelled}")



# main Labelling loop

def main():

    records = load_records()

    if not records:
        print("Golden set is empty.")
        return

    print_previous_progress(records)

    print(
        "\nInstructions:"
        "\n- Read the customer message."
        "\n- Use thread context when necessary."
        "\n- Assign ONE intent."
        "\n- Decide auto_handle vs escalate."
        "\n- If auto_handle, escalation reason is automatically 'none'."
        "\n- Type 'q' at the example prompt to save and quit."
    )

    for index, record in enumerate(records):


        # Skip already labeled examples


        if (
            record.get("gold_intent")
            and record.get("gold_action")
            and record.get("gold_escalation_reason")
        ):
            continue

        print("\n\n")
        print("=" * 70)
        print(
            f"Example {index + 1}/{len(records)}"
        )
        print("=" * 70)

        print(f"\nExample ID: {record.get('example_id')}")


        # Customer message


        print("\n" + "-" * 70)
        print("CUSTOMER MESSAGE")
        print("-" * 70)

        print(record.get("customer_message", ""))


        # Historical AmazonHelp response


        historical_response = record.get(
            "historical_amazonhelp_response",
            ""
        )

        if historical_response:

            print("\n" + "-" * 70)
            print("HISTORICAL AMAZONHELP RESPONSE")
            print("-" * 70)

            print(historical_response)


        # Thread context


        print_context(
            record.get("thread_context", [])
        )


        # Sampling hints


        metadata = record.get(
            "sampling_metadata",
            {}
        )

        print("\n" + "-" * 70)
        print("SAMPLING INFORMATION")
        print("-" * 70)

        print(
            f"Difficulty: "
            f"{metadata.get('difficulty', 'unknown')}"
        )

        print(
            f"Heuristic categories: "
            f"{', '.join(metadata.get('heuristic_categories', [])) or 'none'}"
        )

        print(
            "\nIMPORTANT: heuristic categories are only "
            "sampling hints. Do NOT blindly use them as the label."
        )


        # Quit option


        print("\n" + "-" * 70)

        command = input(
            "Press Enter to label this example, "
            "or type 'q' to save and quit: "
        ).strip().lower()

        if command == "q":
            save_records(records)

            print("\nProgress saved.")
            print(f"File: {OUTPUT_FILE}")

            return


        # Intent


        intent = get_intent()


        # Action


        action = get_action()


        # Escalation reason


        escalation_reason = get_escalation_reason(
            action
        )


        # Notes


        notes = get_notes()


        # Save labels


        record["gold_intent"] = intent
        record["gold_action"] = action
        record["gold_escalation_reason"] = escalation_reason
        record["annotator_notes"] = notes

        # Save after EVERY example.
        # This protects against losing progress.
        save_records(records)

        print("\n✓ Example labeled successfully.")

        print(f"Intent           : {intent}")
        print(f"Action           : {action}")
        print(
            f"Escalation reason: "
            f"{escalation_reason}"
        )


    # Finished


    save_records(records)

    print("\n")
    print("=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)

    print(f"Total examples: {len(records)}")
    print(f"Saved to       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()