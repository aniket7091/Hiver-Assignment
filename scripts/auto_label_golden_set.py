"""
Auto-label the remaining unlabeled examples in the golden set.

Uses an LLM (Gemini or OpenAI) to assign:
  - gold_intent
  - gold_action
  - gold_escalation_reason

Prerequisites:
  pip install google-genai   (for Gemini)
  pip install openai          (for OpenAI)

Usage:
  # Set one of these in .env or environment:
  export GOOGLE_API_KEY="your-key"     # for Gemini (default)
  export OPENAI_API_KEY="your-key"     # for OpenAI

  python scripts/auto_label_golden_set.py
  python scripts/auto_label_golden_set.py --dry-run
  python scripts/auto_label_golden_set.py --provider openai
  python scripts/auto_label_golden_set.py --model gemini-2.5-flash
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_FILE = Path("data/golden/golden_set.jsonl")
OUTPUT_FILE = Path("data/golden/golden_set.jsonl")
GUIDELINES_FILE = Path("data/golden/annotation_guidelines.md")


# ---------------------------------------------------------------------------
# Valid label values (mirrors label_golden_set.py exactly)
# ---------------------------------------------------------------------------

VALID_INTENTS = [
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
]

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

VALID_ACTIONS = ["auto_handle", "escalate"]

VALID_ESCALATION_REASONS = [
    "requires_account_access",
    "requires_order_or_transaction_action",
    "requires_sensitive_information",
    "complex_or_unresolved_issue",
    "customer_requests_human",
    "policy_or_exception_case",
    "insufficient_information",
    "not_suitable_for_automation",
    "none",
]


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def load_records():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Golden set not found: {INPUT_FILE}")

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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_labeled(record):
    return (
        record.get("gold_intent")
        and record.get("gold_action")
        and record.get("gold_escalation_reason")
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_guidelines():
    """Load the annotation guidelines markdown."""
    if GUIDELINES_FILE.exists():
        return GUIDELINES_FILE.read_text(encoding="utf-8")
    return ""


def build_system_prompt(guidelines_text):
    """Build the system prompt with full annotation context."""

    intent_list = "\n".join(
        f"  {i+1:>2}. {intent:<28} - {INTENT_DESCRIPTIONS[intent]}"
        for i, intent in enumerate(VALID_INTENTS)
    )

    action_list = "\n".join(
        f"  - {a}" for a in VALID_ACTIONS
    )

    escalation_list = "\n".join(
        f"  {i+1:>2}. {reason}"
        for i, reason in enumerate(VALID_ESCALATION_REASONS)
    )

    return f"""You are an expert annotator for Amazon customer support intent classification.

Your task: Given a customer message and its thread context from the AmazonHelp Twitter dataset, assign exactly three labels.

## ANNOTATION GUIDELINES

{guidelines_text}

## ALLOWED LABELS

### Intents (pick exactly ONE):
{intent_list}

### Actions (pick exactly ONE):
{action_list}

- auto_handle: The agent can safely handle the request automatically (informational, general question, positive feedback, simple guidance).
- escalate: The request needs a human support specialist (account access, order/transaction actions, sensitive info, complex issues).

### Escalation Reasons (pick exactly ONE):
{escalation_list}

CRITICAL RULE: If action is "auto_handle", escalation_reason MUST be "none".
If action is "escalate", escalation_reason MUST NOT be "none".

## LABELING PRINCIPLES

1. Focus on the CUSTOMER MESSAGE as the primary input. Use thread context for disambiguation only.
2. The heuristic_categories in sampling metadata are ONLY sampling hints. Do NOT blindly use them as labels.
3. If the customer has a clear operational problem (delivery, order, refund, etc.), prefer that specific intent over "support_complaint" or "other".
4. "support_complaint" is ONLY when the PRIMARY issue is dissatisfaction with Amazon's customer support itself.
5. "other" is for messages that truly cannot be classified: greetings, thanks, reactions, insufficient info, unrelated content.
6. For multilingual messages, classify based on the actual content regardless of language.
7. auto_handle is appropriate for: informational questions, positive feedback, general inquiries, content requests, simple guidance.
8. escalate is appropriate when: account access is needed, order/transaction actions are required, sensitive info is involved, issue is complex/unresolved, customer explicitly requests human, policy exceptions needed.

## OUTPUT FORMAT

Respond with ONLY a valid JSON object (no markdown fences, no extra text):
{{"intent": "<intent>", "action": "<action>", "escalation_reason": "<reason>", "reasoning": "<1-2 sentence explanation>"}}
"""


def build_few_shot_examples(labeled_records):
    """Select diverse few-shot examples from hand-labeled data."""

    # Pick examples that cover different intents and both actions
    target_ids = [
        "golden_0001",  # order_issue / escalate / requires_order_or_transaction_action
        "golden_0010",  # shipping_issue / auto_handle / none
        "golden_0014",  # payment_issue / escalate / requires_order_or_transaction_action
        "golden_0015",  # product_or_content_issue / auto_handle / none
        "golden_0025",  # other / auto_handle / none
        "golden_0029",  # delivery_issue / escalate / requires_order_or_transaction_action
        "golden_0030",  # account_issue / escalate / requires_account_access
        "golden_0007",  # support_complaint / escalate / complex_or_unresolved_issue
    ]

    id_to_record = {
        r["example_id"]: r for r in labeled_records
    }

    examples = []
    for tid in target_ids:
        if tid in id_to_record:
            examples.append(id_to_record[tid])

    return examples


def format_example_for_prompt(record):
    """Format a single example as user/assistant pair for few-shot."""

    user_part = format_user_message(record)

    assistant_part = json.dumps({
        "intent": record["gold_intent"],
        "action": record["gold_action"],
        "escalation_reason": record["gold_escalation_reason"],
        "reasoning": record.get("annotator_notes", "") or "Classified based on message content.",
    }, ensure_ascii=False)

    return user_part, assistant_part


def format_user_message(record):
    """Format the user message for a single example."""

    parts = []

    # Customer message
    parts.append(f"CUSTOMER MESSAGE:\n{record.get('customer_message', '')}")

    # Historical response
    hist = record.get("historical_amazonhelp_response", "")
    if hist:
        parts.append(
            f"HISTORICAL AMAZONHELP RESPONSE:\n{hist}"
        )

    # Thread context
    context = record.get("thread_context", [])
    if context:
        ctx_lines = []
        for i, msg in enumerate(context, start=1):
            role = msg.get("role", "unknown").upper()
            text = msg.get("text", "")
            ctx_lines.append(f"[{i}] {role}: {text}")
        parts.append(
            "THREAD CONTEXT:\n" + "\n".join(ctx_lines)
        )

    # Sampling hints
    metadata = record.get("sampling_metadata", {})
    difficulty = metadata.get("difficulty", "unknown")
    heuristic_cats = ", ".join(
        metadata.get("heuristic_categories", [])
    ) or "none"

    parts.append(
        f"SAMPLING HINTS (do NOT blindly use as labels):\n"
        f"Difficulty: {difficulty}\n"
        f"Heuristic categories: {heuristic_cats}"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM clients
# ---------------------------------------------------------------------------

def create_gemini_client(model_name):
    """Create a Gemini client."""
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed.")
        print("Run: pip install google-genai")
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Set it in .env or export GOOGLE_API_KEY='your-key'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    def call_llm(system_prompt, messages):
        """
        messages: list of {"role": "user"|"model", "content": str}
        """
        contents = []
        for msg in messages:
            role = msg["role"]
            # Gemini uses "model" not "assistant"
            if role == "assistant":
                role = "model"
            contents.append(
                genai.types.Content(
                    role=role,
                    parts=[genai.types.Part(text=msg["content"])]
                )
            )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=300,
            ),
        )
        return response.text

    return call_llm


def create_openai_client(model_name):
    """Create an OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed.")
        print("Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Set it in .env or export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    def call_llm(system_prompt, messages):
        """
        messages: list of {"role": "user"|"assistant", "content": str}
        """
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages

        response = client.chat.completions.create(
            model=model_name,
            messages=full_messages,
            temperature=0.1,
            max_tokens=300,
        )
        return response.choices[0].message.content

    return call_llm


# ---------------------------------------------------------------------------
# Response parsing & validation
# ---------------------------------------------------------------------------

def parse_llm_response(raw_text):
    """Parse the LLM response into a dict."""

    text = raw_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try to extract JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def validate_labels(parsed):
    """
    Validate parsed labels.
    Returns (intent, action, escalation_reason, reasoning) or raises ValueError.
    """

    intent = parsed.get("intent", "").strip()
    action = parsed.get("action", "").strip()
    esc_reason = parsed.get("escalation_reason", "").strip()
    reasoning = parsed.get("reasoning", "").strip()

    if intent not in VALID_INTENTS:
        raise ValueError(f"Invalid intent: '{intent}'")

    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: '{action}'")

    if esc_reason not in VALID_ESCALATION_REASONS:
        raise ValueError(
            f"Invalid escalation_reason: '{esc_reason}'"
        )

    # Enforce the auto_handle → none rule
    if action == "auto_handle":
        esc_reason = "none"

    # Enforce escalate → not none rule
    if action == "escalate" and esc_reason == "none":
        raise ValueError(
            "action='escalate' but escalation_reason='none'"
        )

    return intent, action, esc_reason, reasoning


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def label_one_example(call_llm, system_prompt, few_shot_messages,
                      record, max_retries=3):
    """Label a single example with retries."""

    user_msg = format_user_message(record)

    # Build messages: few-shot + current example
    messages = few_shot_messages + [
        {"role": "user", "content": user_msg}
    ]

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(system_prompt, messages)
            parsed = parse_llm_response(raw)
            intent, action, esc_reason, reasoning = validate_labels(
                parsed
            )
            return intent, action, esc_reason, reasoning

        except Exception as e:
            if attempt < max_retries:
                print(
                    f"    Retry {attempt}/{max_retries}: {e}"
                )
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Failed after {max_retries} attempts. "
                    f"Last error: {e}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label golden set using LLM"
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: gemini-2.0-flash / gpt-4o-mini)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts without calling LLM or saving",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0)",
    )

    args = parser.parse_args()

    # Load .env manually (avoid dotenv dependency)
    env_path = Path(".env")
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    os.environ.setdefault(key, value)

    # Resolve model name
    default_models = {
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o-mini",
    }
    model_name = args.model or default_models[args.provider]

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

    records = load_records()
    total = len(records)

    labeled_records = [r for r in records if is_labeled(r)]
    unlabeled_indices = [
        i for i, r in enumerate(records) if not is_labeled(r)
    ]

    print("=" * 70)
    print("AUTO-LABEL GOLDEN SET")
    print("=" * 70)
    print(f"Provider         : {args.provider}")
    print(f"Model            : {model_name}")
    print(f"Total examples   : {total}")
    print(f"Already labeled  : {len(labeled_records)}")
    print(f"To label         : {len(unlabeled_indices)}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Delay            : {args.delay}s")
    print("=" * 70)

    if not unlabeled_indices:
        print("\nAll examples are already labeled. Nothing to do.")
        return

    # -------------------------------------------------------------------
    # Build prompts
    # -------------------------------------------------------------------

    guidelines_text = load_guidelines()
    system_prompt = build_system_prompt(guidelines_text)

    # Build few-shot messages
    few_shot_examples = build_few_shot_examples(labeled_records)
    few_shot_messages = []
    for ex in few_shot_examples:
        user_part, assistant_part = format_example_for_prompt(ex)
        few_shot_messages.append(
            {"role": "user", "content": user_part}
        )
        few_shot_messages.append(
            {"role": "assistant", "content": assistant_part}
        )

    print(f"\nFew-shot examples: {len(few_shot_examples)}")
    for ex in few_shot_examples:
        print(
            f"  {ex['example_id']}: "
            f"{ex['gold_intent']} / {ex['gold_action']}"
        )

    # -------------------------------------------------------------------
    # Dry run
    # -------------------------------------------------------------------

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN — showing first unlabeled example prompt")
        print("=" * 70)

        first_record = records[unlabeled_indices[0]]
        user_msg = format_user_message(first_record)

        print(f"\n--- System prompt ({len(system_prompt)} chars) ---")
        print(system_prompt[:500] + "...")

        print(f"\n--- Few-shot pairs: {len(few_shot_messages)} messages ---")

        print(f"\n--- User message for {first_record['example_id']} ---")
        print(user_msg)

        print("\n✓ Dry run complete. No API calls made, no files changed.")
        return

    # -------------------------------------------------------------------
    # Create LLM client
    # -------------------------------------------------------------------

    if args.provider == "gemini":
        call_llm = create_gemini_client(model_name)
    else:
        call_llm = create_openai_client(model_name)

    # -------------------------------------------------------------------
    # Label loop
    # -------------------------------------------------------------------

    success_count = 0
    error_count = 0
    start_time = time.time()

    from collections import Counter
    intent_counter = Counter()
    action_counter = Counter()
    esc_counter = Counter()

    for progress_idx, record_idx in enumerate(unlabeled_indices):

        record = records[record_idx]
        example_id = record.get("example_id", f"index_{record_idx}")

        print(
            f"\n[{progress_idx + 1}/{len(unlabeled_indices)}] "
            f"{example_id}",
            end=" ... ",
            flush=True,
        )

        try:
            intent, action, esc_reason, reasoning = label_one_example(
                call_llm,
                system_prompt,
                few_shot_messages,
                record,
            )

            # Write labels
            record["gold_intent"] = intent
            record["gold_action"] = action
            record["gold_escalation_reason"] = esc_reason
            record["annotator_notes"] = (
                f"auto-labeled ({model_name}): {reasoning}"
            )

            # Save after every example (crash-safe)
            save_records(records)

            # Track stats
            success_count += 1
            intent_counter[intent] += 1
            action_counter[action] += 1
            esc_counter[esc_reason] += 1

            print(
                f"✓ {intent} / {action} / {esc_reason}"
            )

        except Exception as e:
            error_count += 1
            print(f"✗ ERROR: {e}")

            # Still save to preserve any previous progress
            save_records(records)

        # Rate limit
        if progress_idx < len(unlabeled_indices) - 1:
            time.sleep(args.delay)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n\n" + "=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)

    print(f"Successfully labeled : {success_count}")
    print(f"Errors               : {error_count}")
    print(f"Time                 : {elapsed:.1f}s")
    print(f"Saved to             : {OUTPUT_FILE}")

    print("\n--- Intent distribution (auto-labeled) ---")
    for intent, count in intent_counter.most_common():
        print(f"  {intent:<28} : {count}")

    print("\n--- Action distribution (auto-labeled) ---")
    for action, count in action_counter.most_common():
        print(f"  {action:<28} : {count}")

    print("\n--- Escalation reason distribution (auto-labeled) ---")
    for reason, count in esc_counter.most_common():
        print(f"  {reason:<28} : {count}")

    # Final verification
    final_records = load_records()
    final_labeled = sum(1 for r in final_records if is_labeled(r))

    print(f"\n--- Final status ---")
    print(f"  Total    : {len(final_records)}")
    print(f"  Labeled  : {final_labeled}")
    print(f"  Remaining: {len(final_records) - final_labeled}")

    if final_labeled == len(final_records):
        print("\n🎉 All examples are now labeled!")
    else:
        print(
            f"\n⚠ {len(final_records) - final_labeled} examples "
            f"still need labels. Re-run to retry."
        )


if __name__ == "__main__":
    main()
