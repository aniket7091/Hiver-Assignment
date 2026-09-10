"""Categorize agent prediction failures against gold labels."""

from typing import Any, Dict, List


def identify_failures(
    agent_result: Dict[str, Any],
) -> List[Dict[str, Any]]:

    failures = []

    for example in agent_result["predictions"]:

        # Intent error

        if not example["intent_correct"]:
            failures.append({
                "example_id": example["example_id"],
                "failure_type": "intent_misclassification",
                "customer_message": example["customer_message"],
                "gold_intent": example["gold_intent"],
                "predicted_intent": example["predicted_intent"],
                "intent_confidence": example["intent_confidence"],
            })

        # Action error

        if not example["action_correct"]:
            failures.append({
                "example_id": example["example_id"],
                "failure_type": "action_misclassification",
                "customer_message": example["customer_message"],
                "gold_action": example["gold_action"],
                "predicted_action": example["predicted_action"],
                "gold_escalation_reason": example["gold_escalation_reason"],
                "predicted_escalation_reason": example["predicted_escalation_reason"],
            })

        # Ungrounded auto-handle

        if (
            example["gold_action"] == "auto_handle"
            and not example["grounded"]
        ):
            failures.append({
                "example_id": example["example_id"],
                "failure_type": "ungrounded_auto_handle",
                "customer_message": example["customer_message"],
                "gold_intent": example["gold_intent"],
                "predicted_intent": example["predicted_intent"],
                "predicted_action": example["predicted_action"],
                "top_retrieval_score": example["top_retrieval_score"],
                "reply": example["reply"],
            })

        # Retrieval ambiguity

        if (
            example["top_retrieval_score"] > 0
            and example["retrieval_score_gap"] < 0.05
        ):
            failures.append({
                "example_id": example["example_id"],
                "failure_type": "ambiguous_retrieval",
                "customer_message": example["customer_message"],
                "top_retrieval_score": example["top_retrieval_score"],
                "second_retrieval_score": example["second_retrieval_score"],
                "retrieval_score_gap": example["retrieval_score_gap"],
                "reply": example["reply"],
            })

    return failures
