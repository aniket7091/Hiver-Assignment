"""End-to-end agent evaluation loop against the golden set."""

from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.agent.pipeline import AmazonHelpAgent
from src.evaluation.golden_set import VALID_ACTIONS, VALID_INTENTS


def run_agent_evaluation(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:

    print("\nLoading production agent...")

    agent = AmazonHelpAgent(
        top_k=5,
        min_evidence_score=0.35,
    )

    print("Production agent loaded.")

    predictions = []

    intent_predictions = []
    intent_gold = []

    action_predictions = []
    action_gold = []

    grounded_count = 0
    retrieval_available_count = 0

    for index, record in enumerate(records, start=1):

        customer_message = str(record["customer_message"]).strip()
        gold_intent = str(record["gold_intent"]).strip()
        gold_action = str(record["gold_action"]).strip()

        # Run agent

        try:
            result = agent.process(customer_message)
        except Exception as exc:
            print(f"\nERROR on {record['example_id']}: {exc}")
            result = {
                "intent": "other",
                "intent_confidence": 0.0,
                "reply": "",
                "grounded": False,
                "action": "escalate",
                "escalation_reason": "Agent execution failed.",
                "reason_code": "agent_error",
                "top_retrieval_score": 0.0,
                "second_retrieval_score": 0.0,
                "retrieval_score_gap": 0.0,
                "retrieval": [],
                "evidence": [],
            }

        predicted_intent = str(
            result.get("intent", "other")
        ).strip().lower()

        predicted_action = str(
            result.get("action", "escalate")
        ).strip().lower()

        intent_confidence = result.get("intent_confidence")

        # Accumulate metrics

        intent_gold.append(gold_intent)
        intent_predictions.append(predicted_intent)
        action_gold.append(gold_action)
        action_predictions.append(predicted_action)

        grounded = bool(result.get("grounded", False))
        if grounded:
            grounded_count += 1

        retrieval = result.get("retrieval", [])
        if retrieval:
            retrieval_available_count += 1

        evidence = result.get("evidence", [])
        top_evidence = evidence[0] if evidence else None

        # Store per-example result

        example_result = {
            "example_id": record["example_id"],
            "customer_message": customer_message,

            # Gold labels
            "gold_intent": gold_intent,
            "gold_action": gold_action,
            "gold_escalation_reason": record["gold_escalation_reason"],

            # Intent prediction
            "predicted_intent": predicted_intent,
            "intent_correct": predicted_intent == gold_intent,
            "intent_confidence": intent_confidence,

            # Action prediction
            "predicted_action": predicted_action,
            "action_correct": predicted_action == gold_action,

            # Reply
            "reply": result.get("reply", ""),
            "grounded": grounded,
            "reply_reason": result.get("reply_reason"),

            # Retrieval
            "top_retrieval_score": result.get("top_retrieval_score", 0.0),
            "second_retrieval_score": result.get("second_retrieval_score", 0.0),
            "retrieval_score_gap": result.get("retrieval_score_gap", 0.0),
            "retrieval_count": len(retrieval),

            # Escalation
            "predicted_escalation_reason": result.get("escalation_reason"),
            "reason_code": result.get("reason_code"),

            # Evidence
            "top_evidence": top_evidence,

            # Full retrieval snapshot. This is deliberately retained so
            # LLM-judge caching and later audits can distinguish a change in
            # retrieval from a coincidentally identical generated reply.
            "retrieved_evidence": retrieval,
        }

        predictions.append(example_result)

        if index % 25 == 0:
            print(f"Evaluated {index}/{len(records)}")

    # -----------------------------------------------------------------------
    # Intent metrics
    # -----------------------------------------------------------------------

    intent_accuracy = accuracy_score(intent_gold, intent_predictions)

    intent_macro_f1 = f1_score(
        intent_gold, intent_predictions,
        labels=VALID_INTENTS, average="macro", zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        intent_gold, intent_predictions,
        labels=VALID_INTENTS, average="weighted", zero_division=0,
    )

    intent_report = classification_report(
        intent_gold, intent_predictions,
        labels=VALID_INTENTS, output_dict=True, zero_division=0,
    )

    # -----------------------------------------------------------------------
    # Action metrics
    # -----------------------------------------------------------------------

    action_accuracy = accuracy_score(action_gold, action_predictions)

    action_report = classification_report(
        action_gold, action_predictions,
        labels=VALID_ACTIONS, output_dict=True, zero_division=0,
    )

    # -----------------------------------------------------------------------
    # Auto-handle precision
    # -----------------------------------------------------------------------

    predicted_auto = [p == "auto_handle" for p in action_predictions]
    true_auto = [g == "auto_handle" for g in action_gold]

    auto_handle_tp = sum(
        predicted and actual
        for predicted, actual in zip(predicted_auto, true_auto)
    )
    auto_handle_fp = sum(
        predicted and not actual
        for predicted, actual in zip(predicted_auto, true_auto)
    )

    if (auto_handle_tp + auto_handle_fp) > 0:
        auto_handle_precision = auto_handle_tp / (auto_handle_tp + auto_handle_fp)
    else:
        auto_handle_precision = 0.0

    # -----------------------------------------------------------------------
    # Escalation recall
    # -----------------------------------------------------------------------

    predicted_escalate = [p == "escalate" for p in action_predictions]
    true_escalate = [g == "escalate" for g in action_gold]

    escalation_tp = sum(
        predicted and actual
        for predicted, actual in zip(predicted_escalate, true_escalate)
    )
    escalation_fn = sum(
        (not predicted) and actual
        for predicted, actual in zip(predicted_escalate, true_escalate)
    )

    if (escalation_tp + escalation_fn) > 0:
        escalation_recall = escalation_tp / (escalation_tp + escalation_fn)
    else:
        escalation_recall = 0.0

    # -----------------------------------------------------------------------
    # Grounding
    # -----------------------------------------------------------------------

    grounding_rate = grounded_count / len(records)
    retrieval_rate = retrieval_available_count / len(records)

    return {
        "name": "end_to_end_agent",
        "num_examples": len(records),
        "intent": {
            "accuracy": round(float(intent_accuracy), 4),
            "macro_f1": round(float(intent_macro_f1), 4),
            "weighted_f1": round(float(intent_weighted_f1), 4),
            "classification_report": intent_report,
        },
        "action": {
            "accuracy": round(float(action_accuracy), 4),
            "classification_report": action_report,
            "auto_handle_precision": round(float(auto_handle_precision), 4),
            "escalation_recall": round(float(escalation_recall), 4),
        },
        "grounding": {
            "grounded_reply_rate": round(float(grounding_rate), 4),
            "retrieval_available_rate": round(float(retrieval_rate), 4),
        },
        "predictions": predictions,
    }
