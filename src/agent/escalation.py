import re
from typing import Any, Dict, List, Optional

from src.intent.schemas import has_support_complaint_signal



# CONFIG


# Minimum retrieval similarity required for confident handling.
DEFAULT_MIN_RETRIEVAL_SCORE = 0.50

# Minimum score gap between top-1 and top-2 results.
# A small gap means retrieval is ambiguous.
DEFAULT_MIN_SCORE_GAP = 0.05



# ESCALATION REASONS


ESCALATION_REASONS = {
    "low_retrieval_confidence": (
        "Historical examples are not similar enough "
        "to confidently provide a grounded resolution."
    ),

    "ambiguous_retrieval": (
        "Multiple historical cases have similar scores, "
        "so the correct resolution is ambiguous."
    ),

    "unknown_intent": (
        "The message does not map confidently to a supported "
        "business intent."
    ),

    "sensitive_account_action": (
        "The issue involves account or payment information "
        "that should be handled securely."
    ),

    "refund_or_financial_action": (
        "The request may require account-specific or financial "
        "action that cannot be safely completed from Twitter."
    ),

    "support_complaint": (
        "The customer is expressing dissatisfaction or requesting "
        "escalated support."
    ),

    "insufficient_information": (
        "The customer message does not contain enough information "
        "to determine a reliable resolution."
    ),

    "complex_issue": (
        "The issue requires investigation or account-specific "
        "support beyond the available historical evidence."
    ),

    "safe_to_auto_handle": (
        "The intent is supported and sufficiently similar "
        "historical guidance is available."
    ),

    "low_intent_confidence": (
        "The intent classifier confidence is too low to "
        "safely auto-handle."
    ),

    "conflicting_candidate_intents": (
        "The top retrieved historical cases have conflicting "
        "business intents, indicating ambiguity."
    ),

    "non_english_or_special_chars": (
        "The message contains non-ASCII characters or appears "
        "to be non-English."
    ),

    "complaint_or_unresolved_issue": (
        "The message indicates explicit dissatisfaction or an unresolved "
        "support history that requires investigation."
    ),

    "incompatible_retrieval_intent": (
        "The highest lexical retrieval result does not match the predicted "
        "intent, so it cannot be trusted as supporting evidence."
    ),

    "ungrounded_retrieval_evidence": (
        "No safe, grounded response could be derived from the retrieved "
        "historical evidence."
    ),
}



# UTILITY FUNCTIONS


def normalize_intent(intent: Optional[str]) -> str:
    """
    Normalize an intent label.
    """

    if not intent:
        return ""

    return str(intent).strip().lower()


def get_top_scores(
    retrieved_results: Optional[List[Dict[str, Any]]]
):
    """
    Return top-1 and top-2 retrieval similarity scores.
    """

    if not retrieved_results:
        return 0.0, 0.0

    scores = []

    for result in retrieved_results:

        try:
            score = float(
                result.get("score", 0.0)
            )
        except (TypeError, ValueError):
            score = 0.0

        scores.append(score)

    scores.sort(reverse=True)

    top_1 = scores[0] if scores else 0.0
    top_2 = scores[1] if len(scores) > 1 else 0.0

    return top_1, top_2


def message_is_insufficient(
    customer_message: str
) -> bool:
    """
    Detect very short, empty, or non-English messages.
    """
    if not customer_message:
        return True

    text = str(customer_message).strip()
    if not text:
        return True

    words = text.split()
    if len(words) < 3:
        return True

    return False

def contains_non_ascii(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r'[^\x00-\x7F]+', text))


UNRESOLVED_ISSUE_PATTERNS = (
    r"\b(?:still\s+)?(?:not|hasn't|haven't|isn't)\s+"
    r"(?:been\s+)?resolved\b",
    r"\b(?:still\s+)?no\s+(?:help|response|resolution)\b",
    r"\b(?:multiple|several)\s+(?:times|attempts|contacts)\b",
    r"\b(?:already\s+)?(?:contacted|called|emailed|messaged)\s+"
    r"(?:support|customer\s+service)\b",
)


def has_unresolved_issue_signal(customer_message: str) -> bool:
    """Detect investigation-heavy wording that should not be auto-handled."""

    if not customer_message:
        return False

    text = re.sub(r"\s+", " ", str(customer_message).lower()).strip()

    return any(
        re.search(pattern, text) is not None
        for pattern in UNRESOLVED_ISSUE_PATTERNS
    )


def top_retrieval_matches_intent(
    retrieved_results: List[Dict[str, Any]],
    intent: str,
    candidate_intents: Optional[List[str]],
) -> bool:
    """Require a verified compatible intent for the highest-ranked evidence."""

    if not retrieved_results:
        return False

    top_result = retrieved_results[0]
    candidate_intent = normalize_intent(
        top_result.get("candidate_intent")
    )

    if candidate_intent:
        return candidate_intent == intent

    if top_result.get("intent_compatible") is not None:
        return bool(top_result["intent_compatible"])

    if candidate_intents:
        return normalize_intent(candidate_intents[0]) == intent

    # Absence of an intent check is not evidence compatibility.
    return False




# ESCALATION DECISION


def decide_escalation(
    customer_message: str,
    intent: Optional[str],
    retrieved_results: Optional[List[Dict[str, Any]]] = None,
    min_retrieval_score: float = DEFAULT_MIN_RETRIEVAL_SCORE,
    min_score_gap: float = DEFAULT_MIN_SCORE_GAP,
    intent_confidence: Optional[float] = None,
    candidate_intents: Optional[List[str]] = None,
    has_grounded_reply: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Decide whether a customer message should be
    auto-handled or escalated.

    Parameters
    ----------
    customer_message:
        Incoming customer message.

    intent:
        Predicted intent.

    retrieved_results:
        Historical retrieval results.

    min_retrieval_score:
        Minimum top retrieval score required for
        confident handling.

    min_score_gap:
        Minimum difference between top-1 and top-2
        results when checking ambiguity.

    Returns
    -------
    dict
        Action, reason and supporting signals.
    """

    intent = normalize_intent(intent)

    retrieved_results = retrieved_results or []

    top_1, top_2 = get_top_scores(
        retrieved_results
    )

    score_gap = top_1 - top_2

    # 1. Customer complaint
    #
    # Specific high-risk/business intents take precedence over a generic
    # classifier-confidence diagnostic. They all still escalate.
    if intent == "support_complaint":
        return {
            "action": "escalate",
            "reason_code": "support_complaint",
            "reason": ESCALATION_REASONS[
                "support_complaint"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 2. Sensitive account/payment cases
    if intent in {
        "account_issue",
        "payment_issue",
    }:
        return {
            "action": "escalate",
            "reason_code": "sensitive_account_action",
            "reason": ESCALATION_REASONS[
                "sensitive_account_action"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 3. Refund / financial actions
    if intent == "refund_issue":
        return {
            "action": "escalate",
            "reason_code": "refund_or_financial_action",
            "reason": ESCALATION_REASONS[
                "refund_or_financial_action"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # Explicit complaint/unresolved wording is a safety signal even if a
    # custom or older classifier assigned an operational intent.
    if (
        has_support_complaint_signal(customer_message)
        or has_unresolved_issue_signal(customer_message)
    ):
        return {
            "action": "escalate",
            "reason_code": "complaint_or_unresolved_issue",
            "reason": ESCALATION_REASONS[
                "complaint_or_unresolved_issue"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 4. Non-English / special-character messages
    if contains_non_ascii(customer_message):
        return {
            "action": "escalate",
            "reason_code": "non_english_or_special_chars",
            "reason": ESCALATION_REASONS["non_english_or_special_chars"],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 5. Empty / insufficient message
    if message_is_insufficient(customer_message):
        return {
            "action": "escalate",
            "reason_code": "insufficient_information",
            "reason": ESCALATION_REASONS[
                "insufficient_information"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 6. Unknown / unsupported intent
    if not intent or intent == "other":
        return {
            "action": "escalate",
            "reason_code": "unknown_intent",
            "reason": ESCALATION_REASONS[
                "unknown_intent"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # 7. Generic low-confidence classification
    if intent_confidence is not None and intent_confidence < 0.60:
        return {
            "action": "escalate",
            "reason_code": "low_intent_confidence",
            "reason": ESCALATION_REASONS["low_intent_confidence"],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }


    # 6. No historical evidence


    if not retrieved_results:
        return {
            "action": "escalate",
            "reason_code": "low_retrieval_confidence",
            "reason": ESCALATION_REASONS[
                "low_retrieval_confidence"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # A conservative fallback reply means no safe historical resolution was
    # available, even if lexical retrieval returned a high score.
    if has_grounded_reply is False:
        return {
            "action": "escalate",
            "reason_code": "ungrounded_retrieval_evidence",
            "reason": ESCALATION_REASONS[
                "ungrounded_retrieval_evidence"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }


    # 7. Retrieval confidence too low


    if top_1 < min_retrieval_score:
        return {
            "action": "escalate",
            "reason_code": "low_retrieval_confidence",
            "reason": ESCALATION_REASONS[
                "low_retrieval_confidence"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }

    # A lexical match must be confirmed by the candidate's independently
    # predicted intent. High lexical similarity alone is not semantic proof.
    if not top_retrieval_matches_intent(
        retrieved_results,
        intent,
        candidate_intents,
    ):
        return {
            "action": "escalate",
            "reason_code": "incompatible_retrieval_intent",
            "reason": ESCALATION_REASONS[
                "incompatible_retrieval_intent"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }


    # 8. Retrieval is ambiguous


    if (
        len(retrieved_results) >= 2
        and score_gap < min_score_gap
    ):
        return {
            "action": "escalate",
            "reason_code": "ambiguous_retrieval",
            "reason": ESCALATION_REASONS[
                "ambiguous_retrieval"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }


    # 8.5 Conflicting Intents

    if candidate_intents and len(candidate_intents) >= 2:
        if candidate_intents[0] != candidate_intents[1]:
            return {
                "action": "escalate",
                "reason_code": "conflicting_candidate_intents",
                "reason": ESCALATION_REASONS["conflicting_candidate_intents"],
                "top_retrieval_score": top_1,
                "second_retrieval_score": top_2,
                "retrieval_score_gap": score_gap,
            }

    # 9. Complex issue categories


    if intent in {
        "order_issue",
        "shipping_issue",
        "product_or_content_issue",
    }:
        return {
            "action": "escalate",
            "reason_code": "complex_issue",
            "reason": ESCALATION_REASONS[
                "complex_issue"
            ],
            "top_retrieval_score": top_1,
            "second_retrieval_score": top_2,
            "retrieval_score_gap": score_gap,
        }


    # 10. Auto-handle


    return {
        "action": "auto_handle",
        "reason_code": "safe_to_auto_handle",
        "reason": ESCALATION_REASONS[
            "safe_to_auto_handle"
        ],
        "top_retrieval_score": top_1,
        "second_retrieval_score": top_2,
        "retrieval_score_gap": score_gap,
    }



# SIMPLE API


def should_escalate(
    customer_message: str,
    intent: Optional[str],
    retrieved_results: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Simple boolean API.

    Returns True when the case should be escalated.
    """

    decision = decide_escalation(
        customer_message=customer_message,
        intent=intent,
        retrieved_results=retrieved_results,
    )

    return decision["action"] == "escalate"



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP ESCALATION POLICY TEST")
    print("=" * 70)

    # Fake retrieval results for testing policy logic.
    # In the real pipeline these will come from retrieve.py.

    strong_retrieval = [
        {
            "score": 0.72,
            "conversation_id": "amazonhelp_10123",
        },
        {
            "score": 0.60,
            "conversation_id": "amazonhelp_63787",
        },
    ]

    weak_retrieval = [
        {
            "score": 0.21,
            "conversation_id": "amazonhelp_123",
        },
    ]

    ambiguous_retrieval = [
        {
            "score": 0.70,
            "conversation_id": "amazonhelp_111",
        },
        {
            "score": 0.69,
            "conversation_id": "amazonhelp_222",
        },
    ]

    test_cases = [
        {
            "message": (
                "My package says delivered "
                "but I haven't received it"
            ),
            "intent": "delivery_issue",
            "results": strong_retrieval,
        },
        {
            "message": (
                "Why is Prime Video not working?"
            ),
            "intent": "prime_video_issue",
            "results": strong_retrieval,
        },
        {
            "message": (
                "I want a refund for my order"
            ),
            "intent": "refund_issue",
            "results": strong_retrieval,
        },
        {
            "message": (
                "My account is locked"
            ),
            "intent": "account_issue",
            "results": strong_retrieval,
        },
        {
            "message": (
                "This is terrible support!"
            ),
            "intent": "support_complaint",
            "results": strong_retrieval,
        },
        {
            "message": (
                "Something is wrong"
            ),
            "intent": "other",
            "results": weak_retrieval,
        },
        {
            "message": (
                "My package is delayed"
            ),
            "intent": "delivery_issue",
            "results": weak_retrieval,
        },
        {
            "message": (
                "My package is delivered"
            ),
            "intent": "delivery_issue",
            "results": ambiguous_retrieval,
        },
    ]


    # Run tests


    for i, case in enumerate(
        test_cases,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(f"TEST CASE {i}")

        print(
            f"Customer : {case['message']}"
        )

        print(
            f"Intent   : {case['intent']}"
        )

        decision = decide_escalation(
            customer_message=case["message"],
            intent=case["intent"],
            retrieved_results=case["results"],
        )

        print(
            f"Action   : {decision['action']}"
        )

        print(
            f"Reason   : {decision['reason']}"
        )

        print(
            f"Top score: "
            f"{decision['top_retrieval_score']:.4f}"
        )

        print(
            f"Score gap: "
            f"{decision['retrieval_score_gap']:.4f}"
        )

    print("\n" + "=" * 70)
    print("ESCALATION POLICY TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
