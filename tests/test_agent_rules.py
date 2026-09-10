import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.pipeline import AmazonHelpAgent
from src.agent.escalation import decide_escalation
from src.intent.schemas import has_support_complaint_signal

@pytest.fixture(scope="module")
def agent():
    return AmazonHelpAgent()

def test_delivery_issue_auto_handles_when_safe(agent):
    result = agent.process("My package hasn't arrived. Tracking says delivered.")
    assert result["intent"] == "delivery_issue"

def test_refund_issue_escalates(agent):
    result = agent.process("I want my money back for this order.")
    assert result["intent"] == "refund_issue"
    assert result["action"] == "escalate"
    assert result["reason_code"] == "refund_or_financial_action"

def test_support_complaint_escalates(agent):
    result = agent.process("This is terrible support, nobody helps me.")
    assert result["intent"] == "support_complaint"
    assert result["action"] == "escalate"
    assert result["reason_code"] == "support_complaint"

def test_short_message_escalates(agent):
    result = agent.process("hi")
    assert result["action"] == "escalate"
    assert result["reason_code"] == "insufficient_information"

def test_payment_issue_escalates(agent):
    result = agent.process("My credit card was charged twice.")
    assert result["intent"] == "payment_issue"
    assert result["action"] == "escalate"
    assert result["reason_code"] == "sensitive_account_action"


@pytest.mark.parametrize(
    ("intent", "expected_reason_code"),
    [
        ("support_complaint", "support_complaint"),
        ("refund_issue", "refund_or_financial_action"),
        ("payment_issue", "sensitive_account_action"),
        ("account_issue", "sensitive_account_action"),
    ],
)
def test_specific_high_risk_intents_override_low_confidence(
    intent,
    expected_reason_code,
):
    result = decide_escalation(
        customer_message="This needs secure customer-support follow-up.",
        intent=intent,
        intent_confidence=0.01,
    )

    assert result["action"] == "escalate"
    assert result["reason_code"] == expected_reason_code


def test_generic_low_confidence_still_escalates():
    result = decide_escalation(
        customer_message="My parcel seems delayed and I need help.",
        intent="delivery_issue",
        intent_confidence=0.01,
    )

    assert result["action"] == "escalate"
    assert result["reason_code"] == "low_intent_confidence"


def test_complaint_with_operational_details_escalates(agent):
    result = agent.process(
        "Very poor customer service. I ordered an item and paid upfront."
    )

    assert result["intent"] == "support_complaint"
    assert result["action"] == "escalate"
    assert result["reason_code"] == "support_complaint"


def test_bare_operational_negative_word_is_not_a_support_complaint():
    assert not has_support_complaint_signal(
        "The delivery timing was poor."
    )


def test_ambiguous_retrieval_escalates():
    tied_candidates = [
        {
            "score": 0.91,
            "candidate_intent": "delivery_issue",
            "intent_compatible": True,
        },
        {
            "score": 0.91,
            "candidate_intent": "delivery_issue",
            "intent_compatible": True,
        },
    ]

    result = decide_escalation(
        customer_message="Please help with delivery",
        intent="delivery_issue",
        intent_confidence=0.95,
        retrieved_results=tied_candidates,
        candidate_intents=["delivery_issue", "delivery_issue"],
        has_grounded_reply=True,
    )

    assert result["action"] == "escalate"
    assert result["reason_code"] == "ambiguous_retrieval"


def test_high_lexical_similarity_with_incompatible_intent_escalates():
    incompatible_candidates = [
        {
            "score": 0.99,
            "lexical_score": 1.0,
            "candidate_intent": "delivery_issue",
            "intent_compatible": False,
        },
        {
            "score": 0.20,
            "lexical_score": 0.20,
            "candidate_intent": "shipping_issue",
            "intent_compatible": True,
        },
    ]

    result = decide_escalation(
        customer_message="Can you ship my order internationally?",
        intent="shipping_issue",
        intent_confidence=0.95,
        retrieved_results=incompatible_candidates,
        candidate_intents=["delivery_issue", "shipping_issue"],
        has_grounded_reply=True,
    )

    assert result["action"] == "escalate"
    assert result["reason_code"] == "incompatible_retrieval_intent"
