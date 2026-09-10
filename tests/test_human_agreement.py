import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.human_agreement import (
    AnnotationValidationError,
    calculate_agreement,
    validate_second_annotations,
)


def golden_record(example_id, intent="delivery_issue", action="auto_handle"):
    return {
        "example_id": example_id,
        "gold_intent": intent,
        "gold_action": action,
        "gold_escalation_reason": "none" if action == "auto_handle" else "review",
    }


def second_record(example_id, intent="delivery_issue", action="auto_handle"):
    return {
        "example_id": example_id,
        "second_intent": intent,
        "second_action": action,
        "second_escalation_reason": "none" if action == "auto_handle" else "review",
    }


def test_perfect_agreement_returns_kappa_one():
    golden = [
        golden_record("one", "delivery_issue", "auto_handle"),
        golden_record("two", "refund_issue", "escalate"),
    ]
    second = [
        second_record("one", "delivery_issue", "auto_handle"),
        second_record("two", "refund_issue", "escalate"),
    ]

    result = calculate_agreement(golden, second)

    assert result["intent"]["raw_agreement"] == 1.0
    assert result["intent"]["cohen_kappa"] == 1.0
    assert result["action"]["raw_agreement"] == 1.0
    assert result["action"]["cohen_kappa"] == 1.0


def test_known_disagreement_has_kappa_below_one():
    golden = [
        golden_record("one", "delivery_issue", "auto_handle"),
        golden_record("two", "refund_issue", "escalate"),
        golden_record("three", "delivery_issue", "auto_handle"),
        golden_record("four", "refund_issue", "escalate"),
    ]
    second = [
        second_record("one", "delivery_issue", "auto_handle"),
        second_record("two", "refund_issue", "escalate"),
        second_record("three", "refund_issue", "escalate"),
        second_record("four", "delivery_issue", "auto_handle"),
    ]

    result = calculate_agreement(golden, second)

    assert result["intent"]["cohen_kappa"] < 1.0
    assert result["action"]["cohen_kappa"] < 1.0


def test_duplicate_ids_are_rejected():
    golden = [golden_record("one"), golden_record("two")]
    second = [second_record("one"), second_record("one")]

    with pytest.raises(AnnotationValidationError, match="Duplicate"):
        validate_second_annotations(golden, second, expected_count=2)


def test_invalid_intent_is_rejected():
    golden = [golden_record("one")]
    second = [second_record("one", intent="not_an_intent")]

    with pytest.raises(AnnotationValidationError, match="Invalid second_intent"):
        validate_second_annotations(golden, second, expected_count=1)


def test_invalid_action_is_rejected():
    golden = [golden_record("one")]
    second = [second_record("one", action="ignore")]

    with pytest.raises(AnnotationValidationError, match="Invalid second_action"):
        validate_second_annotations(golden, second, expected_count=1)


def test_incomplete_annotation_is_rejected():
    golden = [golden_record("one")]
    second = [second_record("one")]
    second[0]["second_escalation_reason"] = ""

    with pytest.raises(AnnotationValidationError, match="1 annotations remain incomplete"):
        validate_second_annotations(golden, second, expected_count=1)
