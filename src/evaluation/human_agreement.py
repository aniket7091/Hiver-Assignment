"""Validate independent annotations and calculate inter-annotator agreement."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sklearn.metrics import cohen_kappa_score

from src.intent.schemas import INTENTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = PROJECT_ROOT / "data" / "golden" / "golden_set.jsonl"
SECOND_ANNOTATION_FILE = (
    PROJECT_ROOT / "data" / "golden" / "second_annotation.jsonl"
)
RESULTS_FILE = PROJECT_ROOT / "results" / "human_agreement.json"

VALID_INTENTS = set(INTENTS)
VALID_ACTIONS = {"auto_handle", "escalate"}
REQUIRED_SECOND_FIELDS = (
    "second_intent",
    "second_action",
    "second_escalation_reason",
)


class AnnotationValidationError(ValueError):
    """Raised when independent annotations cannot be compared safely."""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file while reporting malformed records precisely."""

    records = []

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise AnnotationValidationError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise AnnotationValidationError(
                    f"Record {line_number} in {path} must be a JSON object."
                )

            records.append(record)

    return records


def save_json(path: Path, value: Dict[str, Any]) -> None:
    """Persist the current agreement status, including an incomplete status."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def missing_annotation_count(
    second_records: Sequence[Dict[str, Any]],
) -> int:
    """Count records missing at least one required second-human field."""

    return sum(
        any(not str(record.get(field, "")).strip() for field in REQUIRED_SECOND_FIELDS)
        for record in second_records
    )


def validate_second_annotations(
    golden_records: Sequence[Dict[str, Any]],
    second_records: Sequence[Dict[str, Any]],
    expected_count: int = 40,
) -> None:
    """Validate that a complete independent annotation set is comparable."""

    if len(second_records) != expected_count:
        raise AnnotationValidationError(
            f"Expected exactly {expected_count} second annotations, found "
            f"{len(second_records)}."
        )

    golden_ids = {
        str(record.get("example_id", ""))
        for record in golden_records
    }
    second_ids = [
        str(record.get("example_id", "")).strip()
        for record in second_records
    ]

    if any(not example_id for example_id in second_ids):
        raise AnnotationValidationError(
            "Every second annotation must contain an example_id."
        )

    duplicate_ids = sorted(
        example_id
        for example_id, count in Counter(second_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise AnnotationValidationError(
            "Duplicate second annotation example_id(s): "
            + ", ".join(duplicate_ids)
        )

    unknown_ids = sorted(set(second_ids) - golden_ids)
    if unknown_ids:
        raise AnnotationValidationError(
            "Second annotation contains IDs not in golden_set.jsonl: "
            + ", ".join(unknown_ids)
        )

    incomplete = missing_annotation_count(second_records)
    if incomplete:
        raise AnnotationValidationError(
            f"{incomplete} annotations remain incomplete. A human must fill "
            "second_intent, second_action, and second_escalation_reason."
        )

    for record in second_records:
        example_id = str(record["example_id"])
        intent = str(record["second_intent"]).strip()
        action = str(record["second_action"]).strip()

        if intent not in VALID_INTENTS:
            raise AnnotationValidationError(
                f"Invalid second_intent for {example_id}: {intent!r}."
            )

        if action not in VALID_ACTIONS:
            raise AnnotationValidationError(
                f"Invalid second_action for {example_id}: {action!r}."
            )


def _distribution(labels: Sequence[str]) -> Dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def _raw_agreement(
    first_labels: Sequence[str],
    second_labels: Sequence[str],
) -> float:
    return sum(
        first == second
        for first, second in zip(first_labels, second_labels)
    ) / len(first_labels)


def calculate_agreement(
    golden_records: Sequence[Dict[str, Any]],
    second_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate agreement for a validated set of independently labeled IDs."""

    golden_by_id = {
        str(record["example_id"]): record
        for record in golden_records
    }
    aligned_first = [
        golden_by_id[str(record["example_id"])]
        for record in second_records
    ]

    first_intents = [str(record["gold_intent"]).strip() for record in aligned_first]
    second_intents = [
        str(record["second_intent"]).strip()
        for record in second_records
    ]
    first_actions = [str(record["gold_action"]).strip() for record in aligned_first]
    second_actions = [
        str(record["second_action"]).strip()
        for record in second_records
    ]
    first_reasons = [
        str(record.get("gold_escalation_reason", "")).strip()
        for record in aligned_first
    ]
    second_reasons = [
        str(record["second_escalation_reason"]).strip()
        for record in second_records
    ]

    return {
        "available": True,
        "n_examples": len(second_records),
        "intent": {
            "raw_agreement": round(
                _raw_agreement(first_intents, second_intents),
                4,
            ),
            "cohen_kappa": round(
                float(cohen_kappa_score(first_intents, second_intents)),
                4,
            ),
            "annotator_1_distribution": _distribution(first_intents),
            "annotator_2_distribution": _distribution(second_intents),
        },
        "action": {
            "raw_agreement": round(
                _raw_agreement(first_actions, second_actions),
                4,
            ),
            "cohen_kappa": round(
                float(cohen_kappa_score(first_actions, second_actions)),
                4,
            ),
            "annotator_1_distribution": _distribution(first_actions),
            "annotator_2_distribution": _distribution(second_actions),
        },
        "escalation_reason": {
            "raw_agreement": round(
                _raw_agreement(first_reasons, second_reasons),
                4,
            ),
        },
    }


def calculate_human_agreement(
    golden_file: Path = GOLDEN_FILE,
    second_annotation_file: Path = SECOND_ANNOTATION_FILE,
    results_file: Optional[Path] = RESULTS_FILE,
) -> Dict[str, Any]:
    """Load, validate, calculate, and persist independent-human agreement."""

    if not second_annotation_file.exists():
        result = {
            "available": False,
            "status": "missing",
            "message": (
                "No second annotation file was found. Run "
                "scripts/create_second_annotation_set.py, then have an "
                "independent human complete it."
            ),
        }
    else:
        try:
            golden_records = load_jsonl(golden_file)
            second_records = load_jsonl(second_annotation_file)
            validate_second_annotations(golden_records, second_records)
            result = calculate_agreement(golden_records, second_records)
        except AnnotationValidationError as error:
            result = {
                "available": False,
                "status": "incomplete_or_invalid",
                "message": str(error),
            }

    if results_file is not None:
        save_json(Path(results_file), result)

    return result

if __name__ == "__main__":
    result = calculate_human_agreement()
    print(json.dumps(result, indent=2, ensure_ascii=False))
