"""Golden-set I/O, path constants, and validation helpers."""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.intent.schemas import INTENTS


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

GOLDEN_FILE = ROOT_DIR / "data" / "golden" / "golden_set.jsonl"

RESULTS_DIR = ROOT_DIR / "results"

METRICS_FILE = RESULTS_DIR / "metrics.json"
EXAMPLES_FILE = RESULTS_DIR / "evaluation_examples.jsonl"
FAILURES_FILE = RESULTS_DIR / "failure_analysis.jsonl"
LLM_JUDGE_RESULTS_FILE = RESULTS_DIR / "llm_judge_results.jsonl"
LLM_JUDGE_METRICS_FILE = RESULTS_DIR / "llm_judge_metrics.json"


# ---------------------------------------------------------------------------
# Valid label sets (canonical source: src/intent/schemas.py)
# ---------------------------------------------------------------------------

VALID_INTENTS = list(INTENTS)

VALID_ACTIONS = [
    "auto_handle",
    "escalate",
]


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def save_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Golden-set loading and validation
# ---------------------------------------------------------------------------

def load_golden_set() -> List[Dict[str, Any]]:
    """Load the manually labelled golden evaluation set."""
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN_FILE}")
    records = load_jsonl(GOLDEN_FILE)
    if not records:
        raise ValueError("Golden set is empty.")
    return records


def validate_golden_set(
    records: List[Dict[str, Any]],
) -> None:
    required_fields = [
        "example_id",
        "customer_message",
        "gold_intent",
        "gold_action",
        "gold_escalation_reason",
    ]

    for i, record in enumerate(records, start=1):
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"Example {i} is missing field: {field}"
                )

        intent = str(record["gold_intent"]).strip()
        action = str(record["gold_action"]).strip()

        if intent not in VALID_INTENTS:
            raise ValueError(
                f"Invalid intent '{intent}' in example "
                f"{record['example_id']}"
            )

        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{action}' in example "
                f"{record['example_id']}"
            )

        if not str(record["gold_escalation_reason"]).strip():
            raise ValueError(
                f"Missing escalation reason in example "
                f"{record['example_id']}"
            )

    print(f"Loaded golden examples: {len(records)}")
