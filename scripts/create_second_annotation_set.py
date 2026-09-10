"""Create a deterministic, blank, independently annotatable 40-case subset."""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.intent.schemas import INTENTS


GOLDEN_FILE = ROOT_DIR / "data" / "golden" / "golden_set.jsonl"
SECOND_ANNOTATION_FILE = (
    ROOT_DIR / "data" / "golden" / "second_annotation.jsonl"
)
SAMPLE_SIZE = 40
RANDOM_SEED = 42


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def stratified_quotas(
    records: List[Dict[str, Any]],
    sample_size: int,
) -> Dict[str, int]:
    """Allocate a proportional, deterministic quota for every intent class."""

    counts = Counter(str(record["gold_intent"]).strip() for record in records)
    total = len(records)

    if sample_size > total:
        raise ValueError("Sample size cannot exceed the golden-set size.")

    quotas = {
        intent: int(sample_size * counts[intent] // total)
        for intent in INTENTS
    }

    remaining = sample_size - sum(quotas.values())
    ranked_intents = sorted(
        INTENTS,
        key=lambda intent: (
            -(sample_size * counts[intent] / total - quotas[intent]),
            intent,
        ),
    )

    for intent in ranked_intents:
        if not remaining:
            break
        if quotas[intent] < counts[intent]:
            quotas[intent] += 1
            remaining -= 1

    if remaining:
        raise RuntimeError("Could not allocate the requested stratified sample.")

    return quotas


def select_stratified_subset(
    records: List[Dict[str, Any]],
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Sample by gold intent only for representative selection, never labeling."""

    by_intent = defaultdict(list)
    for record in records:
        intent = str(record.get("gold_intent", "")).strip()
        if intent not in INTENTS:
            raise ValueError(
                f"Golden record {record.get('example_id')} has invalid intent: "
                f"{intent!r}."
            )
        by_intent[intent].append(record)

    quotas = stratified_quotas(records, sample_size)
    rng = random.Random(seed)
    selected = []

    for intent in INTENTS:
        candidates = sorted(
            by_intent[intent],
            key=lambda record: str(record["example_id"]),
        )
        selected.extend(rng.sample(candidates, quotas[intent]))

    return sorted(selected, key=lambda record: str(record["example_id"]))


def annotation_record(record: Dict[str, Any]) -> Dict[str, str]:
    """Copy case context only; never copy any first-annotator labels."""

    return {
        "example_id": str(record["example_id"]),
        "customer_message": str(record.get("customer_message", "")),
        "thread_context": str(record.get("thread_context", "")),
        "historical_amazonhelp_response": str(
            record.get("historical_amazonhelp_response", "")
        ),
        "second_intent": "",
        "second_action": "",
        "second_escalation_reason": "",
        "second_annotator_notes": "",
    }


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    golden_records = load_jsonl(GOLDEN_FILE)
    selected = select_stratified_subset(golden_records)
    annotation_records = [annotation_record(record) for record in selected]
    write_jsonl(SECOND_ANNOTATION_FILE, annotation_records)

    distribution = Counter(
        str(record["gold_intent"]).strip()
        for record in selected
    )

    print("Created independent second-annotation set")
    print(f"Seed             : {RANDOM_SEED}")
    print(f"Sampled examples : {len(selected)}")
    print(f"Output           : {SECOND_ANNOTATION_FILE}")
    print("\nSelected IDs:")
    for record in selected:
        print(record["example_id"])

    print("\nSampling distribution (used only to stratify selection):")
    for intent in INTENTS:
        print(f"{intent:<28} {distribution[intent]}")

    print(
        "\nAll second-human label fields are blank. A human must complete "
        "them independently before agreement can be calculated."
    )


if __name__ == "__main__":
    main()
