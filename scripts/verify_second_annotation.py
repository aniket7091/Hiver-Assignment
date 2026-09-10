"""Fail clearly until the independent second annotation is complete."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


from src.evaluation.human_agreement import (
    AnnotationValidationError,
    GOLDEN_FILE,
    SECOND_ANNOTATION_FILE,
    load_jsonl,
    validate_second_annotations,
)


def main() -> None:
    if not SECOND_ANNOTATION_FILE.exists():
        print(
            "Second annotation file is missing. Run "
            "python scripts/create_second_annotation_set.py first."
        )
        raise SystemExit(1)

    try:
        validate_second_annotations(
            load_jsonl(GOLDEN_FILE),
            load_jsonl(SECOND_ANNOTATION_FILE),
        )
    except AnnotationValidationError as error:
        print(f"Second annotation verification failed: {error}")
        raise SystemExit(1)

    print("Second annotation verification passed: 40 complete annotations.")


if __name__ == "__main__":
    main()
