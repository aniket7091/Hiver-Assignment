"""Compatibility entry point for the golden-set evaluation workflow."""

from src.evaluation.runner import main
from src.evaluation.llm_judge import (
    build_llm_judge_cache_key,
    canonical_json,
    get_retrieved_evidence,
    lookup_cached_judgement,
)


__all__ = (
    "build_llm_judge_cache_key",
    "canonical_json",
    "get_retrieved_evidence",
    "lookup_cached_judgement",
    "main",
)


if __name__ == "__main__":
    main()
