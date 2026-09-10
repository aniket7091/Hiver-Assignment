"""LLM-as-judge orchestration, caching, and metric aggregation."""

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.golden_set import (
    LLM_JUDGE_METRICS_FILE,
    LLM_JUDGE_RESULTS_FILE,
    RESULTS_DIR,
    load_jsonl,
    save_jsonl,
)
from src.evaluation.reply_judge import judge_reply


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default: judge 100 examples.
#
# To judge all 200:
#
#     export LLM_JUDGE_N=200
#
# To disable:
#
#     export LLM_JUDGE_ENABLED=0
#

LLM_JUDGE_N = int(os.getenv("LLM_JUDGE_N", "100"))

LLM_JUDGE_ENABLED = os.getenv("LLM_JUDGE_ENABLED", "1") != "0"

LLM_JUDGE_DELAY = float(os.getenv("LLM_JUDGE_DELAY", "0"))


# ---------------------------------------------------------------------------
# Cache-key helpers
# ---------------------------------------------------------------------------

# Changing this constant intentionally invalidates every previous entry when
# the judge prompt or the cache-input contract changes.
LLM_JUDGE_CACHE_SCHEMA_VERSION = "reply-judge-cache-v1"


def _normalize_for_cache(value: Any) -> Any:
    """Return a recursively JSON-safe value with deterministic structure."""

    if isinstance(value, dict):
        return {
            str(key): _normalize_for_cache(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_for_cache(item) for item in value]

    if isinstance(value, float):
        # JSON has no standard representation for NaN/infinity. Treating
        # those explicitly prevents two serializers from producing a subtly
        # different cache key for the same diagnostics.
        if math.isnan(value):
            return "__NaN__"
        if math.isinf(value):
            return "__Infinity__" if value > 0 else "__-Infinity__"
        return value

    if value is None or isinstance(value, (str, int, bool)):
        return value

    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for cache-key generation."""

    return json.dumps(
        _normalize_for_cache(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def get_retrieved_evidence(
    example: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get the complete retrieval snapshot, with legacy-result fallback."""

    evidence = (
        example.get("retrieved_evidence")
        or example.get("retrieval")
        or example.get("evidence")
        or []
    )

    return [item for item in evidence if isinstance(item, dict)]


def build_llm_judge_cache_payload(
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full, agent-derived input contract for a judgement cache."""

    return {
        "cache_schema_version": LLM_JUDGE_CACHE_SCHEMA_VERSION,
        "judge_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "example_id": str(example.get("example_id", "")),
        "customer_message": example.get("customer_message", ""),
        "predicted_intent": example.get("predicted_intent", ""),
        "intent_confidence": example.get("intent_confidence"),
        # This contains each historical customer case, AmazonHelp response,
        # rank, score, conversation ID, and any retrieval metadata emitted by
        # the current agent. It must not be reduced to only the final reply.
        "retrieved_evidence": get_retrieved_evidence(example),
        "top_retrieval_score": example.get("top_retrieval_score", 0.0),
        "second_retrieval_score": example.get("second_retrieval_score", 0.0),
        "retrieval_score_gap": example.get("retrieval_score_gap", 0.0),
        "generated_reply": example.get("reply", ""),
        "action": example.get("predicted_action", ""),
    }


def build_llm_judge_cache_key(
    example: Dict[str, Any],
) -> str:
    """Hash the exact current agent output being submitted to the judge."""

    return hashlib.sha256(
        canonical_json(
            build_llm_judge_cache_payload(example)
        ).encode("utf-8")
    ).hexdigest()


def get_judge_evidence(
    example: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Adapt complete retrieval results to the judge's evidence contract."""

    adapted = []
    for item in get_retrieved_evidence(example):
        adapted.append({
            "rank": item.get("rank"),
            "score": item.get("score", 0.0),
            "conversation_id": item.get("conversation_id"),
            "customer_message": item.get("customer_message", ""),
            "response": (
                item.get("historical_amazonhelp_response")
                or item.get("response")
                or item.get("historical_response")
                or ""
            ),
            "candidate_intent": item.get("candidate_intent"),
            "intent_compatible": item.get("intent_compatible"),
        })
    return adapted


def _call_groq_judge(
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """Judge only current agent output and current retrieval evidence."""

    return judge_reply(
        customer_message=example.get("customer_message", ""),
        agent_reply=example.get("reply", ""),
        retrieved_evidence=get_judge_evidence(example),
        predicted_intent=example.get("predicted_intent"),
        predicted_action=example.get("predicted_action"),
    )


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cached_judgements() -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]:
    """Load cache records indexed by their content hash and example ID."""

    records = load_jsonl(LLM_JUDGE_RESULTS_FILE)
    cache_by_key: Dict[str, Dict[str, Any]] = {}
    records_by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        example_id = str(record.get("example_id", ""))
        if example_id:
            records_by_example[example_id].append(record)

        cache_key = record.get("cache_key")
        if cache_key:
            # A later write supersedes an earlier duplicate for the same exact
            # evaluated input. Saving below removes such duplicates.
            cache_by_key[str(cache_key)] = record

    return records, cache_by_key, records_by_example


def lookup_cached_judgement(
    cache_by_key: Dict[str, Dict[str, Any]],
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    """Return a reusable judgement only for an exact cache-key match."""

    return cache_by_key.get(cache_key)


def deduplicate_cache_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one latest record per content key while retaining legacy rows."""

    seen_keys = set()
    deduplicated_reversed = []

    for record in reversed(records):
        cache_key = record.get("cache_key")
        if cache_key:
            if cache_key in seen_keys:
                continue
            seen_keys.add(cache_key)
        deduplicated_reversed.append(record)

    return list(reversed(deduplicated_reversed))


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_llm_judge_examples(predictions, n):
    """
    Prioritize difficult examples.

    This is not a gold-labeling step. The selection only
    determines which generated replies are inspected by
    the independent LLM judge.
    """

    def priority(example):
        score = 0

        # Intent failure
        if not example["intent_correct"]:
            score += 3

        # Action failure
        if not example["action_correct"]:
            score += 3

        # Weak retrieval
        if example["top_retrieval_score"] < 0.45:
            score += 2

        # Ambiguous retrieval
        if (
            example["top_retrieval_score"] > 0
            and example["retrieval_score_gap"] < 0.05
        ):
            score += 2

        return (
            score,
            -float(example.get("top_retrieval_score", 0.0)),
        )

    ordered = sorted(predictions, key=priority, reverse=True)
    return ordered[:min(n, len(predictions))]


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate_llm_judge_metrics(judged_records):

    metric_names = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall",
    ]

    values = {name: [] for name in metric_names}

    for record in judged_records:
        judge = record.get("judge", {})
        if not isinstance(judge, dict):
            continue

        for name in metric_names:
            value = judge.get(name)
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 5:
                values[name].append(value)

    result = {"num_judged": len(judged_records)}

    for name in metric_names:
        vals = values[name]
        if vals:
            result[f"mean_{name}"] = round(sum(vals) / len(vals), 4)
        else:
            result[f"mean_{name}"] = None

    # Threshold metrics

    overall = values["overall"]
    groundedness = values["groundedness"]
    helpfulness = values["helpfulness"]

    result["overall_4_plus_rate"] = (
        round(sum(v >= 4 for v in overall) / len(overall), 4)
        if overall else None
    )

    result["groundedness_4_plus_rate"] = (
        round(sum(v >= 4 for v in groundedness) / len(groundedness), 4)
        if groundedness else None
    )

    result["helpfulness_4_plus_rate"] = (
        round(sum(v >= 4 for v in helpfulness) / len(helpfulness), 4)
        if helpfulness else None
    )

    return result


# ---------------------------------------------------------------------------
# Save judge results
# ---------------------------------------------------------------------------

def save_judge_results(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_JUDGE_RESULTS_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main LLM judge workflow
# ---------------------------------------------------------------------------

def run_llm_judge(predictions, _golden_records):

    if not LLM_JUDGE_ENABLED:
        result = {
            "enabled": False,
            "available": False,
            "status": "not_refreshed",
            "message": (
                "LLM judge disabled using LLM_JUDGE_ENABLED=0. Existing cached "
                "judgements are not evaluation results for this run."
            ),
            "results_file": str(LLM_JUDGE_RESULTS_FILE),
        }

        # Do not leave a previous run's numeric LLM scores looking like the
        # outcome of this one when a caller intentionally disables judging.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LLM_JUDGE_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    selected = select_llm_judge_examples(predictions, LLM_JUDGE_N)

    # Golden labels are deliberately not passed to the LLM judge or included
    # in cache keys. The judge evaluates only the current agent output.
    (
        cache_records,
        cache_by_key,
        records_by_example,
    ) = load_cached_judgements()

    persisted_cache_records = deduplicate_cache_records(cache_records)

    print("\n")
    print("=" * 70)
    print("GROQ LLM-AS-JUDGE")
    print("=" * 70)

    print(f"Requested examples : {LLM_JUDGE_N}")
    print(f"Selected examples  : {len(selected)}")

    valid_cached_judgements = 0
    stale_cache_entries = 0
    new_api_calls = 0
    errors = 0
    final_judgements = []

    for index, example in enumerate(selected, start=1):

        example_id = str(example["example_id"])

        cache_key = build_llm_judge_cache_key(example)
        cached_judgement = lookup_cached_judgement(cache_by_key, cache_key)

        # A matching example ID is not enough. Records without a key (the old
        # format) and entries whose key differs are stale for this exact
        # current agent output and cannot be reused.
        if cached_judgement is not None:
            valid_cached_judgements += 1
            final_judgements.append(cached_judgement)
            continue

        stale_cache_entries += sum(
            1
            for record in records_by_example.get(example_id, [])
            if record.get("cache_key") != cache_key
        )

        # Call Groq

        try:
            judge_result = _call_groq_judge(example)

            record = {
                "example_id": example_id,
                "customer_message": example["customer_message"],
                "cache_key": cache_key,
                "cache_input": build_llm_judge_cache_payload(example),
                "intent_confidence": example.get("intent_confidence"),
                "reply": example.get("reply", ""),
                "gold_intent": example["gold_intent"],
                "predicted_intent": example["predicted_intent"],
                "gold_action": example["gold_action"],
                "predicted_action": example["predicted_action"],
                "action": example["predicted_action"],
                "top_retrieval_score": example.get("top_retrieval_score", 0.0),
                "second_retrieval_score": example.get("second_retrieval_score", 0.0),
                "retrieval_score_gap": example.get("retrieval_score_gap", 0.0),
                "retrieved_evidence": get_retrieved_evidence(example),
                "judge": judge_result,
                # Keep scores accessible to downstream JSONL consumers while
                # retaining the nested legacy-compatible `judge` object.
                "relevance": judge_result["relevance"],
                "groundedness": judge_result["groundedness"],
                "helpfulness": judge_result["helpfulness"],
                "safety": judge_result["safety"],
                "overall": judge_result["overall"],
                "reason": judge_result["reason"],
                "model": judge_result["model"],
                "provider": judge_result["provider"],
            }

            final_judgements.append(record)
            persisted_cache_records.append(record)
            cache_by_key[cache_key] = record

            new_api_calls += 1

            # Save immediately
            save_judge_results(persisted_cache_records)

            print(f"Judged {index}/{len(selected)} - {example_id}")

            if LLM_JUDGE_DELAY > 0:
                time.sleep(LLM_JUDGE_DELAY)

        except Exception as exc:
            errors += 1
            print(f"Judge error on {example_id}: {exc}")

    metrics = aggregate_llm_judge_metrics(final_judgements)

    metrics.update({
        "enabled": True,
        "available": bool(final_judgements),
        "requested_examples": LLM_JUDGE_N,
        "selected_examples": len(selected),
        "judged_examples": len(final_judgements),
        "final_judgements": len(final_judgements),
        "valid_cached_judgements": valid_cached_judgements,
        "stale_cache_entries": stale_cache_entries,
        "new_api_calls": new_api_calls,
        "errors": errors,
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "provider": "groq",
        "results_file": str(LLM_JUDGE_RESULTS_FILE),
    })

    print(f"Valid cached judgements  : {valid_cached_judgements}")
    print(f"Stale cache entries      : {stale_cache_entries}")
    print(f"New API calls            : {new_api_calls}")
    print(f"Final judgements         : {len(final_judgements)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_JUDGE_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
