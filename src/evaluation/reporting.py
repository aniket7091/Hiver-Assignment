"""Console output and metric persistence for evaluation results."""

import json
from typing import Any, Dict, List

from src.evaluation.golden_set import METRICS_FILE, RESULTS_DIR


# ---------------------------------------------------------------------------
# Save aggregate metrics
# ---------------------------------------------------------------------------

def save_metrics(results: Dict[str, Any]) -> None:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clean_results = {}
    for name, result in results.items():
        if isinstance(result, dict):
            clean_results[name] = {
                key: value
                for key, value in result.items()
                if key != "predictions"
            }
        else:
            clean_results[name] = result

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics saved to:\n{METRICS_FILE}")


# ---------------------------------------------------------------------------
# Print baseline results
# ---------------------------------------------------------------------------

def print_baseline_results(results: Dict[str, Any]) -> None:

    print("\n")
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    for name in ["trivial", "tfidf"]:
        if name not in results:
            continue

        result = results[name]

        print("\n" + "-" * 70)
        print(result["name"])

        if "majority_intent" in result:
            print(f"Majority intent : {result['majority_intent']}")

        print(f"Accuracy        : {result['accuracy']:.4f}")
        print(f"Macro F1        : {result['macro_f1']:.4f}")
        print(f"Weighted F1     : {result['weighted_f1']:.4f}")


# ---------------------------------------------------------------------------
# Print agent results
# ---------------------------------------------------------------------------

def print_agent_results(result: Dict[str, Any]) -> None:

    print("\n")
    print("=" * 70)
    print("END-TO-END AGENT EVALUATION")
    print("=" * 70)

    print("\nINTENT")
    print(f"Accuracy        : {result['intent']['accuracy']:.4f}")
    print(f"Macro F1        : {result['intent']['macro_f1']:.4f}")
    print(f"Weighted F1     : {result['intent']['weighted_f1']:.4f}")

    print("\nACTION")
    print(f"Accuracy        : {result['action']['accuracy']:.4f}")
    print(f"Auto-handle precision : {result['action']['auto_handle_precision']:.4f}")
    print(f"Escalation recall     : {result['action']['escalation_recall']:.4f}")

    print("\nRETRIEVAL / REPLY")
    print(f"Grounded reply rate   : {result['grounding']['grounded_reply_rate']:.4f}")
    print(f"Retrieval available   : {result['grounding']['retrieval_available_rate']:.4f}")


# ---------------------------------------------------------------------------
# Print LLM judge results
# ---------------------------------------------------------------------------

def print_llm_judge_results(result: Dict[str, Any]) -> None:

    print("\n")
    print("=" * 70)
    print("GROQ LLM-AS-JUDGE RESULTS")
    print("=" * 70)

    if not result.get("available", False):
        print(result.get("message", "LLM judge unavailable."))
        return

    print(f"Model                 : {result.get('model')}")
    print(f"Provider              : {result.get('provider')}")
    print(f"Judged examples       : {result.get('judged_examples')}")
    print(f"Requested examples    : {result.get('requested_examples')}")
    print(f"Selected examples     : {result.get('selected_examples')}")
    print(f"Valid cached judgements: {result.get('valid_cached_judgements')}")
    print(f"Stale cache entries   : {result.get('stale_cache_entries')}")
    print(f"New API calls         : {result.get('new_api_calls')}")
    print(f"Final judgements      : {result.get('final_judgements')}")
    print(f"Mean relevance        : {result.get('mean_relevance')}")
    print(f"Mean groundedness     : {result.get('mean_groundedness')}")
    print(f"Mean helpfulness      : {result.get('mean_helpfulness')}")
    print(f"Mean safety           : {result.get('mean_safety')}")
    print(f"Mean overall          : {result.get('mean_overall')}")
    print(f"Overall >= 4 rate     : {result.get('overall_4_plus_rate')}")
    print(f"Groundedness >= 4     : {result.get('groundedness_4_plus_rate')}")
    print(f"Helpfulness >= 4      : {result.get('helpfulness_4_plus_rate')}")


# ---------------------------------------------------------------------------
# Print failure summary
# ---------------------------------------------------------------------------

def print_failure_summary(failures: List[Dict[str, Any]]) -> None:

    print("\n")
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    counts = {}
    for failure in failures:
        failure_type = failure["failure_type"]
        counts[failure_type] = counts.get(failure_type, 0) + 1

    for failure_type, count in sorted(
        counts.items(), key=lambda item: item[1], reverse=True,
    ):
        print(f"{failure_type:<35} {count}")

    print(f"\nTotal failure records: {len(failures)}")
