"""End-to-end golden-set evaluation workflow orchestration."""

import pandas as pd

from src.evaluation.golden_set import (
    EXAMPLES_FILE,
    FAILURES_FILE,
    GOLDEN_FILE,
    LLM_JUDGE_METRICS_FILE,
    LLM_JUDGE_RESULTS_FILE,
    VALID_INTENTS,
    load_golden_set,
    save_jsonl,
    validate_golden_set,
)
from src.evaluation.baselines import run_trivial_baseline, run_tfidf_baseline
from src.evaluation.agent_eval import run_agent_evaluation
from src.evaluation.llm_judge import (
    build_llm_judge_cache_key,
    canonical_json,
    get_retrieved_evidence,
    lookup_cached_judgement,
    run_llm_judge,
)
from src.evaluation.failures import identify_failures
from src.evaluation.reporting import (
    print_agent_results,
    print_baseline_results,
    print_failure_summary,
    print_llm_judge_results,
    save_metrics,
)
from src.evaluation.human_agreement import (
    calculate_human_agreement as calculate_independent_human_agreement,
)


def main():

    print("=" * 70)
    print("HIVER CUSTOMER SUPPORT AGENT - FULL EVALUATION")
    print("=" * 70)

    # Load golden set

    records = load_golden_set()

    # Validate

    validate_golden_set(records)

    df = pd.DataFrame(records)

    print(f"\nEvaluating {len(df)} golden examples...")

    # Baseline 1

    print("\nRunning trivial majority baseline...")

    trivial_result = run_trivial_baseline(df, VALID_INTENTS)

    # Baseline 2

    print("\nRunning TF-IDF + Logistic Regression baseline...")

    tfidf_result = run_tfidf_baseline(df, VALID_INTENTS)

    # Actual Agent

    print("\nRunning end-to-end AmazonHelp agent...")

    agent_result = run_agent_evaluation(records)

    # Human agreement

    print("\nChecking human annotation agreement...")

    human_agreement = calculate_independent_human_agreement()

    # Failure analysis

    failures = identify_failures(agent_result)

    # Save per-example agent results

    save_jsonl(EXAMPLES_FILE, agent_result["predictions"])

    # Save failures

    save_jsonl(FAILURES_FILE, failures)

    # Groq LLM-as-judge

    print("\nRunning Groq LLM-as-judge...")

    llm_judge_result = run_llm_judge(
        agent_result["predictions"],
        records,
    )

    # Aggregate all results

    results = {
        "evaluation_metadata": {
            "golden_examples": len(records),
            "golden_file": str(GOLDEN_FILE),
            "note": (
                "The golden set is used only for evaluation. "
                "The production intent classifier is trained "
                "outside the golden set."
            ),
        },
        "trivial": trivial_result,
        "tfidf": tfidf_result,
        "end_to_end": agent_result,
        "llm_judge": llm_judge_result,
        "human_agreement": human_agreement,
        "failure_analysis": {
            "total_failure_records": len(failures),
            "file": str(FAILURES_FILE),
        },
    }

    # Print results

    print_baseline_results(results)

    print_agent_results(agent_result)

    print_llm_judge_results(llm_judge_result)

    print_failure_summary(failures)

    # Human agreement output

    print("\n")
    print("=" * 70)
    print("HUMAN AGREEMENT")
    print("=" * 70)

    if human_agreement.get("available", False):
        print(
            "Examples annotated by both humans: "
            f"{human_agreement['n_examples']}"
        )
        print(
            "Intent raw agreement: "
            f"{human_agreement['intent']['raw_agreement'] * 100:.2f}%"
        )
        print(
            "Intent Cohen's kappa: "
            f"{human_agreement['intent']['cohen_kappa']:.4f}"
        )
        print(
            "Action raw agreement: "
            f"{human_agreement['action']['raw_agreement'] * 100:.2f}%"
        )
        print(
            "Action Cohen's kappa: "
            f"{human_agreement['action']['cohen_kappa']:.4f}"
        )
    else:
        print("INCOMPLETE ANNOTATION WARNING")
        print(human_agreement["message"])

    # Save aggregate metrics

    save_metrics(results)

    # Output paths

    print(f"\nPer-example results saved to:\n{EXAMPLES_FILE}")
    print(f"\nFailure analysis saved to:\n{FAILURES_FILE}")

    if llm_judge_result.get("enabled", False):
        print(f"\nLLM judge results saved to:\n{LLM_JUDGE_RESULTS_FILE}")
        print(f"\nLLM judge metrics saved to:\n{LLM_JUDGE_METRICS_FILE}")
    else:
        print(
            "\nLLM judge results were not refreshed because the judge was "
            "disabled. Existing cache rows, if any, must not be interpreted "
            "as results for this run."
        )
        print(
            f"\nLLM judge metrics were marked unavailable at:"
            f"\n{LLM_JUDGE_METRICS_FILE}"
        )

    print("\n")
    print("=" * 70)
    print("FULL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
