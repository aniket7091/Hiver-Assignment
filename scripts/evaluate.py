import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)



# PROJECT PATHS


ROOT_DIR = Path(__file__).resolve().parents[1]

GOLDEN_FILE = (
    ROOT_DIR
    / "data"
    / "golden"
    / "golden_set.jsonl"
)

RESULTS_DIR = ROOT_DIR / "results"

METRICS_FILE = (
    RESULTS_DIR / "metrics.json"
)

EXAMPLES_FILE = (
    RESULTS_DIR / "evaluation_examples.jsonl"
)

FAILURES_FILE = (
    RESULTS_DIR / "failure_analysis.jsonl"
)

LLM_JUDGE_RESULTS_FILE = (
    RESULTS_DIR / "llm_judge_results.jsonl"
)

LLM_JUDGE_METRICS_FILE = (
    RESULTS_DIR / "llm_judge_metrics.json"
)



# IMPORT PROJECT MODULES


if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.agent.pipeline import AmazonHelpAgent
from src.evaluation.human_agreement import (
    calculate_human_agreement as calculate_independent_human_agreement,
)
from src.evaluation.reply_judge import judge_reply



# VALID LABELS


VALID_INTENTS = [
    "delivery_issue",
    "order_issue",
    "shipping_issue",
    "payment_issue",
    "refund_issue",
    "account_issue",
    "prime_video_issue",
    "device_issue",
    "product_or_content_issue",
    "support_complaint",
    "other",
]


VALID_ACTIONS = [
    "auto_handle",
    "escalate",
]



# CONFIGURATION


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

LLM_JUDGE_N = int(
    os.getenv(
        "LLM_JUDGE_N",
        "100",
    )
)


LLM_JUDGE_ENABLED = (
    os.getenv(
        "LLM_JUDGE_ENABLED",
        "1",
    )
    != "0"
)


LLM_JUDGE_DELAY = float(
    os.getenv(
        "LLM_JUDGE_DELAY",
        "0",
    )
)



# JSONL HELPERS


def save_jsonl(
    path: Path,
    records: List[Dict[str, Any]],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(
    path: Path,
) -> List[Dict[str, Any]]:

    if not path.exists():
        return []

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records



# LOAD GOLDEN SET


def load_golden_set() -> List[Dict[str, Any]]:
    """
    Load the manually labelled golden evaluation set.
    """

    if not GOLDEN_FILE.exists():

        raise FileNotFoundError(
            f"Golden set not found: "
            f"{GOLDEN_FILE}"
        )

    records = load_jsonl(
        GOLDEN_FILE
    )

    if not records:

        raise ValueError(
            "Golden set is empty."
        )

    return records



# VALIDATE GOLDEN SET


def validate_golden_set(
    records: List[Dict[str, Any]]
) -> None:

    required_fields = [
        "example_id",
        "customer_message",
        "gold_intent",
        "gold_action",
        "gold_escalation_reason",
    ]

    for i, record in enumerate(
        records,
        start=1,
    ):

        for field in required_fields:

            if field not in record:

                raise ValueError(
                    f"Example {i} is missing "
                    f"field: {field}"
                )

        intent = str(
            record["gold_intent"]
        ).strip()

        action = str(
            record["gold_action"]
        ).strip()

        if intent not in VALID_INTENTS:

            raise ValueError(
                f"Invalid intent '{intent}' "
                f"in example "
                f"{record['example_id']}"
            )

        if action not in VALID_ACTIONS:

            raise ValueError(
                f"Invalid action '{action}' "
                f"in example "
                f"{record['example_id']}"
            )

        if not str(
            record[
                "gold_escalation_reason"
            ]
        ).strip():

            raise ValueError(
                "Missing escalation reason "
                f"in example "
                f"{record['example_id']}"
            )

    print(
        f"Loaded golden examples: "
        f"{len(records)}"
    )



# BASELINE 1: TRIVIAL MAJORITY


def run_trivial_baseline(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Majority-class baseline.

    Predicts the most frequent intent for
    every golden example.
    """

    majority_intent = (
        df["gold_intent"]
        .value_counts()
        .idxmax()
    )

    predictions = [
        majority_intent
    ] * len(df)

    accuracy = accuracy_score(
        df["gold_intent"],
        predictions,
    )

    macro_f1 = f1_score(
        df["gold_intent"],
        predictions,
        labels=VALID_INTENTS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        df["gold_intent"],
        predictions,
        labels=VALID_INTENTS,
        average="weighted",
        zero_division=0,
    )

    return {
        "name": (
            "trivial_majority_baseline"
        ),
        "majority_intent": (
            majority_intent
        ),
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "macro_f1": round(
            float(macro_f1),
            4,
        ),
        "weighted_f1": round(
            float(weighted_f1),
            4,
        ),
        "predictions": predictions,
    }



# BASELINE 2: TF-IDF + LOGISTIC REGRESSION


def run_tfidf_baseline(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Simple lexical baseline:

        TF-IDF
             +
        Logistic Regression

    Uses 5-fold cross-validation so that the
    golden examples are not evaluated using
    models trained on those same examples.
    """

    X = (
        df["customer_message"]
        .astype(str)
    )

    y = (
        df["gold_intent"]
        .astype(str)
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        labels=VALID_INTENTS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        labels=VALID_INTENTS,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=VALID_INTENTS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "name": (
            "tfidf_logistic_regression"
        ),
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "macro_f1": round(
            float(macro_f1),
            4,
        ),
        "weighted_f1": round(
            float(weighted_f1),
            4,
        ),
        "classification_report": (
            report
        ),
        "predictions": (
            predictions.tolist()
        ),
    }



# ACTUAL AGENT EVALUATION


def run_agent_evaluation(
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    print(
        "\nLoading production agent..."
    )

    agent = AmazonHelpAgent(
        top_k=5,
        min_evidence_score=0.35,
    )

    print(
        "Production agent loaded."
    )

    predictions = []

    intent_predictions = []
    intent_gold = []

    action_predictions = []
    action_gold = []

    grounded_count = 0

    retrieval_available_count = 0

    for index, record in enumerate(
        records,
        start=1,
    ):

        customer_message = str(
            record[
                "customer_message"
            ]
        ).strip()

        gold_intent = str(
            record["gold_intent"]
        ).strip()

        gold_action = str(
            record["gold_action"]
        ).strip()


        # Run agent


        try:

            result = agent.process(
                customer_message
            )

        except Exception as exc:

            print(
                f"\nERROR on "
                f"{record['example_id']}: "
                f"{exc}"
            )

            result = {

                "intent": "other",

                "intent_confidence": 0.0,

                "reply": "",

                "grounded": False,

                "action": "escalate",

                "escalation_reason": (
                    "Agent execution failed."
                ),

                "reason_code": (
                    "agent_error"
                ),

                "top_retrieval_score": 0.0,

                "second_retrieval_score": 0.0,

                "retrieval_score_gap": 0.0,

                "retrieval": [],

                "evidence": [],
            }

        predicted_intent = str(
            result.get(
                "intent",
                "other",
            )
        ).strip().lower()

        predicted_action = str(
            result.get(
                "action",
                "escalate",
            )
        ).strip().lower()

        intent_confidence = result.get(
            "intent_confidence"
        )


        # Intent metrics


        intent_gold.append(
            gold_intent
        )

        intent_predictions.append(
            predicted_intent
        )


        # Action metrics


        action_gold.append(
            gold_action
        )

        action_predictions.append(
            predicted_action
        )


        # Grounding


        grounded = bool(
            result.get(
                "grounded",
                False,
            )
        )

        if grounded:

            grounded_count += 1


        # Retrieval


        retrieval = result.get(
            "retrieval",
            [],
        )

        if retrieval:

            retrieval_available_count += 1


        # Evidence


        evidence = result.get(
            "evidence",
            [],
        )

        top_evidence = None

        if evidence:

            top_evidence = evidence[0]


        # Store per-example result


        example_result = {

            "example_id": (
                record["example_id"]
            ),

            "customer_message": (
                customer_message
            ),

            # Gold labels
            "gold_intent": (
                gold_intent
            ),

            "gold_action": (
                gold_action
            ),

            "gold_escalation_reason": (
                record[
                    "gold_escalation_reason"
                ]
            ),

            # Intent prediction
            "predicted_intent": (
                predicted_intent
            ),

            "intent_correct": (
                predicted_intent
                == gold_intent
            ),

            "intent_confidence": (
                intent_confidence
            ),

            # Action prediction
            "predicted_action": (
                predicted_action
            ),

            "action_correct": (
                predicted_action
                == gold_action
            ),

            # Reply
            "reply": result.get(
                "reply",
                "",
            ),

            "grounded": grounded,

            "reply_reason": result.get(
                "reply_reason"
            ),

            # Retrieval
            "top_retrieval_score": (
                result.get(
                    "top_retrieval_score",
                    0.0,
                )
            ),

            "second_retrieval_score": (
                result.get(
                    "second_retrieval_score",
                    0.0,
                )
            ),

            "retrieval_score_gap": (
                result.get(
                    "retrieval_score_gap",
                    0.0,
                )
            ),

            "retrieval_count": len(
                retrieval
            ),

            # Escalation
            "predicted_escalation_reason": (
                result.get(
                    "escalation_reason"
                )
            ),

            "reason_code": result.get(
                "reason_code"
            ),

            # Evidence
            "top_evidence": (
                top_evidence
            ),

            # Full retrieval snapshot. This is deliberately retained so
            # LLM-judge caching and later audits can distinguish a change in
            # retrieval from a coincidentally identical generated reply.
            "retrieved_evidence": retrieval,
        }

        predictions.append(
            example_result
        )

        if index % 25 == 0:

            print(
                f"Evaluated "
                f"{index}/"
                f"{len(records)}"
            )

    # =================================================================
    # Intent metrics
    # =================================================================

    intent_accuracy = accuracy_score(
        intent_gold,
        intent_predictions,
    )

    intent_macro_f1 = f1_score(
        intent_gold,
        intent_predictions,
        labels=VALID_INTENTS,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        intent_gold,
        intent_predictions,
        labels=VALID_INTENTS,
        average="weighted",
        zero_division=0,
    )

    intent_report = classification_report(
        intent_gold,
        intent_predictions,
        labels=VALID_INTENTS,
        output_dict=True,
        zero_division=0,
    )

    # =================================================================
    # Action metrics
    # =================================================================

    action_accuracy = accuracy_score(
        action_gold,
        action_predictions,
    )

    action_report = classification_report(
        action_gold,
        action_predictions,
        labels=VALID_ACTIONS,
        output_dict=True,
        zero_division=0,
    )

    # =================================================================
    # Auto-handle precision
    # =================================================================

    predicted_auto = [
        prediction == "auto_handle"
        for prediction
        in action_predictions
    ]

    true_auto = [
        gold == "auto_handle"
        for gold
        in action_gold
    ]

    auto_handle_tp = sum(
        predicted and actual
        for predicted, actual
        in zip(
            predicted_auto,
            true_auto,
        )
    )

    auto_handle_fp = sum(
        predicted and not actual
        for predicted, actual
        in zip(
            predicted_auto,
            true_auto,
        )
    )

    if (
        auto_handle_tp
        + auto_handle_fp
    ) > 0:

        auto_handle_precision = (
            auto_handle_tp
            / (
                auto_handle_tp
                + auto_handle_fp
            )
        )

    else:

        auto_handle_precision = 0.0

    # =================================================================
    # Escalation recall
    # =================================================================

    predicted_escalate = [
        prediction == "escalate"
        for prediction
        in action_predictions
    ]

    true_escalate = [
        gold == "escalate"
        for gold
        in action_gold
    ]

    escalation_tp = sum(
        predicted and actual
        for predicted, actual
        in zip(
            predicted_escalate,
            true_escalate,
        )
    )

    escalation_fn = sum(
        (not predicted) and actual
        for predicted, actual
        in zip(
            predicted_escalate,
            true_escalate,
        )
    )

    if (
        escalation_tp
        + escalation_fn
    ) > 0:

        escalation_recall = (
            escalation_tp
            / (
                escalation_tp
                + escalation_fn
            )
        )

    else:

        escalation_recall = 0.0

    # =================================================================
    # Grounding
    # =================================================================

    grounding_rate = (
        grounded_count
        / len(records)
    )

    retrieval_rate = (
        retrieval_available_count
        / len(records)
    )

    return {

        "name": (
            "end_to_end_agent"
        ),

        "num_examples": len(
            records
        ),

        "intent": {

            "accuracy": round(
                float(
                    intent_accuracy
                ),
                4,
            ),

            "macro_f1": round(
                float(
                    intent_macro_f1
                ),
                4,
            ),

            "weighted_f1": round(
                float(
                    intent_weighted_f1
                ),
                4,
            ),

            "classification_report": (
                intent_report
            ),
        },

        "action": {

            "accuracy": round(
                float(
                    action_accuracy
                ),
                4,
            ),

            "classification_report": (
                action_report
            ),

            "auto_handle_precision": (
                round(
                    float(
                        auto_handle_precision
                    ),
                    4,
                )
            ),

            "escalation_recall": (
                round(
                    float(
                        escalation_recall
                    ),
                    4,
                )
            ),
        },

        "grounding": {

            "grounded_reply_rate": (
                round(
                    float(
                        grounding_rate
                    ),
                    4,
                )
            ),

            "retrieval_available_rate": (
                round(
                    float(
                        retrieval_rate
                    ),
                    4,
                )
            ),
        },

        "predictions": predictions,
    }



# GROQ JUDGE HELPERS


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

    return [
        item
        for item in evidence
        if isinstance(item, dict)
    ]


def build_llm_judge_cache_payload(
    example: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full, agent-derived input contract for a judgement cache."""

    return {
        "cache_schema_version": LLM_JUDGE_CACHE_SCHEMA_VERSION,
        "judge_model": os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
        "example_id": str(example.get("example_id", "")),
        "customer_message": example.get("customer_message", ""),
        "predicted_intent": example.get("predicted_intent", ""),
        "intent_confidence": example.get("intent_confidence"),
        # This contains each historical customer case, AmazonHelp response,
        # rank, score, conversation ID, and any retrieval metadata emitted by
        # the current agent. It must not be reduced to only the final reply.
        "retrieved_evidence": get_retrieved_evidence(example),
        "top_retrieval_score": example.get("top_retrieval_score", 0.0),
        "second_retrieval_score": example.get(
            "second_retrieval_score",
            0.0,
        ),
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


# LOAD CACHED LLM JUDGEMENTS


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



# SELECT LLM JUDGE EXAMPLES


def select_llm_judge_examples(
    predictions,
    n,
):
    """
    Prioritize difficult examples.

    This is not a gold-labeling step. The selection only
    determines which generated replies are inspected by
    the independent LLM judge.
    """

    def priority(
        example
    ):

        score = 0

        # Intent failure
        if not example[
            "intent_correct"
        ]:

            score += 3

        # Action failure
        if not example[
            "action_correct"
        ]:

            score += 3

        # Weak retrieval
        if (
            example[
                "top_retrieval_score"
            ] < 0.45
        ):

            score += 2

        # Ambiguous retrieval
        if (
            example[
                "top_retrieval_score"
            ] > 0
            and example[
                "retrieval_score_gap"
            ] < 0.05
        ):

            score += 2

        return (
            score,
            -float(
                example.get(
                    "top_retrieval_score",
                    0.0,
                )
            ),
        )

    ordered = sorted(
        predictions,
        key=priority,
        reverse=True,
    )

    return ordered[
        :min(
            n,
            len(predictions),
        )
    ]



# AGGREGATE LLM JUDGE METRICS


def aggregate_llm_judge_metrics(
    judged_records,
):

    metric_names = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall",
    ]

    values = {
        name: []
        for name in metric_names
    }

    for record in judged_records:

        judge = record.get(
            "judge",
            {},
        )

        if not isinstance(
            judge,
            dict,
        ):

            continue

        for name in metric_names:

            value = judge.get(
                name
            )

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if 1 <= value <= 5:

                values[
                    name
                ].append(
                    value
                )

    result = {
        "num_judged": len(
            judged_records
        ),
    }

    for name in metric_names:

        vals = values[
            name
        ]

        if vals:

            result[
                f"mean_{name}"
            ] = round(
                sum(vals)
                / len(vals),
                4,
            )

        else:

            result[
                f"mean_{name}"
            ] = None


    # Threshold metrics


    overall = values[
        "overall"
    ]

    groundedness = values[
        "groundedness"
    ]

    helpfulness = values[
        "helpfulness"
    ]

    result[
        "overall_4_plus_rate"
    ] = (
        round(
            sum(
                value >= 4
                for value
                in overall
            )
            / len(overall),
            4,
        )
        if overall
        else None
    )

    result[
        "groundedness_4_plus_rate"
    ] = (
        round(
            sum(
                value >= 4
                for value
                in groundedness
            )
            / len(groundedness),
            4,
        )
        if groundedness
        else None
    )

    result[
        "helpfulness_4_plus_rate"
    ] = (
        round(
            sum(
                value >= 4
                for value
                in helpfulness
            )
            / len(helpfulness),
            4,
        )
        if helpfulness
        else None
    )

    return result



# RUN GROQ LLM-AS-JUDGE


def run_llm_judge(
    predictions,
    _golden_records,
):

    if not LLM_JUDGE_ENABLED:
        result = {

            "enabled": False,

            "available": False,

            "status": "not_refreshed",

            "message": (
                "LLM judge disabled "
                "using "
                "LLM_JUDGE_ENABLED=0. Existing cached judgements "
                "are not evaluation results for this run."
            ),

            "results_file": str(LLM_JUDGE_RESULTS_FILE),
        }

        # Do not leave a previous run's numeric LLM scores looking like the
        # outcome of this one when a caller intentionally disables judging.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(LLM_JUDGE_METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    selected = (
        select_llm_judge_examples(
            predictions,
            LLM_JUDGE_N,
        )
    )

    # Golden labels are deliberately not passed to the LLM judge or included
    # in cache keys. The judge evaluates only the current agent output.
    (
        cache_records,
        cache_by_key,
        records_by_example,
    ) = load_cached_judgements()

    persisted_cache_records = deduplicate_cache_records(
        cache_records
    )

    print("\n")
    print("=" * 70)
    print(
        "GROQ LLM-AS-JUDGE"
    )
    print("=" * 70)

    print(
        f"Requested examples : "
        f"{LLM_JUDGE_N}"
    )

    print(
        f"Selected examples  : "
        f"{len(selected)}"
    )

    valid_cached_judgements = 0
    stale_cache_entries = 0
    new_api_calls = 0
    errors = 0
    final_judgements = []

    for index, example in enumerate(
        selected,
        start=1,
    ):

        example_id = str(
            example[
                "example_id"
            ]
        )

        cache_key = build_llm_judge_cache_key(example)
        cached_judgement = lookup_cached_judgement(
            cache_by_key,
            cache_key,
        )

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

        # ---------------------------------------------------------
        # Call Groq
        # ---------------------------------------------------------

        try:

            judge_result = (
                _call_groq_judge(
                    example
                )
            )

            record = {

                "example_id": (
                    example_id
                ),

                "customer_message": (
                    example[
                        "customer_message"
                    ]
                ),

                "cache_key": cache_key,

                "cache_input": build_llm_judge_cache_payload(
                    example
                ),

                "intent_confidence": example.get(
                    "intent_confidence"
                ),

                "reply": (
                    example.get(
                        "reply",
                        "",
                    )
                ),

                "gold_intent": (
                    example[
                        "gold_intent"
                    ]
                ),

                "predicted_intent": (
                    example[
                        "predicted_intent"
                    ]
                ),

                "gold_action": (
                    example[
                        "gold_action"
                    ]
                ),

                "predicted_action": (
                    example[
                        "predicted_action"
                    ]
                ),

                "action": example[
                    "predicted_action"
                ],

                "top_retrieval_score": (
                    example.get(
                        "top_retrieval_score",
                        0.0,
                    )
                ),

                "second_retrieval_score": (
                    example.get(
                        "second_retrieval_score",
                        0.0,
                    )
                ),

                "retrieval_score_gap": (
                    example.get(
                        "retrieval_score_gap",
                        0.0,
                    )
                ),

                "retrieved_evidence": get_retrieved_evidence(
                    example
                ),

                "judge": (
                    judge_result
                ),

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

            # -----------------------------------------------------
            # Save immediately
            # -----------------------------------------------------

            save_judge_results(
                persisted_cache_records
            )

            print(
                f"Judged "
                f"{index}/"
                f"{len(selected)} "
                f"- {example_id}"
            )

            if (
                LLM_JUDGE_DELAY
                > 0
            ):

                time.sleep(
                    LLM_JUDGE_DELAY
                )

        except Exception as exc:

            errors += 1

            print(
                f"Judge error on "
                f"{example_id}: "
                f"{exc}"
            )

    metrics = (
        aggregate_llm_judge_metrics(
            final_judgements
        )
    )

    metrics.update({

        "enabled": True,

        "available": bool(
            final_judgements
        ),

        "requested_examples": (
            LLM_JUDGE_N
        ),

        "selected_examples": (
            len(selected)
        ),

        "judged_examples": (
            len(final_judgements)
        ),

        "final_judgements": len(final_judgements),

        "valid_cached_judgements": valid_cached_judgements,

        "stale_cache_entries": stale_cache_entries,

        "new_api_calls": (
            new_api_calls
        ),

        "errors": errors,

        "model": os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),

        "provider": "groq",

        "results_file": str(
            LLM_JUDGE_RESULTS_FILE
        ),
    })

    print(
        f"Valid cached judgements  : "
        f"{valid_cached_judgements}"
    )

    print(
        f"Stale cache entries      : "
        f"{stale_cache_entries}"
    )

    print(
        f"New API calls            : "
        f"{new_api_calls}"
    )

    print(
        f"Final judgements         : "
        f"{len(final_judgements)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LLM_JUDGE_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return metrics



# SAVE JUDGE RESULTS


def save_judge_results(
    records,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LLM_JUDGE_RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )



# FAILURE ANALYSIS


def identify_failures(
    agent_result: Dict[str, Any]
) -> List[Dict[str, Any]]:

    failures = []

    for example in agent_result[
        "predictions"
    ]:


        # Intent error


        if not example[
            "intent_correct"
        ]:

            failures.append({

                "example_id": (
                    example[
                        "example_id"
                    ]
                ),

                "failure_type": (
                    "intent_misclassification"
                ),

                "customer_message": (
                    example[
                        "customer_message"
                    ]
                ),

                "gold_intent": (
                    example[
                        "gold_intent"
                    ]
                ),

                "predicted_intent": (
                    example[
                        "predicted_intent"
                    ]
                ),

                "intent_confidence": (
                    example[
                        "intent_confidence"
                    ]
                ),
            })


        # Action error


        if not example[
            "action_correct"
        ]:

            failures.append({

                "example_id": (
                    example[
                        "example_id"
                    ]
                ),

                "failure_type": (
                    "action_misclassification"
                ),

                "customer_message": (
                    example[
                        "customer_message"
                    ]
                ),

                "gold_action": (
                    example[
                        "gold_action"
                    ]
                ),

                "predicted_action": (
                    example[
                        "predicted_action"
                    ]
                ),

                "gold_escalation_reason": (
                    example[
                        "gold_escalation_reason"
                    ]
                ),

                "predicted_escalation_reason": (
                    example[
                        "predicted_escalation_reason"
                    ]
                ),
            })


        # Ungrounded auto-handle


        if (
            example[
                "gold_action"
            ]
            == "auto_handle"
            and not example[
                "grounded"
            ]
        ):

            failures.append({

                "example_id": (
                    example[
                        "example_id"
                    ]
                ),

                "failure_type": (
                    "ungrounded_auto_handle"
                ),

                "customer_message": (
                    example[
                        "customer_message"
                    ]
                ),

                "gold_intent": (
                    example[
                        "gold_intent"
                    ]
                ),

                "predicted_intent": (
                    example[
                        "predicted_intent"
                    ]
                ),

                "predicted_action": (
                    example[
                        "predicted_action"
                    ]
                ),

                "top_retrieval_score": (
                    example[
                        "top_retrieval_score"
                    ]
                ),

                "reply": (
                    example[
                        "reply"
                    ]
                ),
            })


        # Retrieval ambiguity


        if (
            example[
                "top_retrieval_score"
            ] > 0
            and example[
                "retrieval_score_gap"
            ] < 0.05
        ):

            failures.append({

                "example_id": (
                    example[
                        "example_id"
                    ]
                ),

                "failure_type": (
                    "ambiguous_retrieval"
                ),

                "customer_message": (
                    example[
                        "customer_message"
                    ]
                ),

                "top_retrieval_score": (
                    example[
                        "top_retrieval_score"
                    ]
                ),

                "second_retrieval_score": (
                    example[
                        "second_retrieval_score"
                    ]
                ),

                "retrieval_score_gap": (
                    example[
                        "retrieval_score_gap"
                    ]
                ),

                "reply": (
                    example[
                        "reply"
                    ]
                ),
            })

    return failures

# SAVE AGGREGATE METRICS


def save_metrics(
    results: Dict[str, Any]
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_results = {}

    for name, result in (
        results.items()
    ):

        if isinstance(
            result,
            dict,
        ):

            clean_results[
                name
            ] = {

                key: value

                for key, value
                in result.items()

                if key
                != "predictions"
            }

        else:

            clean_results[
                name
            ] = result

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean_results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nMetrics saved to:"
        f"\n{METRICS_FILE}"
    )



# PRINT BASELINE RESULTS


def print_baseline_results(
    results: Dict[str, Any]
):

    print("\n")
    print("=" * 70)
    print(
        "BASELINE EVALUATION"
    )
    print("=" * 70)

    for name in [
        "trivial",
        "tfidf",
    ]:

        if name not in results:

            continue

        result = results[
            name
        ]

        print(
            "\n"
            + "-" * 70
        )

        print(
            result["name"]
        )

        if (
            "majority_intent"
            in result
        ):

            print(
                f"Majority intent : "
                f"{result['majority_intent']}"
            )

        print(
            f"Accuracy        : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Macro F1        : "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1     : "
            f"{result['weighted_f1']:.4f}"
        )



# PRINT AGENT RESULTS


def print_agent_results(
    result: Dict[str, Any]
):

    print("\n")
    print("=" * 70)
    print(
        "END-TO-END AGENT EVALUATION"
    )
    print("=" * 70)

    print("\nINTENT")

    print(
        f"Accuracy        : "
        f"{result['intent']['accuracy']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{result['intent']['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1     : "
        f"{result['intent']['weighted_f1']:.4f}"
    )

    print("\nACTION")

    print(
        f"Accuracy        : "
        f"{result['action']['accuracy']:.4f}"
    )

    print(
        f"Auto-handle precision : "
        f"{result['action']['auto_handle_precision']:.4f}"
    )

    print(
        f"Escalation recall     : "
        f"{result['action']['escalation_recall']:.4f}"
    )

    print("\nRETRIEVAL / REPLY")

    print(
        f"Grounded reply rate   : "
        f"{result['grounding']['grounded_reply_rate']:.4f}"
    )

    print(
        f"Retrieval available   : "
        f"{result['grounding']['retrieval_available_rate']:.4f}"
    )



# PRINT LLM JUDGE RESULTS


def print_llm_judge_results(
    result: Dict[str, Any]
):

    print("\n")
    print("=" * 70)
    print(
        "GROQ LLM-AS-JUDGE RESULTS"
    )
    print("=" * 70)

    if not result.get(
        "available",
        False,
    ):

        print(
            result.get(
                "message",
                "LLM judge unavailable.",
            )
        )

        return

    print(
        f"Model                 : "
        f"{result.get('model')}"
    )

    print(
        f"Provider              : "
        f"{result.get('provider')}"
    )

    print(
        f"Judged examples       : "
        f"{result.get('judged_examples')}"
    )

    print(
        f"Requested examples    : "
        f"{result.get('requested_examples')}"
    )

    print(
        f"Selected examples     : "
        f"{result.get('selected_examples')}"
    )

    print(
        f"Valid cached judgements: "
        f"{result.get('valid_cached_judgements')}"
    )

    print(
        f"Stale cache entries   : "
        f"{result.get('stale_cache_entries')}"
    )

    print(
        f"New API calls         : "
        f"{result.get('new_api_calls')}"
    )

    print(
        f"Final judgements      : "
        f"{result.get('final_judgements')}"
    )

    print(
        f"Mean relevance        : "
        f"{result.get('mean_relevance')}"
    )

    print(
        f"Mean groundedness     : "
        f"{result.get('mean_groundedness')}"
    )

    print(
        f"Mean helpfulness      : "
        f"{result.get('mean_helpfulness')}"
    )

    print(
        f"Mean safety           : "
        f"{result.get('mean_safety')}"
    )

    print(
        f"Mean overall          : "
        f"{result.get('mean_overall')}"
    )

    print(
        f"Overall >= 4 rate     : "
        f"{result.get('overall_4_plus_rate')}"
    )

    print(
        f"Groundedness >= 4     : "
        f"{result.get('groundedness_4_plus_rate')}"
    )

    print(
        f"Helpfulness >= 4      : "
        f"{result.get('helpfulness_4_plus_rate')}"
    )



# PRINT FAILURE SUMMARY


def print_failure_summary(
    failures: List[Dict[str, Any]]
):

    print("\n")
    print("=" * 70)
    print(
        "FAILURE ANALYSIS"
    )
    print("=" * 70)

    counts = {}

    for failure in failures:

        failure_type = (
            failure[
                "failure_type"
            ]
        )

        counts[
            failure_type
        ] = (
            counts.get(
                failure_type,
                0,
            )
            + 1
        )

    for (
        failure_type,
        count,
    ) in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        print(
            f"{failure_type:<35} "
            f"{count}"
        )

    print(
        f"\nTotal failure records: "
        f"{len(failures)}"
    )



# MAIN


def main():

    print("=" * 70)
    print(
        "HIVER CUSTOMER SUPPORT AGENT - "
        "FULL EVALUATION"
    )
    print("=" * 70)


    # Load golden set


    records = load_golden_set()


    # Validate


    validate_golden_set(
        records
    )

    df = pd.DataFrame(
        records
    )

    print(
        f"\nEvaluating "
        f"{len(df)} "
        f"golden examples..."
    )


    # Baseline 1


    print(
        "\nRunning trivial "
        "majority baseline..."
    )

    trivial_result = (
        run_trivial_baseline(
            df
        )
    )


    # Baseline 2


    print(
        "\nRunning TF-IDF + "
        "Logistic Regression "
        "baseline..."
    )

    tfidf_result = (
        run_tfidf_baseline(
            df
        )
    )


    # Actual Agent


    print(
        "\nRunning end-to-end "
        "AmazonHelp agent..."
    )

    agent_result = (
        run_agent_evaluation(
            records
        )
    )


    # Human agreement


    print(
        "\nChecking human "
        "annotation agreement..."
    )

    human_agreement = (
        calculate_independent_human_agreement()
    )


    # Failure analysis


    failures = (
        identify_failures(
            agent_result
        )
    )


    # Save per-example agent results


    save_jsonl(
        EXAMPLES_FILE,
        agent_result[
            "predictions"
        ],
    )


    # Save failures


    save_jsonl(
        FAILURES_FILE,
        failures,
    )


    # Groq LLM-as-judge


    print(
        "\nRunning Groq "
        "LLM-as-judge..."
    )

    llm_judge_result = (
        run_llm_judge(
            agent_result[
                "predictions"
            ],
            records,
        )
    )


    # Aggregate all results


    results = {

        "evaluation_metadata": {

            "golden_examples": (
                len(records)
            ),

            "golden_file": (
                str(
                    GOLDEN_FILE
                )
            ),

            "note": (
                "The golden set is used "
                "only for evaluation. "
                "The production intent "
                "classifier is trained "
                "outside the golden set."
            ),
        },

        "trivial": (
            trivial_result
        ),

        "tfidf": (
            tfidf_result
        ),

        "end_to_end": (
            agent_result
        ),

        "llm_judge": (
            llm_judge_result
        ),

        "human_agreement": (
            human_agreement
        ),

        "failure_analysis": {

            "total_failure_records": (
                len(failures)
            ),

            "file": (
                str(
                    FAILURES_FILE
                )
            ),
        },
    }


    # Print results


    print_baseline_results(
        results
    )

    print_agent_results(
        agent_result
    )

    print_llm_judge_results(
        llm_judge_result
    )

    print_failure_summary(
        failures
    )


    # Human agreement output


    print("\n")
    print("=" * 70)
    print(
        "HUMAN AGREEMENT"
    )
    print("=" * 70)

    if human_agreement.get(
        "available",
        False,
    ):
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


    save_metrics(
        results
    )


    # Output paths


    print(
        f"\nPer-example results saved to:"
        f"\n{EXAMPLES_FILE}"
    )

    print(
        f"\nFailure analysis saved to:"
        f"\n{FAILURES_FILE}"
    )

    if llm_judge_result.get("enabled", False):
        print(
            f"\nLLM judge results saved to:"
            f"\n{LLM_JUDGE_RESULTS_FILE}"
        )

        print(
            f"\nLLM judge metrics saved to:"
            f"\n{LLM_JUDGE_METRICS_FILE}"
        )
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
    print(
        "FULL EVALUATION COMPLETE"
    )
    print("=" * 70)



# ENTRY POINT


if __name__ == "__main__":
    main()
