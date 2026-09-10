"""Baseline classifiers used only for evaluation of the golden set."""

from typing import Any, Dict, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def run_trivial_baseline(
    examples: pd.DataFrame,
    valid_intents: Sequence[str],
) -> Dict[str, Any]:
    """Evaluate the majority-intent baseline without changing its definition."""

    majority_intent = examples["gold_intent"].value_counts().idxmax()
    predictions = [majority_intent] * len(examples)

    return {
        "name": "trivial_majority_baseline",
        "majority_intent": majority_intent,
        "accuracy": round(
            float(accuracy_score(examples["gold_intent"], predictions)), 4
        ),
        "macro_f1": round(
            float(
                f1_score(
                    examples["gold_intent"],
                    predictions,
                    labels=valid_intents,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    examples["gold_intent"],
                    predictions,
                    labels=valid_intents,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "predictions": predictions,
    }


def run_tfidf_baseline(
    examples: pd.DataFrame,
    valid_intents: Sequence[str],
) -> Dict[str, Any]:
    """Run the existing 5-fold TF-IDF plus Logistic Regression baseline."""

    customer_messages = examples["customer_message"].astype(str)
    intents = examples["gold_intent"].astype(str)
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
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    predictions = cross_val_predict(
        model,
        customer_messages,
        intents,
        cv=cross_validation,
        method="predict",
    )

    return {
        "name": "tfidf_logistic_regression",
        "accuracy": round(float(accuracy_score(intents, predictions)), 4),
        "macro_f1": round(
            float(
                f1_score(
                    intents,
                    predictions,
                    labels=valid_intents,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(
                    intents,
                    predictions,
                    labels=valid_intents,
                    average="weighted",
                    zero_division=0,
                )
            ),
            4,
        ),
        "classification_report": classification_report(
            intents,
            predictions,
            labels=valid_intents,
            output_dict=True,
            zero_division=0,
        ),
        "predictions": predictions.tolist(),
    }
