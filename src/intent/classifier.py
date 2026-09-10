import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.intent.schemas import has_support_complaint_signal



# CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

THREADS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "amazon_help_threads.jsonl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_PATH = MODEL_DIR / "intent_classifier.pkl"



# INTENTS


INTENTS = [
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



# TEXT CLEANING


def clean_text(text: str) -> str:
    """
    Normalize customer text before classification.
    """

    if not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        " ",
        text,
    )

    # Remove HTML entities
    text = re.sub(
        r"&\w+;",
        " ",
        text,
    )

    # Replace numbers
    text = re.sub(
        r"\b\d+\b",
        " NUM ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()



# KEYWORD TAXONOMY


KEYWORDS = {

    "delivery_issue": [
        "delivered",
        "not delivered",
        "delivery",
        "delivery issue",
        "package hasn't arrived",
        "package hasnt arrived",
        "package not arrived",
        "parcel hasn't arrived",
        "parcel hasnt arrived",
        "haven't received",
        "havent received",
        "didn't receive",
        "didnt receive",
        "not received",
        "where is my package",
        "where's my package",
        "missing package",
    ],

    "shipping_issue": [
        "shipping",
        "shipment",
        "tracking",
        "tracking number",
        "track my order",
        "carrier",
        "shipping delay",
        "shipped",
        "dispatch",
        "in transit",
    ],

    "refund_issue": [
        "refund",
        "refunded",
        "money back",
        "give my money back",
        "want my money back",
        "return my money",
        "reimbursement",
        "reimburse",
    ],

    "payment_issue": [
        "payment",
        "payment failed",
        "payment failure",
        "charged",
        "charge",
        "charged twice",
        "charged me",
        "billing",
        "credit card",
        "debit card",
        "card charged",
        "transaction",
        "declined payment",
    ],

    "account_issue": [
        "account",
        "account locked",
        "locked account",
        "login",
        "log in",
        "sign in",
        "signin",
        "password",
        "cannot login",
        "can't login",
        "cant login",
        "unable to login",
        "access my account",
    ],

    "prime_video_issue": [
        "prime video",
        "primevideo",
        "prime video not working",
        "video not working",
        "video won't play",
        "video wont play",
        "can't watch prime",
        "cant watch prime",
        "prime movie",
        "prime tv",
    ],

    "device_issue": [
        "fire tv",
        "firetv",
        "fire stick",
        "firestick",
        "kindle",
        "echo",
        "alexa",
        "device",
        "tablet",
        "tv not working",
        "device not working",
    ],

    "support_complaint": [
        "terrible support",
        "bad support",
        "worst support",
        "awful support",
        "horrible support",
        "terrible service",
        "bad service",
        "worst service",
        "unacceptable",
        "disappointed",
        "very disappointed",
        "this is terrible",
        "this is awful",
        "this is horrible",
        "no one helps",
        "nobody helps",
    ],

    "order_issue": [
        "order",
        "my order",
        "order issue",
        "order problem",
        "wrong order",
        "wrong item",
        "missing item",
        "cancel order",
        "cancel my order",
        "order cancelled",
        "order canceled",
        "order status",
    ],

    "product_or_content_issue": [
        "product",
        "item",
        "wrong product",
        "defective",
        "broken item",
        "damaged item",
        "product damaged",
        "book",
        "movie",
        "show",
        "content",
    ],
}



# HEURISTIC LABELING


def keyword_label(
    text: str,
) -> str:
    """
    Assign a weak training label using keyword rules.

    IMPORTANT:
    These labels are used only to create a training corpus.
    The manually labelled golden set is NOT used here.
    """

    text = clean_text(text)

    if not text:
        return "other"


    # Specific intents first


    # Prime Video
    for keyword in KEYWORDS["prime_video_issue"]:
        if keyword in text:
            return "prime_video_issue"

    # Device
    for keyword in KEYWORDS["device_issue"]:
        if keyword in text:
            return "device_issue"

    # Refund
    for keyword in KEYWORDS["refund_issue"]:
        if keyword in text:
            return "refund_issue"

    # Payment
    for keyword in KEYWORDS["payment_issue"]:
        if keyword in text:
            return "payment_issue"

    # Account
    for keyword in KEYWORDS["account_issue"]:
        if keyword in text:
            return "account_issue"

    # Explicit dissatisfaction with service/support must outrank operational
    # vocabulary such as "order" or "delivery". This avoids weakly labelling
    # complaint-heavy historical cases as routine delivery issues.
    if has_support_complaint_signal(text):
        return "support_complaint"

    # Delivery
    for keyword in KEYWORDS["delivery_issue"]:
        if keyword in text:
            return "delivery_issue"

    # Shipping
    for keyword in KEYWORDS["shipping_issue"]:
        if keyword in text:
            return "shipping_issue"

    # Product/content
    for keyword in KEYWORDS["product_or_content_issue"]:
        if keyword in text:
            return "product_or_content_issue"

    # Order last because "order" is extremely common
    for keyword in KEYWORDS["order_issue"]:
        if keyword in text:
            return "order_issue"

    return "other"



# LOAD TRAINING DATA


def load_training_data(
    threads_path=THREADS_PATH,
) -> Tuple[List[str], List[str]]:
    """
    Extract first customer message from each historical thread.

    Weak labels are generated using the intent keyword taxonomy.
    """

    texts = []
    labels = []

    threads_path = Path(
        threads_path
    )

    if not threads_path.exists():
        raise FileNotFoundError(
            f"Threads file not found: {threads_path}"
        )

    with open(
        threads_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                thread = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = thread.get(
                "messages",
                [],
            )

            if not messages:
                continue

            customer_message = None

            for message in messages:

                # Actual dataset schema uses is_amazon
                is_amazon = message.get(
                    "is_amazon",
                    False,
                )

                if not is_amazon:

                    text = message.get(
                        "text",
                        "",
                    )

                    if text and text.strip():
                        customer_message = text
                        break

            if not customer_message:
                continue

            cleaned = clean_text(
                customer_message
            )

            if not cleaned:
                continue

            label = keyword_label(
                cleaned
            )

            texts.append(cleaned)
            labels.append(label)

    return texts, labels



# TRAIN CLASSIFIER


def train_classifier():
    """
    Train TF-IDF + Logistic Regression classifier.
    """

    print("=" * 70)
    print("AMAZONHELP INTENT CLASSIFIER")
    print("=" * 70)

    print("\nLoading historical training data...")

    texts, labels = load_training_data()

    print(
        f"Training examples : {len(texts):,}"
    )

    if not texts:
        raise RuntimeError(
            "No training examples were created."
        )


    # Show label distribution


    print("\nLabel distribution:")

    counts = {}

    for label in labels:
        counts[label] = (
            counts.get(label, 0) + 1
        )

    for label in INTENTS:
        print(
            f"  {label:<30} "
            f"{counts.get(label, 0):>7,}"
        )


    # TF-IDF


    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        max_features=30000,
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X = vectorizer.fit_transform(
        texts
    )

    print(
        f"Matrix shape : {X.shape}"
    )

    print(
        f"Vocabulary   : "
        f"{len(vectorizer.vocabulary_):,}"
    )


    # Logistic Regression


    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    classifier.fit(
        X,
        labels,
    )


    # Save model


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_data = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "intents": INTENTS,
        "config": {
            "max_features": 30000,
            "ngram_range": (1, 2),
            "min_df": 3,
            "max_df": 0.95,
            "random_state": 42,
        },
    }

    joblib.dump(
        model_data,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model_data



# CLASSIFIER CLASS


class IntentClassifier:
    """
    Load and use the trained AmazonHelp intent classifier.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
    ):

        model_path = Path(
            model_path
        )

        if not model_path.exists():

            print(
                "Intent classifier model not found."
            )

            print(
                "Training model now..."
            )

            train_classifier()

        model_data = joblib.load(
            model_path
        )

        self.vectorizer = (
            model_data["vectorizer"]
        )

        self.classifier = (
            model_data["classifier"]
        )

        self.intents = (
            model_data["intents"]
        )



    # PREDICT


    def predict(
        self,
        text: str,
    ) -> str:
        """
        Predict one intent.
        """

        cleaned = clean_text(
            text
        )

        if not cleaned:
            return "other"

        if has_support_complaint_signal(cleaned):
            return "support_complaint"

        X = self.vectorizer.transform(
            [cleaned]
        )

        prediction = self.classifier.predict(
            X
        )[0]

        return str(
            prediction
        )



    # PREDICT WITH CONFIDENCE


    def predict_with_confidence(
        self,
        text: str,
    ) -> Dict[str, object]:
        """
        Return predicted intent, confidence and probabilities.
        """

        cleaned = clean_text(
            text
        )

        if not cleaned:
            return {
                "intent": "other",
                "confidence": 0.0,
                "probabilities": {},
            }

        X = self.vectorizer.transform(
            [cleaned]
        )

        probabilities = (
            self.classifier.predict_proba(X)[0]
        )

        classes = (
            self.classifier.classes_
        )

        best_index = int(
            np.argmax(probabilities)
        )

        prediction = str(
            classes[best_index]
        )

        confidence = float(
            probabilities[best_index]
        )

        probability_dict = {
            str(label): float(prob)
            for label, prob in zip(
                classes,
                probabilities,
            )
        }

        if has_support_complaint_signal(cleaned):
            # The explicit service/support complaint rule is intentionally a
            # higher-priority decision than an operational ML prediction. The
            # model probability is still returned unchanged as a diagnostic,
            # rather than pretending it is a calibrated rule confidence.
            return {
                "intent": "support_complaint",
                "confidence": probability_dict.get(
                    "support_complaint",
                    0.0,
                ),
                "probabilities": probability_dict,
                "decision_source": "support_complaint_rule",
            }

        return {
            "intent": prediction,
            "confidence": confidence,
            "probabilities": probability_dict,
            "decision_source": "model",
        }



# SIMPLE API


def predict_intent(
    text: str,
) -> str:
    """
    Convenience function.
    """

    classifier = IntentClassifier()

    return classifier.predict(
        text
    )



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP INTENT CLASSIFIER TEST")
    print("=" * 70)


    # Train model if required


    if not MODEL_PATH.exists():

        print(
            "\nNo trained model found."
        )

        print(
            "Training model..."
        )

        train_classifier()


    # Load classifier


    print(
        "\nLoading classifier..."
    )

    classifier = IntentClassifier()

    print(
        "Classifier loaded."
    )


    # Test messages


    test_messages = [
        "My package says delivered but I haven't received it",

        "Why is Prime Video not working?",

        "I want a refund for my order",

        "My account is locked",

        "My payment was declined",

        "Where is my shipment?",

        "My Fire TV is not working",

        "This is terrible support!",

        "Something is wrong with my order",

        "Can you help me?",
    ]


    # Predictions


    for message in test_messages:

        result = (
            classifier.predict_with_confidence(
                message
            )
        )

        print("\n" + "-" * 70)

        print(
            f"Message    : {message}"
        )

        print(
            f"Intent     : "
            f"{result['intent']}"
        )

        print(
            f"Confidence : "
            f"{result['confidence']:.4f}"
        )

    print("\n" + "=" * 70)
    print("CLASSIFIER TEST COMPLETE")
    print("=" * 70)



# ENTRY POINT


if __name__ == "__main__":
    main()
