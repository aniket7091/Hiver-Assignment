import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable



# PROJECT PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



# IMPORTS


from src.retrieval.retrieve import AmazonHelpRetriever
from src.agent.generate_reply import generate_reply
from src.agent.escalation import decide_escalation
from src.intent.classifier import IntentClassifier



# SIMPLE INTENT FALLBACK


def simple_intent_classifier(
    customer_message: str,
) -> str:
    """
    Lightweight fallback classifier.

    This is kept only for backward compatibility.
    The default production pipeline now uses the trained
    IntentClassifier.
    """

    text = customer_message.lower()


    # Delivery


    delivery_keywords = [
        "delivered",
        "delivery",
        "package",
        "parcel",
        "not received",
        "haven't received",
        "didn't receive",
        "where is my package",
    ]

    if any(
        keyword in text
        for keyword in delivery_keywords
    ):
        return "delivery_issue"


    # Prime Video


    prime_video_keywords = [
        "prime video",
        "primevideo",
        "video not working",
        "video isn't working",
        "video won't play",
        "cannot watch",
    ]

    if any(
        keyword in text
        for keyword in prime_video_keywords
    ):
        return "prime_video_issue"


    # Refund


    refund_keywords = [
        "refund",
        "money back",
        "give my money back",
    ]

    if any(
        keyword in text
        for keyword in refund_keywords
    ):
        return "refund_issue"


    # Payment


    payment_keywords = [
        "payment",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "billing",
    ]

    if any(
        keyword in text
        for keyword in payment_keywords
    ):
        return "payment_issue"


    # Account


    account_keywords = [
        "account",
        "login",
        "log in",
        "password",
        "locked",
        "sign in",
    ]

    if any(
        keyword in text
        for keyword in account_keywords
    ):
        return "account_issue"


    # Shipping


    shipping_keywords = [
        "shipment",
        "shipping",
        "tracking",
        "tracking number",
        "carrier",
    ]

    if any(
        keyword in text
        for keyword in shipping_keywords
    ):
        return "shipping_issue"


    # Device


    device_keywords = [
        "fire tv",
        "firetv",
        "kindle",
        "echo",
        "alexa",
        "device",
    ]

    if any(
        keyword in text
        for keyword in device_keywords
    ):
        return "device_issue"


    # Product / Content


    product_keywords = [
        "product",
        "item",
        "book",
        "movie",
        "show",
        "content",
    ]

    if any(
        keyword in text
        for keyword in product_keywords
    ):
        return "product_or_content_issue"


    # Support Complaint


    complaint_keywords = [
        "terrible",
        "worst",
        "awful",
        "horrible",
        "bad support",
        "terrible support",
        "unacceptable",
        "disappointed",
    ]

    if any(
        keyword in text
        for keyword in complaint_keywords
    ):
        return "support_complaint"

    return "other"



# AMAZON HELP AGENT


class AmazonHelpAgent:
    """
    End-to-end AmazonHelp support agent.

    Pipeline:

        Customer message
              ↓
        Intent classification
              ↓
        Historical retrieval
              ↓
        Grounded reply generation
              ↓
        Escalation decision
              ↓
        Final structured result
    """

    def __init__(
        self,
        retriever: Optional[AmazonHelpRetriever] = None,
        intent_classifier: Optional[
            Callable[[str], str]
        ] = None,
        top_k: int = 5,
        min_evidence_score: float = 0.35,
    ):
        """
        Parameters
        ----------
        retriever:
            AmazonHelpRetriever instance.

        intent_classifier:
            Optional custom classifier.

            If None, the trained IntentClassifier is loaded.

        top_k:
            Number of historical examples to retrieve.

        min_evidence_score:
            Minimum similarity required for grounded reply.
        """


        # Retriever


        self.retriever = (
            retriever
            if retriever is not None
            else AmazonHelpRetriever()
        )


        # Intent Classifier


        if intent_classifier is not None:
            # Allows passing a custom callable classifier.
            self.intent_classifier = intent_classifier
        else:
            # Default production classifier.
            #
            # This loads the trained TF-IDF +
            # Logistic Regression model from classifier.py.
            self.intent_classifier = IntentClassifier()

        self.top_k = top_k

        self.min_evidence_score = (
            min_evidence_score
        )

    # PROCESS MESSAGE

    def process(
        self,
        customer_message: str,
    ) -> Dict[str, Any]:
        """
        Process one customer message end-to-end.
        """


        # Validate input


        if not customer_message:
            return {
                "customer_message": "",
                "intent": "other",
                "intent_confidence": 0.0,
                "retrieval": [],
                "reply": "",
                "grounded": False,
                "action": "escalate",
                "escalation_reason": (
                    "Customer message is empty."
                ),
                "reason_code": "insufficient_information",
                "top_retrieval_score": 0.0,
                "second_retrieval_score": 0.0,
                "retrieval_score_gap": 0.0,
                "evidence": [],
            }

        customer_message = str(
            customer_message
        ).strip()

        if not customer_message:
            return {
                "customer_message": "",
                "intent": "other",
                "intent_confidence": 0.0,
                "retrieval": [],
                "reply": "",
                "grounded": False,
                "action": "escalate",
                "escalation_reason": (
                    "Customer message is empty."
                ),
                "reason_code": "insufficient_information",
                "top_retrieval_score": 0.0,
                "second_retrieval_score": 0.0,
                "retrieval_score_gap": 0.0,
                "evidence": [],
            }


        # STEP 1: Intent classification


        if hasattr(
            self.intent_classifier,
            "predict_with_confidence",
        ):

            prediction = (
                self.intent_classifier
                .predict_with_confidence(
                    customer_message
                )
            )

            # Expected format:
            #
            # {
            #     "intent": "delivery_issue",
            #     "confidence": 0.9997
            # }

            intent = str(
                prediction.get(
                    "intent",
                    "other",
                )
                or "other"
            ).strip().lower()

            intent_confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

        else:

            # Backward-compatible support for
            # a normal callable classifier.

            intent = self.intent_classifier(
                customer_message
            )

            intent = str(
                intent or "other"
            ).strip().lower()

            intent_confidence = None


        # STEP 2: Historical retrieval


        retrieval_results = (
            self.retriever.retrieve(
                query=customer_message,
                top_k=self.top_k,
                predicted_intent=intent,
            )
        )


        # STEP 3: Generate grounded reply


        reply_result = generate_reply(
            customer_message=customer_message,
            retrieved_results=retrieval_results,
            intent=intent,
            min_evidence_score=(
                self.min_evidence_score
            ),
        )


        # STEP 4: Escalation decision


        candidate_intents = []
        for res in retrieval_results[:2]:
            candidate_intent = res.get("candidate_intent")

            if candidate_intent:
                candidate_intents.append(candidate_intent)
                continue

            # Backward-compatible fallback for retrieval results produced by
            # an older index/retriever without candidate diagnostics.
            if hasattr(self.intent_classifier, "predict"):
                candidate_message = res.get("customer_message", "")
                if candidate_message:
                    candidate_intents.append(
                        self.intent_classifier.predict(candidate_message)
                    )

        escalation_result = decide_escalation(
            customer_message=customer_message,
            intent=intent,
            retrieved_results=retrieval_results,
            intent_confidence=intent_confidence,
            candidate_intents=candidate_intents,
            has_grounded_reply=reply_result["grounded"],
        )


        # STEP 5: Final structured result


        result = {
            "customer_message": customer_message,

            # Intent
            "intent": intent,
            "intent_confidence": intent_confidence,

            # Retrieval
            "retrieval": retrieval_results,

            # Reply
            "reply": reply_result["reply"],

            "grounded": reply_result[
                "grounded"
            ],

            "reply_reason": reply_result[
                "reason"
            ],

            # Escalation
            "action": escalation_result[
                "action"
            ],

            "escalation_reason": (
                escalation_result["reason"]
                if escalation_result["action"]
                == "escalate"
                else None
            ),

            "reason_code": escalation_result[
                "reason_code"
            ],

            # Retrieval diagnostics
            "top_retrieval_score": (
                escalation_result[
                    "top_retrieval_score"
                ]
            ),

            "second_retrieval_score": (
                escalation_result[
                    "second_retrieval_score"
                ]
            ),

            "retrieval_score_gap": (
                escalation_result[
                    "retrieval_score_gap"
                ]
            ),

            # Evidence
            "evidence": reply_result[
                "evidence"
            ],
        }

        return result



# CONVENIENCE FUNCTION


def run_agent(
    customer_message: str,
    intent_classifier: Optional[
        Callable[[str], str]
    ] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Simple function for processing one message.
    """

    agent = AmazonHelpAgent(
        intent_classifier=intent_classifier,
        top_k=top_k,
    )

    return agent.process(
        customer_message
    )



# DISPLAY RESULT


def print_result(
    result: Dict[str, Any]
):
    """
    Pretty-print agent output.
    """

    print("\n" + "=" * 70)
    print("AMAZONHELP AGENT RESULT")
    print("=" * 70)


    # Customer


    print("\nCUSTOMER:")

    print(
        result["customer_message"]
    )


    # Intent


    print("\nINTENT:")

    confidence = result.get(
        "intent_confidence"
    )

    if confidence is not None:

        print(
            f"{result['intent']} "
            f"(confidence={confidence:.4f})"
        )

    else:

        print(
            result["intent"]
        )


    # Action


    print("\nACTION:")

    print(
        result["action"]
    )


    # Escalation reason


    if result["escalation_reason"]:

        print("\nESCALATION REASON:")

        print(
            result["escalation_reason"]
        )


    # Reply


    print("\nREPLY:")

    print(
        result["reply"]
    )


    # Grounded


    print("\nGROUNDED:")

    print(
        result["grounded"]
    )


    # Reply reason


    print("\nREPLY REASON:")

    print(
        result["reply_reason"]
    )


    # Retrieval


    print("\nRETRIEVAL:")

    print(
        f"Top score       : "
        f"{result['top_retrieval_score']:.4f}"
    )

    print(
        f"Second score    : "
        f"{result['second_retrieval_score']:.4f}"
    )

    print(
        f"Score gap       : "
        f"{result['retrieval_score_gap']:.4f}"
    )

    print(
        f"Retrieved cases : "
        f"{len(result['retrieval'])}"
    )


    # Evidence


    print("\nEVIDENCE:")

    if not result["evidence"]:

        print(
            "No historical evidence used."
        )

    else:

        for i, evidence in enumerate(
            result["evidence"],
            start=1,
        ):

            print(
                f"\n[{i}] "
                f"score="
                f"{evidence['score']:.4f}"
            )

            print(
                f"Conversation: "
                f"{evidence['conversation_id']}"
            )

            print(
                f"Historical response: "
                f"{evidence['response']}"
            )

    print("\n" + "=" * 70)



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP END-TO-END AGENT")
    print("=" * 70)

    print("\nLoading agent...")

    agent = AmazonHelpAgent(
        top_k=5,
        min_evidence_score=0.35,
    )

    print(
        "Agent loaded successfully."
    )


    # Test cases


    test_cases = [

        "My package says delivered but I haven't received it",

        "Why is Prime Video not working?",

        "I want a refund for my order",

        "My account is locked",

        "This is terrible support!",

        "Something is wrong with my order",

    ]


    # Run agent


    for customer_message in test_cases:

        result = agent.process(
            customer_message
        )

        print_result(
            result
        )



# ENTRY POINT


if __name__ == "__main__":
    main()
