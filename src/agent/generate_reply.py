import re
from typing import Any, Dict, List, Optional


# TEXT UTILITIES


def clean_text(text: str) -> str:
    """
    Basic text normalization.
    """

    if not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_agent_handle(text: str) -> str:
    """
    Remove leading @username from historical AmazonHelp replies.
    """

    if not text:
        return ""

    return re.sub(
        r"^@\w+\s*",
        "",
        text.strip()
    ).strip()



# EVIDENCE SELECTION


def select_best_evidence(
    retrieved_results: List[Dict[str, Any]],
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Select useful historical responses from retrieval results.
    """

    if not retrieved_results:
        return []

    valid_results = []

    for result in retrieved_results:

        score = float(
            result.get("score", 0.0)
        )

        response = str(
            result.get(
                "historical_amazonhelp_response",
                ""
            )
        ).strip()

        if score <= 0:
            continue

        if not response:
            continue

        valid_results.append(result)

    # Sort by similarity score
    valid_results.sort(
        key=lambda x: float(
            x.get("score", 0.0)
        ),
        reverse=True,
    )

    return valid_results[:max_results]



# RESPONSE CLEANING AND SAFETY


def is_safe_response(text: str) -> bool:
    """
    Check if a response is safe to copy without hallucinating specifics.
    Returns False if it contains specific actions, amounts, or timelines.
    """
    if not text:
        return False

    text_lower = text.lower()

    dangerous_patterns = [
        r"\$", r"£", r"€",              # Currency
        r"refund",                      # Specific actions taken
        r"cancel",
        r"process", r"ship",
        r"replacement",
        r"safari", r"browser",          # too specific context
        r"\d+",                         # block ANY numbers (dates, amounts)
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, text_lower):
            return False

    return True




def clean_historical_response(
    response: str,
) -> str:
    """
    Clean a historical AmazonHelp response.

    Removes:
    - Twitter handles
    - URLs
    - excessive whitespace

    Keeps:
    - actual support guidance
    """

    if not response:
        return ""

    # Remove leading Twitter handle
    response = remove_agent_handle(response)

    # Remove URLs
    response = re.sub(
        r"https?://\S+",
        "",
        response
    )

    # Remove excessive whitespace
    response = re.sub(
        r"\s+",
        " ",
        response
    )

    return response.strip()



# FALLBACK RESPONSE


def fallback_reply(
    customer_message: str,
    intent: Optional[str] = None,
) -> str:
    """
    Conservative fallback when historical evidence
    is insufficient.

    We deliberately avoid inventing a specific solution.
    """

    if intent == "delivery_issue":
        return (
            "I'm sorry you're having trouble with your delivery. "
            "Please contact Amazon support so the delivery details "
            "can be checked."
        )

    if intent == "refund_issue":
        return (
            "I'm sorry you're having trouble with your refund. "
            "Please contact Amazon support so we can review the "
            "available options."
        )

    if intent == "payment_issue":
        return (
            "I'm sorry you're having trouble with your payment. "
            "Please contact Amazon support so the issue can be "
            "checked securely."
        )

    if intent == "account_issue":
        return (
            "I'm sorry you're having trouble with your account. "
            "Please contact Amazon support so we can help securely."
        )

    if intent == "prime_video_issue":
        return (
            "I'm sorry you're having trouble with Prime Video. "
            "Please contact Amazon support or provide more details "
            "about the issue so it can be checked."
        )

    if intent == "device_issue":
        return (
            "I'm sorry you're having trouble with your device. "
            "Please contact Amazon support so we can help "
            "troubleshoot the issue."
        )

    if intent == "order_issue":
        return (
            "I'm sorry you're having trouble with your order. "
            "Please contact Amazon support so the order details "
            "can be checked."
        )

    if intent == "shipping_issue":
        return (
            "I'm sorry you're having trouble with shipping. "
            "Please contact Amazon support so the shipment "
            "details can be checked."
        )

    if intent == "product_or_content_issue":
        return (
            "I'm sorry you're having trouble with the product or "
            "content. Please contact Amazon support so we can "
            "help investigate the issue."
        )

    if intent == "support_complaint":
        return (
            "I'm sorry you've had this experience. "
            "Please contact Amazon support so we can look into "
            "the issue."
        )

    return (
        "Sorry you're having trouble. "
        "Please contact Amazon support so we can help investigate "
        "the issue."
    )



# GROUNDED REPLY GENERATION


def generate_reply(
    customer_message: str,
    retrieved_results: Optional[List[Dict[str, Any]]] = None,
    intent: Optional[str] = None,
    min_evidence_score: float = 0.35,
    max_evidence: int = 2,
) -> Dict[str, Any]:
    """
    Generate a support reply grounded in historical
    AmazonHelp responses.

    Parameters
    ----------
    customer_message:
        Incoming customer message.

    retrieved_results:
        Results returned by AmazonHelpRetriever.retrieve().

    intent:
        Predicted intent.

    min_evidence_score:
        Minimum retrieval similarity required before
        historical evidence is used.

    max_evidence:
        Maximum number of historical examples used.

    Returns
    -------
    dict
        Reply and grounding metadata.
    """


    # Validate customer message


    if not customer_message or not customer_message.strip():
        return {
            "reply": "",
            "grounded": False,
            "evidence": [],
            "reason": "Empty customer message.",
        }

    retrieved_results = retrieved_results or []


    # Select evidence


    evidence = select_best_evidence(
        retrieved_results,
        max_results=max_evidence,
    )

    # Only use sufficiently similar evidence
    strong_evidence = [
        result
        for result in evidence
        if float(
            result.get("score", 0.0)
        ) >= min_evidence_score
    ]


    # No sufficiently strong evidence


    if not strong_evidence:

        reply = fallback_reply(
            customer_message,
            intent=intent,
        )

        return {
            "reply": reply,
            "grounded": False,
            "evidence": [],
            "reason": (
                "No historical example exceeded "
                f"similarity threshold "
                f"{min_evidence_score:.2f}."
            ),
        }


    # Extract historical responses


    historical_responses = []

    for result in strong_evidence:

        response = clean_historical_response(
            result.get(
                "historical_amazonhelp_response",
                ""
            )
        )

        if response:

            historical_responses.append(
                {
                    "response": response,
                    "score": float(
                        result.get("score", 0.0)
                    ),
                    "conversation_id": result.get(
                        "conversation_id"
                    ),
                }
            )


    # No usable historical response


    if not historical_responses:

        reply = fallback_reply(
            customer_message,
            intent=intent,
        )

        return {
            "reply": reply,
            "grounded": False,
            "evidence": [],
            "reason": (
                "Retrieved examples contained "
                "no usable response."
            ),
        }


    # Select strongest safe historical resolution

    best_safe = None
    for item in historical_responses:
        if is_safe_response(item["response"]):
            best_safe = item
            break

    if not best_safe:
        reply = fallback_reply(
            customer_message,
            intent=intent,
        )

        return {
            "reply": reply,
            "grounded": False,
            "evidence": historical_responses,
            "reason": (
                "Historical responses were retrieved but "
                "contained specific/unsafe claims (e.g. refunds). "
                "Fell back to safe template."
            ),
        }

    reply = best_safe["response"]


    # Return result


    return {
        "reply": reply,
        "grounded": True,
        "evidence": historical_responses,
        "reason": (
            "Reply grounded in the highest-scoring "
            "safe historical AmazonHelp response."
        ),
    }



# FORMAT EVIDENCE


def format_evidence(
    evidence: List[Dict[str, Any]]
) -> str:
    """
    Create a human-readable evidence block.
    """

    if not evidence:
        return "No historical evidence."

    lines = []

    for i, item in enumerate(
        evidence,
        start=1,
    ):

        lines.append(
            f"[Evidence {i}] "
            f"similarity={item['score']:.4f}"
        )

        lines.append(
            f"Conversation: "
            f"{item.get('conversation_id')}"
        )

        lines.append(
            f"Historical response: "
            f"{item['response']}"
        )

    return "\n".join(lines)



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP GROUNDED REPLY GENERATOR")
    print("=" * 70)

    from src.retrieval.retrieve import AmazonHelpRetriever

    print("\nLoading retrieval index...")

    retriever = AmazonHelpRetriever()

    print(
        f"Documents loaded : "
        f"{len(retriever.documents):,}"
    )

    print(
        f"TF-IDF features  : "
        f"{retriever.matrix.shape[1]:,}"
    )


    # Test queries


    test_queries = [
        (
            "My package says delivered but I haven't received it",
            "delivery_issue",
        ),
        (
            "Why is Prime Video not working?",
            "prime_video_issue",
        ),
        (
            "I want a refund for my order",
            "refund_issue",
        ),
    ]


    # Run tests


    for customer_message, intent in test_queries:

        print("\n" + "-" * 70)

        print("CUSTOMER:")
        print(customer_message)

        # Retrieve historical examples
        results = retriever.retrieve(
            query=customer_message,
            top_k=5,
        )

        print(
            f"\nRetrieved results: "
            f"{len(results)}"
        )

        # Generate grounded response
        output = generate_reply(
            customer_message=customer_message,
            retrieved_results=results,
            intent=intent,
        )

        print("\nGENERATED REPLY:")
        print(output["reply"])

        print(
            f"\nGrounded: "
            f"{output['grounded']}"
        )

        print(
            f"Reason: "
            f"{output['reason']}"
        )

        print("\nEVIDENCE:")

        print(
            format_evidence(
                output["evidence"]
            )
        )

    print("\n" + "=" * 70)
    print("REPLY GENERATION TEST COMPLETE")
    print("=" * 70)



# ENTRY POINT


if __name__ == "__main__":
    main()
