import re
from dataclasses import dataclass
from typing import List


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


INTENT_DESCRIPTIONS = {
    "delivery_issue": (
        "Problems with a package or delivery, including "
        "late, missing, or incorrectly marked delivered packages."
    ),

    "order_issue": (
        "Problems specifically related to an Amazon order, "
        "including order status, order details, or order-specific issues."
    ),

    "shipping_issue": (
        "Questions or complaints about shipping methods, "
        "shipping speed, Prime shipping, or promised delivery speed."
    ),

    "payment_issue": (
        "Problems involving charges, payment methods, "
        "billing, unauthorized charges, or payment processing."
    ),

    "refund_issue": (
        "Requests or problems involving refunds, returned money, "
        "refund status, or money-back requests."
    ),

    "account_issue": (
        "Problems involving the Amazon account, account access, "
        "account closure, or account settings."
    ),

    "prime_video_issue": (
        "Problems with Prime Video, including playback, "
        "streaming, video errors, or video quality."
    ),

    "device_issue": (
        "Problems with Amazon devices or related hardware, "
        "such as Fire TV, Echo, Kindle, or device functionality."
    ),

    "product_or_content_issue": (
        "Feedback or problems concerning a product, movie, "
        "show, listing, or other Amazon content."
    ),

    "support_complaint": (
        "Complaints about Amazon customer service, previous "
        "support interactions, or dissatisfaction with support."
    ),

    "other": (
        "Messages that do not contain enough information to "
        "assign a specific support intent, including greetings, "
        "thanks, reactions, or unrelated messages."
    ),
}


# These patterns deliberately require an explicit complaint about support or
# service. A bare negative adjective such as "bad" or "poor" is not enough to
# turn an operational request into a support complaint.
SUPPORT_COMPLAINT_PATTERNS = (
    r"\b(?:terrible|awful|horrible|worst|unacceptable|useless)\s+"
    r"(?:customer\s+)?(?:support|service|care|handling)\b",
    r"\b(?:very\s+)?poor\s+customer\s+"
    r"(?:support|service|care|handling)\b",
    r"\bcustomer\s+(?:support|service|care|handling)\s+(?:is\s+)?"
    r"(?:terrible|awful|horrible|worst|unacceptable|useless)\b",
    r"\b(?:no\s+one|nobody)\s+(?:is\s+)?help(?:ing|s)?\b",
    r"\b(?:still\s+)?no\s+(?:help|resolution)\s+"
    r"(?:from|by)\s+(?:support|customer\s+service|you)\b",
)


def has_support_complaint_signal(text: str) -> bool:
    """Detect explicit dissatisfaction with support/service, conservatively."""

    if not text:
        return False

    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()

    return any(
        re.search(pattern, normalized) is not None
        for pattern in SUPPORT_COMPLAINT_PATTERNS
    )


@dataclass
class IntentPrediction:
    intent: str
    confidence: float


def is_valid_intent(intent: str) -> bool:
    return intent in INTENTS


def get_intent_description(intent: str) -> str:
    return INTENT_DESCRIPTIONS.get(
        intent,
        INTENT_DESCRIPTIONS["other"]
    )


def get_all_intents() -> List[str]:
    return INTENTS.copy()
