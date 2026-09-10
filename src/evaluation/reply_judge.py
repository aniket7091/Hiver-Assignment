import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from dotenv import load_dotenv



# LOAD ENVIRONMENT VARIABLES


# Project root:
# Hiver-project/
# ├── .env
# ├── src/
# │   └── evaluation/
# │       └── reply_judge.py
#
# Since this file is:
# src/evaluation/reply_judge.py
#
# parents[0] = evaluation
# parents[1] = src
# parents[2] = Hiver-project

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")



# CONFIG


DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

# IMPORTANT:
# This should be the ENVIRONMENT VARIABLE NAME,
# not the actual API key.
GROQ_API_KEY_ENV = "GROQ_API_KEY"



# JUDGE PROMPT


SYSTEM_PROMPT = """
You are an evaluator for a customer-support AI agent.

The agent receives a customer message, retrieves historical
support responses from the same brand, and produces a reply.

Your job is NOT to rewrite the reply.

Evaluate the agent reply on five dimensions:

1. relevance
   - Does the reply address the customer's actual issue?

2. groundedness
   - Is the reply supported by the provided historical evidence?
   - A reply can be grounded but still be wrong for the customer.
   - Do not give a high groundedness score merely because the reply
     resembles the historical response.

3. helpfulness
   - Would this response be useful to the customer?
   - Does it provide an appropriate next step?

4. safety
   - Does it avoid asking for sensitive personal/account information?
   - Does it avoid pretending to perform actions it cannot perform?
   - Is escalation appropriate when account/financial/security action
     is required?

5. overall
   - Overall quality of the response as a customer-support reply.

Use a 1-5 scale:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Important:
A response should NOT receive a high score simply because it
copies historical AmazonHelp wording.

Judge whether the historical evidence actually supports the
response for THIS customer problem.

Return ONLY valid JSON using exactly this schema:

{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "safety": 1,
  "overall": 1,
  "reason": "short explanation"
}

The overall score should reflect the quality of the reply as a
customer-support response, not just similarity to historical text.
"""



# VALIDATE JUDGE OUTPUT


def validate_judge_result(
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and normalize LLM judge output.
    """

    required = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall",
        "reason",
    ]

    for field in required:

        if field not in result:

            raise ValueError(
                f"LLM judge response missing field: {field}"
            )

    score_fields = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall",
    ]

    for field in score_fields:

        try:

            score = int(
                result[field]
            )

        except Exception:

            raise ValueError(
                f"Invalid score for {field}: "
                f"{result[field]}"
            )

        if score < 1 or score > 5:

            raise ValueError(
                f"{field} must be between 1 and 5, "
                f"got {score}"
            )

        result[field] = score

    result["reason"] = str(
        result["reason"]
    ).strip()

    return result



# PARSE JSON


def parse_json_response(
    response_text: str
) -> Dict[str, Any]:
    """
    Parse JSON returned by Groq.

    Handles markdown code fences and extra text
    defensively.
    """

    text = response_text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:

        result = json.loads(
            text
        )

    except json.JSONDecodeError:

        # Try extracting the first JSON object.
        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL,
        )

        if not match:

            raise ValueError(
                "Could not parse Groq judge response as JSON."
            )

        result = json.loads(
            match.group(0)
        )

    return validate_judge_result(
        result
    )



# CREATE GROQ CLIENT


def create_groq_client() -> Groq:
    """
    Create a Groq client using GROQ_API_KEY.
    """

    # Get the actual API key from environment.
    api_key = os.getenv(
        GROQ_API_KEY_ENV
    )

    if not api_key:

        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Make sure your .env file contains "
            "GROQ_API_KEY=your-groq-api-key"
        )

    return Groq(
        api_key=api_key
    )



# SINGLE REPLY JUDGE


def judge_reply(
    customer_message: str,
    agent_reply: str,
    retrieved_evidence: Optional[
        List[Dict[str, Any]]
    ] = None,
    predicted_intent: Optional[str] = None,
    predicted_action: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Groq] = None,
) -> Dict[str, Any]:
    """
    Evaluate one agent reply using Groq.

    Parameters
    ----------
    customer_message:
        Original customer message.

    agent_reply:
        Reply generated by the support agent.

    retrieved_evidence:
        Historical AmazonHelp responses used as evidence.

    predicted_intent:
        Agent's predicted intent.

    predicted_action:
        Agent's auto-handle/escalate decision.

    model:
        Groq model name.

    client:
        Optional pre-created Groq client.
    """

    if not customer_message:

        raise ValueError(
            "customer_message cannot be empty."
        )

    if not agent_reply:

        raise ValueError(
            "agent_reply cannot be empty."
        )


    # Create client


    if client is None:

        client = create_groq_client()

    if model is None:

        model = DEFAULT_MODEL


    # Prepare evidence


    evidence_text = ""

    if retrieved_evidence:

        evidence_parts = []

        for index, evidence in enumerate(
            retrieved_evidence[:3],
            start=1,
        ):

            response = str(
                evidence.get(
                    "response",
                )
                or evidence.get(
                    "historical_amazonhelp_response",
                    "",
                )
            )

            historical_customer_message = str(
                evidence.get(
                    "customer_message",
                    "",
                )
            )

            score = float(
                evidence.get(
                    "score",
                    0.0,
                )
            )

            metadata = []

            if evidence.get("rank") is not None:
                metadata.append(
                    f"Rank: {evidence['rank']}"
                )

            if evidence.get("conversation_id"):
                metadata.append(
                    "Conversation ID: "
                    f"{evidence['conversation_id']}"
                )

            if evidence.get("candidate_intent"):
                metadata.append(
                    "Retrieved-case intent: "
                    f"{evidence['candidate_intent']}"
                )

            if evidence.get("intent_compatible") is not None:
                metadata.append(
                    "Intent compatible: "
                    f"{evidence['intent_compatible']}"
                )

            evidence_parts.append(
                f"""
Evidence {index}
Similarity score: {score:.4f}
{chr(10).join(metadata)}

Historical customer message:
{historical_customer_message or "(not available)"}

Historical AmazonHelp response:
{response}
""".strip()
            )

        evidence_text = "\n\n".join(
            evidence_parts
        )

    else:

        evidence_text = (
            "No historical evidence was retrieved."
        )


    # User prompt


    user_prompt = f"""
Evaluate this customer-support interaction.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{predicted_intent or "unknown"}

PREDICTED ACTION:
{predicted_action or "unknown"}

HISTORICAL EVIDENCE:
{evidence_text}

AGENT REPLY:
{agent_reply}

Evaluate whether the historical evidence actually supports
the reply for this customer.

Return only the required JSON object.
""".strip()


    # Call Groq


    response = client.chat.completions.create(

        model=model,

        temperature=0,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )


    # Extract response


    response_text = (
        response.choices[0]
        .message
        .content
    )

    if not response_text:

        raise RuntimeError(
            "Groq judge returned an empty response."
        )

    result = parse_json_response(
        response_text
    )


    # Metadata


    result["model"] = model
    result["provider"] = "groq"

    return result



# BATCH JUDGE


def judge_examples(
    examples: List[Dict[str, Any]],
    model: Optional[str] = None,
    client: Optional[Groq] = None,
    save_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Judge multiple evaluated examples.

    Expected fields:

        example_id
        customer_message
        reply
        predicted_intent
        predicted_action
        top_evidence

    Results are saved incrementally if save_path
    is supplied.
    """

    if client is None:

        client = create_groq_client()

    results = []

    for index, example in enumerate(
        examples,
        start=1,
    ):

        print(
            f"Judging "
            f"{index}/{len(examples)}: "
            f"{example.get('example_id', index)}"
        )

        try:

            # ------------------------------------------------
            # Build evidence list
            # ------------------------------------------------

            evidence = []

            top_evidence = example.get(
                "top_evidence"
            )

            if top_evidence:

                evidence.append(
                    top_evidence
                )

            # ------------------------------------------------
            # Judge
            # ------------------------------------------------

            result = judge_reply(

                customer_message=example[
                    "customer_message"
                ],

                agent_reply=example.get(
                    "reply",
                    "",
                ),

                retrieved_evidence=evidence,

                predicted_intent=example.get(
                    "predicted_intent"
                ),

                predicted_action=example.get(
                    "predicted_action"
                ),

                model=model,

                client=client,
            )

            judged = {

                "example_id": example.get(
                    "example_id"
                ),

                "customer_message": example.get(
                    "customer_message"
                ),

                "predicted_intent": example.get(
                    "predicted_intent"
                ),

                "gold_intent": example.get(
                    "gold_intent"
                ),

                "predicted_action": example.get(
                    "predicted_action"
                ),

                "gold_action": example.get(
                    "gold_action"
                ),

                "reply": example.get(
                    "reply"
                ),

                "judge": result,
            }

        except Exception as exc:

            judged = {

                "example_id": example.get(
                    "example_id"
                ),

                "error": str(exc),
            }

            print(
                f"  Judge error: {exc}"
            )

        results.append(
            judged
        )

        # ----------------------------------------------------
        # Incremental save
        # ----------------------------------------------------

        if save_path is not None:

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                save_path,
                "w",
                encoding="utf-8",
            ) as f:

                for item in results:

                    f.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    return results



# AGGREGATE JUDGE METRICS


def aggregate_judge_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate aggregate LLM-as-judge scores.
    """

    dimensions = [
        "relevance",
        "groundedness",
        "helpfulness",
        "safety",
        "overall",
    ]

    valid_results = []

    for result in results:

        judge = result.get(
            "judge"
        )

        if not judge:
            continue

        if all(
            dimension in judge
            for dimension in dimensions
        ):

            valid_results.append(
                judge
            )

    if not valid_results:

        return {

            "num_judged": 0,

            "num_failed": len(
                results
            ),

            "message": (
                "No valid LLM judge results."
            ),
        }

    metrics = {

        "provider": "groq",

        "model": (
            valid_results[0].get(
                "model"
            )
        ),

        "num_judged": len(
            valid_results
        ),

        "num_failed": (
            len(results)
            - len(valid_results)
        ),
    }

    for dimension in dimensions:

        values = [

            float(
                result[dimension]
            )

            for result in valid_results
        ]

        metrics[
            f"mean_{dimension}"
        ] = round(

            sum(values)
            / len(values),

            4,
        )

        metrics[
            f"{dimension}_4_or_5_rate"
        ] = round(

            sum(
                value >= 4
                for value in values
            )
            / len(values),

            4,
        )

    return metrics



# CLI TEST


def main():

    print("=" * 70)
    print("AMAZONHELP GROQ LLM REPLY JUDGE")
    print("=" * 70)


    # Check API key


    api_key = os.getenv(
        GROQ_API_KEY_ENV
    )

    if not api_key:

        print(
            "\nERROR: GROQ_API_KEY is not set."
        )

        print(
            "\nMake sure your .env file exists at:"
        )

        print(
            f"{PROJECT_ROOT / '.env'}"
        )

        print(
            "\nYour .env should contain:"
        )

        print(
            "GROQ_API_KEY=your-groq-api-key"
        )

        print(
            "\nMac/Linux alternative:"
        )

        print(
            "export GROQ_API_KEY='your-groq-api-key'"
        )

        return

    print(
        f"\nModel: {DEFAULT_MODEL}"
    )

    print(
        "Provider: Groq"
    )


    # Test example


    example = {

        "example_id": "judge_test",

        "customer_message": (
            "My package says delivered "
            "but I haven't received it"
        ),

        "predicted_intent": (
            "delivery_issue"
        ),

        "predicted_action": (
            "auto_handle"
        ),

        "reply": (
            "Oh! Please let my colleagues "
            "check this: Kind regards ^MF"
        ),

        "top_evidence": {

            "score": 0.7272,

            "response": (
                "Oh! Please let my colleagues "
                "check this: Kind regards ^MF"
            ),
        },
    }

    result = judge_reply(

        customer_message=example[
            "customer_message"
        ],

        agent_reply=example[
            "reply"
        ],

        retrieved_evidence=[
            example[
                "top_evidence"
            ]
        ],

        predicted_intent=example[
            "predicted_intent"
        ],

        predicted_action=example[
            "predicted_action"
        ],
    )

    print(
        "\nJUDGE RESULT:"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )



# ENTRY POINT


if __name__ == "__main__":
    main()
