import copy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.evaluate import (
    build_llm_judge_cache_key,
    lookup_cached_judgement,
)


def current_example():
    """A minimal evaluation result containing all cache-sensitive fields."""

    return {
        "example_id": "golden_cache_test",
        "customer_message": "My parcel shows delivered, but it is missing.",
        "predicted_intent": "delivery_issue",
        "intent_confidence": 0.91,
        "predicted_action": "escalate",
        "reply": "Please contact support so the delivery details can be checked.",
        "top_retrieval_score": 0.82,
        "second_retrieval_score": 0.44,
        "retrieval_score_gap": 0.38,
        "retrieved_evidence": [
            {
                "rank": 1,
                "score": 0.82,
                "conversation_id": "amazonhelp_1",
                "customer_message": "Tracking says delivered but no parcel arrived.",
                "historical_amazonhelp_response": "Please contact us so we can investigate.",
                "candidate_intent": "delivery_issue",
                "intent_compatible": True,
            }
        ],
    }


def test_llm_judge_cache_reuses_only_identical_agent_outputs():
    original = current_example()
    original_key = build_llm_judge_cache_key(original)
    cache = {
        original_key: {
            "example_id": original["example_id"],
            "cache_key": original_key,
            "judge": {"overall": 4},
        }
    }

    # A: exact current evaluation input/output is a cache hit.
    assert build_llm_judge_cache_key(copy.deepcopy(original)) == original_key
    assert lookup_cached_judgement(cache, original_key) is cache[original_key]

    # B: changing only the generated reply invalidates the old judgement.
    changed_reply = copy.deepcopy(original)
    changed_reply["reply"] = "A different current response."
    changed_reply_key = build_llm_judge_cache_key(changed_reply)
    assert changed_reply_key != original_key
    assert lookup_cached_judgement(cache, changed_reply_key) is None

    # C: changing only retrieved historical evidence invalidates it too.
    changed_evidence = copy.deepcopy(original)
    changed_evidence["retrieved_evidence"][0][
        "historical_amazonhelp_response"
    ] = "A different historical resolution."
    changed_evidence_key = build_llm_judge_cache_key(changed_evidence)
    assert changed_evidence_key != original_key
    assert lookup_cached_judgement(cache, changed_evidence_key) is None

    # D: changing only the action must also force a new judgement.
    changed_action = copy.deepcopy(original)
    changed_action["predicted_action"] = "auto_handle"
    changed_action_key = build_llm_judge_cache_key(changed_action)
    assert changed_action_key != original_key
    assert lookup_cached_judgement(cache, changed_action_key) is None


if __name__ == "__main__":
    test_llm_judge_cache_reuses_only_identical_agent_outputs()
    print(
        "cache hit: same input; cache misses: changed reply, "
        "evidence, and action"
    )
