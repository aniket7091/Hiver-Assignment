# Decision Log

## 1. Selected AmazonHelp as the target brand

**Decision:** Use AmazonHelp as the single brand for the support agent.
**Why:** It has a large number of outbound support responses in the dataset, providing enough historical conversations for retrieval and evaluation.

## 2. Filter by `author_id`, not tweet text

**Decision:** Identify AmazonHelp messages using `author_id == "amazonhelp"` rather than searching for `@AmazonHelp` in tweet text.
**Why:** Mentions can occur in customer messages and do not reliably identify the responding brand account.

## 3. Reconstruct threads before modelling

**Decision:** Build conversation threads using `response_tweet_id` and `in_response_to_tweet_id`.
**Why:** Individual tweets often lack enough context to understand the customer's issue or the historical resolution.

## 4. Use the first customer message for intent discovery

**Decision:** Use the first customer message in each reconstructed thread as the primary intent signal.
**Why:** It represents the initial support request before the conversation becomes influenced by the support interaction.

## 5. Discover intents with clustering, then manually consolidate

**Decision:** Use TF-IDF + KMeans to discover recurring patterns, followed by manual consolidation into a smaller business taxonomy.
**Why:** Fully manual taxonomy discovery over tens of thousands of conversations would be inefficient, while raw clusters were too noisy to use directly.

## 6. Keep an `other` intent

**Decision:** Include `other` as a valid intent.
**Why:** Many Twitter messages are extremely short, noisy, or ambiguous. Forcing these messages into a business category would create unreliable labels.

## 7. Keep the golden set separate from weak-label training

**Decision:** Do not use the manually labelled 200-example golden set to train the classifier.
**Why:** This prevents evaluation leakage and keeps the golden set as an independent evaluation benchmark.

## 8. Use historical AmazonHelp responses as retrieval evidence

**Decision:** Ground reply drafts in previously observed AmazonHelp responses.
**Why:** The assignment specifically asks for replies grounded in how the brand historically resolved similar issues.

## 9. Prefer conservative fallback over unsupported replies

**Decision:** If retrieval confidence is below the configured threshold, return a conservative fallback rather than inventing a response.
**Why:** A support agent should prefer escalation/uncertainty over unsupported claims.

## 10. Treat retrieval ambiguity explicitly

**Decision:** Use the score gap between the top retrieval candidates as an ambiguity signal.
**Why:** A high top score alone can be misleading when multiple unrelated candidates have similarly high scores.

## 11. Escalate sensitive account and financial issues

**Decision:** Account, payment, refund, and support-complaint cases receive conservative escalation treatment.
**Why:** Incorrect automated handling of sensitive account or financial issues can cause substantially more harm than an unnecessary escalation.

## 12. Give high-risk rules precedence over generic confidence rules

**Decision:** Explicit high-risk intents override generic low-intent-confidence decisions.
**Why:** A high classifier confidence should not make a sensitive operation automatically safe.

## 13. Evaluate auto-handle precision separately from escalation recall

**Decision:** Report both metrics rather than only overall action accuracy.
**Why:** Overall accuracy can hide an unsafe or overly conservative system. These metrics expose the trade-off between automation and safety.

## 14. Add independent annotation agreement

**Decision:** Evaluate human agreement on a separate 40-example annotation set.
**Why:** Model performance alone does not reveal how consistently humans interpret the intent and action taxonomy.

## 15. Report the LLM-judge quota limitation

**Decision:** Report the completed LLM-judge sample and provider quota limitation rather than presenting it as a full evaluation.
**Why:** An incomplete evaluation should not be represented as if all intended examples were successfully judged.
