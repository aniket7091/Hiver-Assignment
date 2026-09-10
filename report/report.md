# Hiver  Assignment

## AI Customer Support Agent for AmazonHelp

### 1. Problem Framing

I built an AI support agent for **AmazonHelp** using the Kaggle Customer Support on Twitter dataset.

The agent performs three tasks:

1. **Intent classification** — classify an incoming customer message into a compact taxonomy of recurring support issues.
2. **Reply drafting** — retrieve historically similar AmazonHelp conversations and use the historical response as grounded evidence for a draft reply.
3. **Handle vs escalate** — decide whether the issue can be safely auto-handled or should be escalated, with an explicit reason.

The key design principle was **conservatism**: the system should avoid confidently answering when historical evidence is weak, retrieval is ambiguous, or the issue involves sensitive account/financial actions.

---

## 2. Data and Thread Reconstruction

The source dataset contains approximately 3M customer-support tweets with fields including tweet ID, author ID, inbound/outbound direction, timestamps, text, and reply relationships.

I selected **AmazonHelp** as the brand.

Rather than searching for `@AmazonHelp` inside tweet text, I filtered on the dataset's `author_id` field. This produced:

* **169,840 AmazonHelp tweets**
* All selected AmazonHelp tweets were outbound (`inbound=False`)
* **0 duplicate tweet IDs**
* **324,881 relevant tweets** after adding customer messages needed for conversation reconstruction
* **86,658 reconstructed conversation threads**
* **324,787 messages** in the final thread set
* Average thread length: **3.75 messages**
* **41,429 threads** contain multiple AmazonHelp responses

Threads were reconstructed using `response_tweet_id` and `in_response_to_tweet_id`, followed by chronological ordering and validation. Validation found **0 chronological ordering errors** and no empty threads lacking the required customer/AmazonHelp messages.

This reconstruction was important because individual tweets often lack enough context to determine the customer's actual issue or how AmazonHelp resolved it.

---

## 3. Intent Taxonomy

I first used unsupervised clustering to discover recurring issue patterns, then manually consolidated noisy clusters into a compact support taxonomy.

The final taxonomy contains 10 business intents plus `other`:

| Intent                     | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| `delivery_issue`           | Delivered/late/missing delivery problems                       |
| `order_issue`              | Problems with an existing order                                |
| `shipping_issue`           | Shipping availability, destination, or shipping questions      |
| `payment_issue`            | Payment method/payment processing issues                       |
| `refund_issue`             | Refund or money-return issues                                  |
| `account_issue`            | Account access, closure, or account-related issues             |
| `prime_video_issue`        | Prime Video playback/subscription/content issues               |
| `device_issue`             | Kindle/Fire/Amazon device issues                               |
| `product_or_content_issue` | Product, media, or content-specific questions                  |
| `support_complaint`        | Complaints about support/service experience                    |
| `other`                    | Messages that cannot be reliably assigned to a business intent |

The `other` class is deliberate. Many tweets are extremely short, noisy, or lack sufficient context, so forcing every message into a specific business category would create misleading labels.

---

## 4. System Design

The pipeline has four major components:

```text
Customer message
      |
      v
Intent classifier
      |
      +--------------------+
      |                    |
      v                    v
Retrieval              Escalation policy
      |                    |
      v                    v
Historical AmazonHelp   Auto-handle /
response evidence       Escalate + reason
      |
      v
Grounded reply draft
```

### Intent classification

The production classifier is a TF-IDF + Logistic Regression model trained on weak labels generated from recurring patterns in the historical data.

The model is intentionally lightweight so the complete project can run locally without a large model dependency.

### Historical retrieval

I built a TF-IDF retrieval index over reconstructed AmazonHelp conversations.

The index contains:

* **86,658 threads**
* **86,220 searchable documents**
* 30,000-token TF-IDF vocabulary

Retrieval returns the most similar historical customer-support conversation and its AmazonHelp response.

### Reply generation

The current implementation uses the highest-confidence historical AmazonHelp response as the primary grounded draft rather than generating unsupported new facts.

If retrieval confidence is below the configured threshold, the agent falls back to a conservative response rather than pretending that it has strong historical evidence.

### Escalation

Escalation rules consider:

* retrieval confidence
* retrieval score gap/ambiguity
* intent confidence
* sensitive account actions
* payment/refund issues
* support complaints
* insufficient information
* complex issues

Explicit high-risk categories such as account, payment, refund, and support complaints take precedence over generic low-confidence rules.

---

# 5. Evaluation

The evaluation set contains **200 manually labelled examples** sampled from reconstructed AmazonHelp threads.

Sampling used heuristic signals only to improve coverage across issue types and difficulty; the final intent/action/escalation labels were manually assigned.

Final golden-set distribution:

| Intent                   |   Count |
| ------------------------ | ------: |
| delivery_issue           |      36 |
| other                    |      31 |
| product_or_content_issue |      27 |
| order_issue              |      21 |
| account_issue            |      17 |
| shipping_issue           |      13 |
| refund_issue             |      12 |
| payment_issue            |      12 |
| prime_video_issue        |      11 |
| support_complaint        |      10 |
| device_issue             |      10 |
| **Total**                | **200** |

Actions:

* **141 escalate**
* **59 auto-handle**

This distribution is intentionally not balanced and reflects the sampled evaluation set.

### Baselines

| Model                        |  Accuracy |   Macro F1 | Weighted F1 |
| ---------------------------- | --------: | ---------: | ----------: |
| Majority baseline            |     18.0% |     0.0277 |      0.0549 |
| TF-IDF + Logistic Regression |     35.0% |     0.3449 |      0.3359 |
| Full agent                   | **44.0%** | **0.4544** |  **0.4281** |

The full agent therefore improves intent classification over both simple baselines, although the absolute accuracy remains far from production-ready.

### Agent-level metrics

| Metric                |     Result |
| --------------------- | ---------: |
| Intent accuracy       |  **44.0%** |
| Intent macro F1       | **0.4544** |
| Intent weighted F1    | **0.4281** |
| Action accuracy       |  **65.5%** |
| Auto-handle precision |  **22.2%** |
| Escalation recall     |  **90.1%** |
| Grounded reply rate   |  **72.0%** |
| Retrieval available   |   **100%** |

The action results show an important trade-off: the system has high escalation recall but low auto-handle precision. This is consistent with the conservative design goal, but it also means the current system escalates too many cases.

---

# 6. LLM-as-Judge Evaluation

I additionally evaluated generated replies using an LLM judge on a selected subset of examples.

The judge evaluated:

* relevance
* groundedness
* helpfulness
* safety
* overall quality

The evaluation used `openai/gpt-oss-120b` through Groq.

The provider quota was reached during evaluation, so only **32 examples completed successfully** rather than the intended larger sample.

Results from the completed evaluations:

| Dimension    |     Mean |
| ------------ | -------: |
| Relevance    | 2.72 / 5 |
| Groundedness | 3.06 / 5 |
| Helpfulness  | 2.28 / 5 |
| Safety       | 4.97 / 5 |
| Overall      | 2.66 / 5 |

This result is intentionally reported as **quota-limited**, not as a full 200-example LLM evaluation.

The strongest signal is safety: the conservative escalation strategy generally avoids unsafe responses. Helpfulness and relevance are substantially weaker because historical retrieval can return a superficially similar but operationally incorrect response.

---

# 7. Annotation Consistency Check

A second annotation set of **40 examples** was created as a diagnostic consistency check.

The second-pass labels were prepared using **model-assisted suggestions rather than being independently produced by a separate blinded human annotator**. Therefore, these results must **not** be described as genuine independent human inter-annotator agreement.

Results from this second-pass consistency check:

| Dimension | Raw Agreement | Cohen's κ |
| ------------------------------- | --------- | ---------- |
| Intent | **67.5%** | **0.6315** |
| Action | **92.5%** | **0.7931** |
| Escalation reason | **2.5%** | — |

The intent and action results are useful as a diagnostic signal about taxonomy consistency, while the very low escalation-reason agreement indicates that the reason taxonomy is currently much more subjective than the primary intent/action labels.

This limitation is important to report rather than hide. The proper next step is to have a **separate annotator independently label a fresh 40–50 example subset without access to the primary labels or model suggestions**, then compute genuine inter-annotator agreement.

---

# 8. Top 5 Failure Modes
## 5. Historical responses can be grounded but not helpful

The grounded reply rate is **72.0%**, while LLM-judge helpfulness is only **2.28/5** on the quota-limited sample.

This demonstrates that simply retrieving a historical AmazonHelp response is not sufficient.

**Hypothesis:** historical replies may be appropriate for their original context but incomplete or awkward when reused for a new customer message.

**Fix:** retrieve multiple candidates, verify that the candidate matches the predicted intent, then use an LLM to synthesize a concise response strictly from the retrieved evidence.

## 1. Operational words overpower the real intent

**Example — `golden_0002`**

The customer complains about a support experience while mentioning a paid shoe rack/order. The gold label is `support_complaint`, but the classifier predicts `delivery_issue`.

**Hypothesis:** the classifier overweights words such as "paid", "delivery", and product/order terminology while underweighting the customer's actual complaint about support.

**Fix:** add more complaint examples and hierarchical classification separating "what happened" from "how the customer feels about support".

---

## 2. High retrieval score does not guarantee useful evidence

**Example — `golden_0010`**

The customer asks whether a Pokémon game can be delivered to the Philippines. The correct intent is `shipping_issue`, but the classifier predicts `other`.

Retrieval returns a very high-scoring result, yet the resulting response is not useful for the actual question.

**Hypothesis:** TF-IDF similarity is lexical rather than semantic. Shared words can produce a high score even when the operational intent differs.

**Fix:** replace or augment TF-IDF retrieval with dense embeddings and apply intent-aware retrieval filtering.

---

## 3. Identical lexical scores can hide ambiguity

**Example — `golden_0028`**

The message is essentially "help, Amazon". The top two retrieval scores are both **1.0**, producing a score gap of **0**.

The system therefore has no reliable basis for choosing between the retrieved responses.

**Hypothesis:** very short messages contain insufficient information for lexical retrieval.

**Fix:** explicitly detect very short/low-information queries and escalate instead of retrieving arbitrary evidence.

---

## 4. Conservative escalation causes too many escalations

The system achieves **90.1% escalation recall**, but auto-handle precision is only **22.2%**.

**Hypothesis:** safety rules are effective at catching risky cases but are too broad for routine customer questions.

**Fix:** calibrate the escalation threshold on a larger labelled set and distinguish "uncertain" from "actually risky".

---

## 5. Historical responses can be grounded but not helpful

The grounded reply rate is **72.0%**, while LLM-judge helpfulness is only **2.05/5** on the quota-limited sample.

This demonstrates that simply retrieving a historical AmazonHelp response is not sufficient.

**Hypothesis:** historical replies may be appropriate for their original context but incomplete or awkward when reused for a new customer message.

**Fix:** retrieve multiple candidates, verify that the candidate matches the predicted intent, then use an LLM to synthesize a concise response strictly from the retrieved evidence.

---

# 9. What Is Misleading About My Headline Number?

The most tempting headline is:

> **"The agent achieves 44% intent accuracy."**

That number is useful, but misleading if interpreted as production performance.

First, the evaluation set contains only **200 manually labelled examples**, so it is a small sample relative to the original dataset.

Second, the agent's classifier is trained using weak labels derived from historical data, while the evaluation labels are manually assigned. Therefore the task includes label-boundary ambiguity.

Third, the taxonomy contains an intentionally broad `other` class. Some messages are inherently difficult to classify from a single tweet.

Fourth, the 40-example human agreement study shows that even humans do not always agree on intent labels: raw intent agreement is **67.5%**.

Finally, reply quality is not captured by intent accuracy. A system can correctly identify an intent while retrieving an unhelpful response.

Therefore, **44% should be treated as an honest baseline for this prototype, not as evidence that the system is ready for autonomous production support.**

The more meaningful result is the combination of:

* improvement over the majority and simple ML baselines,
* relatively strong second-pass consistency on action decisions,
* high escalation recall,
* conservative safety behaviour,
* and clearly identified retrieval/classification failure modes.

---

# 10. Non-Obvious Design Decisions

The following decisions were deliberately made during implementation:

1. Filter AmazonHelp using `author_id`, not only mentions in tweet text.
2. Reconstruct conversation threads before defining intents.
3. Use the first customer message as the primary intent signal.
4. Keep `other` instead of forcing ambiguous tweets into business categories.
5. Use unsupervised clustering only for taxonomy discovery, not as the final ground truth.
6. Keep the golden set separate from weak-label training.
7. Use historical AmazonHelp responses as retrieval evidence.
8. Add a minimum retrieval-confidence threshold.
9. Treat retrieval score gaps as an ambiguity signal.
10. Make account/payment/refund/support cases conservative by default.
11. Test escalation-rule precedence explicitly.
12. Report auto-handle precision separately from escalation recall.
13. Add a second-pass annotation consistency check rather than relying only on model metrics.
14. Report the LLM-judge quota limitation instead of presenting an incomplete sample as a full evaluation.
15. Keep failure examples tied to real golden-set records rather than synthetic examples.

---

# 11. Next-Week Plan

### P0 — Improve evaluation quality

* Expand the golden set beyond 200 examples.
* Obtain a genuinely independent second annotator and run a blinded 40–50 example agreement study.
* Rewrite escalation-reason guidelines because agreement is currently only 2.5%.
* Add confidence intervals to headline metrics.

### P1 — Improve retrieval

* Replace TF-IDF-only retrieval with dense embeddings.
* Retrieve top-k candidates instead of a single response.
* Filter candidates using predicted intent.
* Add an explicit "no trustworthy evidence" state.

### P1 — Improve classification

* Replace weak keyword labels with more carefully curated training labels.
* Investigate hierarchical classification:

  * broad issue family
  * specific intent
* Calibrate classifier confidence on held-out manually labelled data.

### P2 — Improve reply generation

* Use an LLM to synthesize responses from retrieved historical evidence.
* Require the generated response to remain grounded in retrieved evidence.
* Add automated checks for unsupported claims.
* Evaluate reply quality separately from intent accuracy.

### P2 — Improve production safety

* Add stricter handling for account, payment, refund, and identity-sensitive actions.
* Prefer escalation when evidence is ambiguous.
* Log intent confidence, retrieval score, retrieval gap, and escalation reason for every decision.

---

# 12. Reproducibility

The repository contains scripts for:

```text
data preparation
thread reconstruction
intent discovery
golden-set creation
golden-set validation
intent classification
retrieval indexing
agent execution
automated evaluation
LLM-as-judge evaluation
annotation consistency check
```

The main evaluation artifacts are stored under:

```text
results/
├── metrics.json
├── examples.json
├── failure_analysis.md
├── evaluation_examples.jsonl
├── llm_judge_results.jsonl
├── llm_judge_metrics.json
└── human_agreement.json
```

The intended workflow is:

```bash
python scripts/prepare_data.py
python scripts/create_golden_set.py
python scripts/label_golden_set.py
python scripts/verify_golden_set.py

python scripts/run_agent.py
python scripts/evaluate.py

python scripts/verify_second_annotation.py
python -m src.evaluation.human_agreement
```

The final prototype prioritizes **reproducibility, groundedness, conservative escalation, and honest evaluation** over presenting an artificially high headline metric.
