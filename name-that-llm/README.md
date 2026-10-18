# Name That LLM — competition workbench

Identify which of 18 frontier LLMs wrote each passage. Metric: **balanced accuracy**
(mean per-class recall), so rare classes matter as much as common ones.

* **Local CV (prompt-grouped 5-fold)**: `0.3771 ± 0.0076`
* **Public leaderboard**: `0.360` — rank 123/192 on day one
* Official starter notebook: `0.2997` (reproduced locally)
* Leaderboard: https://www.kaggle.com/competitions/name-that-llm/code
* Notebook: https://www.kaggle.com/code/danyalabbas1/ntl-baseline-v1-stylometry-tf-idf

### Where the field is

| rank | score |
|---|---|
| 1 | 0.512 |
| 2–10 | 0.501 – 0.475 |
| 16–18 | 0.473 |
| **us (v1)** | **0.360** |

CV runs ~1.7 pts above the public LB. That gap is the signature of the task: test prompts
are unseen, so the honest number to tune against is grouped CV, not the LB.

---

## What the task actually is

| | |
|---|---|
| train | 2,699 passages = 150 prompts × 18 models, **one response per model per prompt** |
| test | ~4,000 passages, prompts **not** in train, **class counts not equal** |
| classes | 18 models across OpenAI / Google / Anthropic / Qwen / ZAI / Moonshot |
| lengths | 40–250 words, sampled randomly |
| submission | code competition — notebook writes `submission.parquet` |

Two consequences that drive every design decision:

1. **Validate on held-out prompts, never a random split.** All 18 answers to one prompt
   stay in the same fold. A random split leaks topic vocabulary and inflates the score.
2. **Provider family is nearly free; the model within the family is the hard part.**
   Our v1 gets 69.4% provider-family accuracy but only 37.7% exact-model accuracy.
   Per-class OOF recall: `claude-fable-5` 0.08, `gpt-5.6-sol` 0.10, `glm-5.3` 0.17.

### Punctuation is the strongest cheap signal

Mean counts per 1,000 characters on the training set:

| model | em dash `—` | curly `’` | `:` |
|---|---|---|---|
| `glm-5.3` | 2.36 | 0.00 | 1.17 |
| `claude-opus-5` | 1.97 | 0.00 | 1.29 |
| `gpt-5.5` | 0.28 | 1.08 | 1.30 |
| `gemini-3.6-flash` | 0.00 | 0.30 | **0.25** |
| `qwen3.8-max` | 0.08 | 0.77 | 0.75 |

Anthropic/GLM/Kimi use em dashes, Gemini essentially never does, OpenAI uses curly
quotes while Claude/GLM/Qwen use straight ones. Formatting was stripped from the corpus
(no newlines, no markdown), so what remains is pure punctuation habit.

---

## v1 baseline — `baseline_v1_stylometry.py`

CPU-only, no downloads, ~10 min on Kaggle CPU.

* features: case-sensitive char 1–5 grams + word 1–2 grams TF-IDF + explicit
  stylometry block (punctuation rates per 1k chars, lengths, casing, type-token ratio)
* model: `LinearSVC`, decision scores averaged over the 5 prompt-grouped folds
* every output is asserted against the submission contract before writing

Reproduce:

```bash
python3 build_notebook.py                  # .py -> notebook/*.ipynb
python3 smoke_test.py                      # fast local end-to-end check
kaggle kernels push -p <staging-dir>       # then commit + submit from the UI
```

---

## Roadmap (all validated with prompt-grouped CV)

Published notebooks already report these numbers; they are the ladder from 0.38 upward.

| approach | author | public LB |
|---|---|---|
| starter TF-IDF + LogisticRegression | Kaggle staff | ~0.30 |
| frozen Qwen3-1.7B layers 17/20 + SVM, arithmetic blend, class offsets | hanifnoerrofiq | 0.430 |
| frozen Qwen3-8B layers 4/6/8/22, plain + topic-centred heads, log blend | qasminumber | 0.475 |
| frozen Qwen3-1.7B, out-of-fold-fitted blend weights | lucasboesen | 0.472 |
| fine-tuned UltraBERT-380M, 5-fold softmax average | pjmathematician | 0.468 |
| v1 (ours) — stylometry + char/word TF-IDF LinearSVC | — | 0.360 |
| v1.5 (ours) — v1 + cluster assignment | — | 0.359 |
| **v2 (ours) — Qwen3-8B centred head + SVM + assignment** | — | **0.461** |

Read from the leaderboard CSV; ranks were 59, 9, 18, 21 and 123 of 192 on day one.

What the top two have in common, and where they stop:

* frozen **Qwen** hidden states, mean-pooled, one logistic head per layer
* **early layers** win at 4B/8B (4/6/8/22) but late layers win at 1.7B (17/20)
* blend in **log space**, not on probabilities
* topic clustering (TF-IDF, average linkage, threshold 0.9, ARI 0.955) used only to
  centre embeddings before the head

Neither uses the cluster structure to assign labels — which is where we looked, and where
the idea turned out not to transfer (see below).

### Biggest lever left

Only 2,699 training rows for 18 classes. The forum's top thread is exactly this:
generating more data by calling the 18 APIs is allowed under the rules (external data is
fine if "reasonably accessible to all"), and Kaggle hands out $100/month of tokens per
benchmark. A fine-tuned encoder on 100k+ passages per model would beat every frozen-feature
trick on this list. That is the expensive path — treat it as a team project, not a solo one.

---

## v2 — `v2_qwen_clusters.py`

Qwen3-8B-Base embeddings plus the cluster-aware assignment described below.
Kaggle: GPU T4 x2, internet off, inputs = competition data + `qwen-lm/qwen-3/transformers/8b-base/1`.
**Public LB 0.461.**

Three branches, blended in log space with weights fitted on OOF:

| branch | what it sees | OOF |
|---|---|---|
| `svm` | case-sensitive char 1-5 + word 1-2 TF-IDF + stylometry → sigmoid-calibrated LinearSVC | 0.3575 |
| `emb_plain` | mean-pooled Qwen3-8B layers 4/6/8/22 → StandardScaler + logistic head per layer | 0.4452 |
| `emb_centred` | the same heads on cluster-centred embeddings, so the classifier reads within-cluster contrast | 0.5623 |
| blend | svm 0.3 / emb_centred 0.7 (fitted on OOF) | 0.5708 |

The centred heads are trained on random chunks of the cluster sizes **actually observed in
the test set**, so the head meets the same group geometry at prediction time. Treat that
OOF number as optimistic for the reason given below.

### Where v2 still leaves money on the table

`emb_plain` (0.4452 OOF) does **not** depend on cluster structure at all, and it is almost
certainly underused: the blend search drove its weight to 0.0 because on train the centred
head is inflated by train-only clustering quality. All 37 layers are already extracted and
stored, so the obvious next moves are cheap in embedding terms:

1. **Sweep layers on `emb_plain` alone.** Layers 4/6/8/22 were inherited from another
   competitor, not selected. All 37 are in the array.
2. **Re-tune blend weights without the centred branch**, or shrink it, to stop a
   train-favourable component dominating.
3. Fine-tuned encoder as a decorrelated 4th branch (0.468 LB solo for pjmathematician).

### The idea that looked best and turned out not to transfer: cluster-aware assignment

Train is 150 prompts x 18 models, so **every prompt is a permutation of the 18 labels**.
Recovering those groups by clustering lets you replace 18 independent argmaxes with one
`linear_sum_assignment` over negated log-probabilities — the maximum-probability bijection,
which uses each label exactly once.

On train this is a large, clean win. Measured on v1's out-of-fold probabilities, so the
comparison is exact — identical predictions, only the decision rule changes:

| decision rule | balanced accuracy |
|---|---|
| independent argmax | 0.3771 |
| Hungarian on clusters of size exactly 18 | 0.4508 |
| **Hungarian on size 2-8 and exactly 18** | **0.4523** |
| Hungarian on every cluster of size 2-18 | 0.4508 |
| soft Sinkhorn, strength 0.5 | 0.4134 |
| control: shuffle labels inside clusters | 0.0670 |

**On the leaderboard it was worth nothing.** v1 vs v1.5 differ only in this rule:

| | grouped CV | public LB |
|---|---|---|
| v1 argmax | 0.3771 | 0.360 |
| v1.5 + assignment | 0.4523 (**+0.075**) | 0.359 (**-0.001**) |

The prior is valid on train and simply does not hold on test. The train set is exactly
18-per-prompt by construction; the hidden test set is not, and its passages cluster into
groups of ~2-3, not 18 — so the constraint never has enough members to fire. No harm done
(it is a no-op when clusters are small), but the gain was an artifact of the train split's
structure.

**Lesson: a transductive trick validated only on a structurally uniform train set tells
you nothing about a test set with a different structure.** The train/test structural
mismatch has to be measured, not assumed — and the only way to measure it was to spend a
submission.

The same caveat applies to `emb_centred`, whose CV (0.5623) depends on the same clustering
quality. It clearly helped on test (0.360 -> 0.461) but by much less than CV implied,
because clustering recovers train prompts at ARI 0.955 and does not do as well on test.