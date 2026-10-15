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

| step | approach | reported CV |
|---|---|---|
| v2 | frozen **Qwen3-1.7B-Base** hidden states, mean-pool layers 17/20, logistic head, blend with v1 | ~0.43 |
| v3 | **Qwen3-8B-Base**, early layers 4/6/8/22 (early layers beat late ones at 4B/8B) | ~0.46 |
| v4 | **fine-tuned encoder** (`dmis-lab/UltraBERT-380M`), 5-fold softmax average | ~0.41 |
| v5 | prompt-group-**centred** embeddings — recover topic clusters by TF-IDF linkage clustering, subtract the cluster mean so the head reads style not topic | add-on |
| v6 | stack v2–v5 + per-class log-score offsets fit by class-balanced log loss | target |

Notes that matter:

* **GPU goes to the embedding branch**, not to CV. The TF-IDF baseline is cheap and
  stable on CPU; embeddings need a T4 and the 8B model needs two.
* Early layers win on style attribution — the task is surface statistics, not semantics.
* Test passages cluster by prompt (~2–3 per prompt, from the leaderboard). Centring
  embeddings inside those clusters removes topic signal and is worth a real gain.
* Blend in **log space**, not on raw probabilities.
* Max 5 submissions/day, max 2 final submissions. Budget accordingly.

### Biggest lever left

Only 2,699 training rows for 18 classes. The forum's top thread is exactly this:
generating more data by calling the 18 APIs is allowed under the rules (external data is
fine if "reasonably accessible to all"), and Kaggle hands out $100/month of tokens per
benchmark. A fine-tuned encoder on 100k+ passages per model would beat every frozen-feature
trick on this list. That is the expensive path — treat it as a team project, not a solo one.