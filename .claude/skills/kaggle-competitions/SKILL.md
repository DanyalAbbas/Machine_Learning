---
name: kaggle-competitions
description: Use when working on a Kaggle competition - read the problem statement, rules, evaluation metric and public notebooks, then build, validate and submit a leaderboard entry. Triggers on a kaggle.com/competitions link, or when the user asks to compete, enter or submit to a Kaggle competition, asks for a baseline or starter notebook, or mentions Kaggle leaderboard, submissions, or competition code.
---

# Kaggle Competition Playbook

A repeatable procedure for taking a Kaggle competition from zero to a scored
submission. Optimised for the case where the user wants a strong result, not a
toy.

## Contents

1. [Tools](#1-tools)
2. [Step 1 — Read the problem](#2-step-1--read-the-problem)
3. [Step 2 — Read the rules](#3-step-2--read-the-rules)
4. [Step 3 — Pin down the metric](#4-step-3--pin-down-the-metric)
5. [Step 4 — Recon the public code](#5-step-4--recon-the-public-code)
6. [Step 5 — Get the real leaderboard](#6-step-5--get-the-real-leaderboard)
7. [Step 6 — Build a baseline](#7-step-6--build-a-baseline)
8. [Step 7 — Build a validation harness FIRST](#8-step-7--build-a-validation-harness-first)
9. [Step 8 — Iterate](#9-step-8--iterate)
10. [Step 9 — Submit](#10-step-9--submit)
11. [Hard-won lessons](#hard-won-lessons)

---

## 1. Tools

### MCP tools (read-only; work without extra setup)

| tool | use |
|---|---|
| `kaggle_get_competition` | metadata, metric, deadline, team count, **your rank** |
| `kaggle_list_competition_pages` | Overview, Data, Evaluation, **Rules** — read all of them |
| `kaggle_get_competition_data_files_summary` | file inventory and column types |
| `kaggle_list_competition_topics` | forum index |
| `kaggle_get_forum_topic` | one thread + comments (often the highest-signal source) |
| `kaggle_get_competition_leaderboard` | top of board (may 403 if not enrolled) |
| `kaggle_save_notebook` | create/update a notebook |
| `kaggle_create_code_competition_submission` | submit a notebook to a code comp |

**MCP gotcha:** read-only calls work, but any authenticated call may return
`Unauthenticated` or `Unauthorized`. When that happens, do **not** burn time
debugging it — fall back to the CLI immediately (below).

### Kaggle CLI (the reliable path)

```bash
pip install --user --break-system-packages kaggle     # PEP-668 environments
export PATH="$HOME/.local/bin:$PATH"
# auth: token lives in ~/.kaggle/access_token (used automatically)
```

```bash
# data
kaggle competitions download -c COMP -p data/
kaggle datasets download OWNER/SLUG -p data/ --unzip

# notebooks
kaggle kernels push -p STAGING_DIR          # uploads a version AND runs it
kaggle kernels status OWNER/SLUG
kaggle kernels output OWNER/SLUG -p out/    # downloads submission + log
kaggle kernels logs  OWNER/SLUG
kaggle kernels pull  OWNER/SLUG -p dir/

# leaderboard  (--download gets the FULL csv; --show only paginates)
kaggle competitions leaderboard -c COMP --download -p lb/
kaggle competitions list -s "COMP"          # includes your rank

# submissions
kaggle competitions submit -c COMP -k OWNER/SLUG -f submission.parquet -v VERSION -m "msg"
kaggle competitions submissions -c COMP
```

### `kernel-metadata.json` (required by `kernels push`)

```json
{
  "id": "OWNER/SLUG",
  "title": "Display Title",
  "code_file": "notebook.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_internet": false,
  "enable_gpu": true,
  "accelerator": "gpu-t4",
  "dataset_sources": ["owner/dataset"],
  "kernel_sources": ["owner/upstream-notebook"],
  "competition_sources": ["comp-slug"],
  "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"]
}
```

Notes that cost real time if missed:
- `id` and `title` must slugify identically, or the CLI warns and reassigns the slug.
- Model sources **must** include the version: `owner/model/framework/instance/1`.
- `accelerator` values: `gpu-t4`, `gpu-t4-x2`, `gpu-p100`, etc. Check what the
  competition's existing top notebooks actually use before assuming.

### Reading another notebook's full input manifest (for forking)

```bash
TOKEN=$(cat ~/.kaggle/access_token)
curl -sL -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/pull/OWNER/SLUG" \
| python3 -c "
import json, sys
d = json.load(sys.stdin)                 # top level: 'metadata' + 'blob'
md = d['metadata']
print('datasets:', md['datasetDataSources'])
print('kernels :', md['kernelDataSources'])
print('models  :', md['modelDataSources'])
open('nb.ipynb', 'w').write(d['blob']['source'])
"
```

This is the only reliable way to reproduce someone else's notebook — the attached
datasets are not in the `.ipynb`.

---

## 2. Step 1 — Read the problem

Read the Overview and Data Description pages in full before writing anything.

Capture explicitly, in writing:

- **Task shape** — regression / classification / ranking / generation.
- **Row and column counts** for train and test, and the test size if it differs.
- **Grouping structure.** Look for an id that partitions the data into groups
  (e.g. many rows per `prompt_id`, `patient_id`, `session_id`). **This single
  fact determines your validation design** — see Step 7.
- **Leakage candidates** — timestamp columns, target-derived aggregates,
  duplicate rows, future information.
- **Class/label balance** and whether the metric cares (see Step 3).
- **Entity overlap** between train and test. Do the same entities appear in both?
  Is the test set a later time window, a different site, a different generator?

Then **load the data and actually look at it.** Do not trust the description:

```python
df.shape, df.dtypes, df.isna().sum()
df[col].nunique()
df.groupby(group_col).size().describe()
# always: does one group contain exactly one row per class?
```
That last line is how you discover balanced-by-construction designs, which
often unlock a large gain — see Step 7.

---

## 3. Step 2 — Read the rules

Read the Rules page end to end. Most competitions enforce limits by omission, so
check specifically:

| thing to check | why it matters |
|---|---|
| Submission format and **exact filename** | code comps usually require `submission.parquet` in the working dir |
| Daily submission cap | typically 5/day; budgets are scarce |
| How many submissions count for the final score | often 2 — you cannot tune 50 and keep the best |
| Whether it is **kernels/notebooks only** | no local `.csv` upload |
| Runtime / hardware limits | GPU hours are the real constraint |
| **Internet on/off** | often must be off; pre-attach models as inputs |
| External data rules | is generated-from-API data allowed? under what conditions? |
| Prohibition on hand-labelling eval data | rules usually forbid *human* labelling of val/test |
| Team size / merge deadline | |

Then **read the forum.** The top thread and the "how to get started" thread
frequently contain the answers that matter more than the rules text — cost
thresholds, clarifications, known pitfalls. Check for unanswered questions too:
if nobody has asked, you may be first, and asking gets a citable answer.

---

## 4. Step 3 — Pin down the metric

Write the metric down and implement it by hand before modelling anything.

- **Which metric?** accuracy / AUC / F1 / **balanced accuracy** / MAP@k / RMSE.
- **How is it computed exactly?** With what averaging, class weighting, and
  tie-breaking?
- **Is it computed globally or per group?** (Mean-over-groups changes everything.)
- **What does it reward that accuracy does not?**

Two consequences worth internalising:

**If the metric is class-balanced or macro-averaged, plain accuracy is a
misleading proxy.** Optimise the metric itself. Under balanced accuracy a rare
class with 5 true examples that absorbs 2,000 predictions scores ~0.

**If the metric rewards calibration or marginals**, transductive post-processing
(Sinkhorn / IPF toward a target class distribution) is often worth more than
better features.

Build a scorer early and use it on every OOF prediction:

```python
from sklearn.metrics import balanced_accuracy_score
# extend for your metric; verify against a hand-worked example
```

---

## 4.5 Step 3b — Read the data-description page for the submission schema

Code competitions usually specify the exact output contract. Copy it into an
assertion block at the end of your notebook so a schema failure is impossible:

```python
assert list(sub.columns) == ["id", "target"]
assert len(sub) == len(test) and sub.id.is_unique
assert sub.id.tolist() == test.id.tolist()
assert not sub.isna().any().any()
assert sub.target.isin(ALLOWED_LABELS).all()
sub.to_parquet("submission.parquet", index=False)
assert pd.read_parquet("submission.parquet").equals(sub)   # round-trip check
```

---

## 5. Step 4 — Recon the public code

**This step is where most of the free score is.** A Featured competition's top
notebooks routinely implement 80% of the achievable score.

```bash
kaggle kernels list --competition COMP --sort-by voteCount   # via CLI or MCP search
```

For each promising notebook, pull the **full source** and read it:

```bash
kaggle kernels pull OWNER/SLUG -p /tmp/nb/
```

Read for:
- the **feature set** — which signal does the winning model actually key on?
- the **model class** and its regularisation
- the **blending weights**, and whether they were fitted out-of-fold or hardcoded
- **what the author says their CV is**, and whether it matches the LB
- the **version history** in the markdown header — it often narrates the whole
  improvement path, including *removed* ideas and why

Then extract the ideas into a short list with an expected-value judgement on
each. Prefer things that are: cheap, proven by someone else, and targeted at a
measured weakness.

**Explicitly check what top authors removed and why.** Removed ideas are
free knowledge — they usually say "scored +0.001 locally, forks lost 0.003–0.005
on the LB", which tells you not to spend a day on it.

Also note the **lineage**: if a stack is public and forks of it are scoring well,
that is the field's consensus, and building on it is normal practice. Attribute
openly.

---

## 6. Step 5 — Get the real leaderboard

```bash
kaggle competitions leaderboard -c COMP --download -p lb/
```

Read the **CSV**, not the rendered top-20. Then compute:

```python
d = pd.read_csv('lb/name-that-llm.zip')
d.Score.describe(percentiles=[.5,.75,.9,.95])
d[d.TeamMemberUserNames.astype(str).str.contains('your_handle', na=False)]
```

You need: total teams, **median**, **p90/p95** (the real medal line), the leader,
and your own rank. The rendered page is often a partial or stale snapshot — in
one real case it showed 192 teams and a top score of 0.512 while the CSV showed
1,577 teams and 0.622 an hour later.

Re-pull periodically. Fields move fast in the first week of a hot competition,
and your target moves with them.

---

## 7. Step 6 — Build a baseline

Build the simplest thing that scores, and get it **submitted early** to establish
a real leaderboard anchor. Do not spend a week on a baseline.

Design rules:

- **Mirror the submission contract exactly** from day one.
- **Keep the notebook self-contained** where practical: fewer downloads, fewer
  moving parts, fewer ways to fail at scoring time.
- **Run it locally first.** A cheap CPU smoke test that exercises the whole path
  (including writing and re-reading `submission.parquet`) catches more bugs than
  reading the code.
- Fold-ensemble or refit-on-all, whichever the model needs; for linear models on
  small data, averaging the fold models usually wins.

**Source of truth pattern.** Keep the logic in a plain `.py`, generate the
`.ipynb` from it, and smoke-test the `.py`. The notebook that runs on Kaggle then
cannot drift from what you tested locally. (Percent-format cell markers + a small
builder script is enough; no jupytext needed.)

---

## 8. Step 7 — Build a validation harness FIRST

**This is the single highest-value step, and it is the one people skip.**

Your validation must simulate the *test* condition, not the train condition.
The most common failure in Kaggle is a method that looks excellent on CV and
does nothing on the leaderboard.

### 8a. Match the split structure first

If the test set uses **unseen groups**, use `GroupKFold` on the group column:

```python
folds = list(GroupKFold(5).split(X, y, groups))
for f, v in folds:
    assert not (set(groups[f]) & set(groups[v]))   # assert, don't assume
```

A random `train_test_split` here leaks topic/entity identity and inflates the
score. Expect a large drop once you fix it — that drop was always there.

### 8b. Then validate **per regime**, never as one number

If the test set's *shape* differs from train's — e.g. train has 18 rows per
prompt but test has 2–3 — then a single CV number is meaningless for any method
that depends on that shape. Build a harness:

1. Simulate the test shape by sub-sampling the train set to the test's
   rows-per-group (k = 1, 2, 3, 5, 8, 18, …).
2. Rebuild **any test-time structure** (clustering, normalisation, nearest
   neighbours) *inside that subsample*, exactly as it would be built on the test
   set.
3. Report the metric per k, averaged over several random draws.

Real example — a balanced-assignment rule measured on balanced train data:

| k (rows per group) | CV gain |
|---|---|
| 1 | −0.0013 |
| 2 | **+0.0027** |
| 3 | **+0.0031** |
| 8 | +0.0205 |
| 18 | **+0.0519** |

It looked like +0.075 on plain CV and delivered +0.003 on the leaderboard. The
single number was measuring k=18 — the only k the train split can produce. The
per-regime harness found that in minutes and for free.

### 8c. QA any test-time structure you rely on

Clustering, normalisation, and nearest-neighbour logic can be validated directly
against ground truth on train:

```python
same_c = clusters[:, None] == clusters[None, :]
same_g = groups[:, None] == groups[None, :]
off = ~np.eye(len(clusters), dtype=bool)
precision = (same_c & same_g & off).sum() / max((same_c & off).sum(), 1)
recall    = (same_c & same_g & off).sum() / max((same_g & off).sum(), 1)
```

**Precision is usually the number that decides whether a rule is sound.** And QA
the two hypotheses separately — clustering may work fine while the *premise* you
built on top of it does not hold. That distinction saved a lot of confusion.

### 8d. Rules of thumb for measurement

- Prefer **OOF predictions** over refits — one artefact, many questions answered.
- **Save OOF probabilities and test probabilities to disk.** Stacking, blending,
  calibration, and error analysis all reuse them. Never recompute.
- Keep a **control** in every experiment (e.g. shuffle-within-group) to prove a
  gain comes from the mechanism and not from a bug.
- When a test-time method fails, report the **mechanism**, not just the number.
  "Gain scales with k and collapses at k=2" is worth far more than "it didn't work".

---

## 9. Step 8 — Iterate

Order of expected value, and roughly what each buys:

1. **Reproduce a strong public baseline.** Usually the fastest path to a
   respectable score. Attributing is normal practice.
2. **Fix what the leader is leaving on the table.** Look at their code for
   asymmetry — a family covered by a specialist while a comparable one is not is
   a concrete, cheap gap.
3. **Error decomposition.** Build the confusion matrix and ask:
   - what share is cross-group vs within-group?
   - which specific classes/pairs dominate?
   - **Symmetric mutual confusion between two classes is the signature of
     genuine underdetermination** — not undertraining. Fix it with data, not
     regularisation.
   This one table drives every subsequent decision.
4. **Blend in log space**, not on raw probabilities, and **fit the blend weights
   out-of-fold** with a tie-break toward sane defaults (a near-uniform member
   makes many weight vectors equivalent — break ties deliberately).
5. **More data beats more modelling** when the labels are thin. Check the rules
   for external/generated data, count what you actually need per class, and
   price it before committing.
6. **Transductive extras** — self-training, Sinkhorn balancing, centred twins.
   Legitimate (no test labels are used) and often effective, but each MUST pass
   the Step 7 harness. Soft/partial-strength variants degrade gracefully; hard
   constraints do not, and will silently force wrong answers when the premise
   fails on test.

### Cost discipline

- Validate on CPU locally as far as possible. Offload the expensive part.
- **Verify a pushed notebook reproduces before modifying it.** A GPU run is ~30
  min; find out you broke something in 30 min, not in 3.
- Keep a **cheap, submission-ready fallback** ready at all times so you always
  have something valid to submit.
- Know the **daily cap** and treat the first 2–3 submissions as information
  gathering, not attempts.

---

## 10. Step 9 — Submit

```bash
kaggle kernels push -p STAGE                        # creates a new version AND runs it
kaggle kernels status OWNER/SLUG                    # poll
kaggle kernels output OWNER/SLUG -p out/            # grab the log + submission
kaggle competitions submit -c COMP -k OWNER/SLUG -f submission.parquet -v VERSION -m "why"
```

Discipline:

- **Read the log before submitting.** The log tells you the row count, the
  cluster-size distribution, whether each stage fired, and the prediction class
  distribution. It often answers the open question before the leaderboard does.
- **Never submit a version whose only purpose is a controlled experiment**
  unless you have spare budget. Use the harness for that.
- **Use the description field as a lab notebook** — record the CV number and the
  mechanism so you can compare submissions later.
- **Make the notebook public** (`"is_private": false`) and credit what you built
  on. Public code sharing is required by most competitions' rules anyway.
- Remember `KAGGLE_IS_COMPETITION_RERUN` — set during scoring. Use it to skip
  expensive diagnostics at scoring time while keeping them for interactive runs.
- **Re-pull the leaderboard CSV** after each submission and record your rank.

---

## Hard-won lessons

**Validation design dominates everything.** The largest single error in Kaggle
is validating on a split that does not resemble the test set. Two distinct
failures that look identical from the LB:

- *Split mismatch* — random split where the test uses unseen groups. Fix with
  `GroupKFold`.
- *Shape mismatch* — same grouping, but the test set has a different
  rows-per-group distribution. **Only a per-regime harness catches this.**

**A transductive trick validated on structurally uniform train data tells you
nothing about a test set with a different structure.** It won't be rejected, and
it will not error. It will simply never fire. Suspect this whenever a method
"does nothing" rather than "does badly".

**Verify a reimplementation reproduces the leader's number before you build on
it.** An invisible bug (wrong row order, unsorted joins, a mis-aligned index)
produces a chance-level score with no exception raised. Cheap insurance:

```python
# re-run a few rows in a fresh batch and require agreement with what was stored
assert np.allclose(stored_rows, fresh_forward(same_rows), atol=...)
```

**Prefer a `.py` source of truth over hand-editing `.ipynb`.** It makes the
notebook testable locally and stops the submitted artefact from drifting from
what you validated.

**Optimise the actual metric, and never with plain accuracy as a stand-in** when
the metric is balanced/macro-averaged.

**Read the version history of top notebooks.** It is a free, honest record of
what worked, what didn't, and what the author removed after finding it didn't
transfer.

**The field's ceiling is set by data volume more often than by architecture.**
Count rows per class early. "150 rows per class" explains a surprising number of
apparently irreducible confusions — and points at the only fix that matters.

**Budget submissions like money.** They cap per day, usually count only a couple
toward the final score, and every one of them is an experiment you pay for.