# SECOM Yield Risk Prediction

Ranking semiconductor process measurements by how closely they resemble past
in-house test failures.

Built for the DataTalks.Club Machine Learning Zoomcamp midterm. The model is served
with FastAPI behind a small JSON API, and everything here can be reproduced from a
clean clone.

**Live demo:** <https://secom-yield-risk.onrender.com> — interactive API
documentation at <https://secom-yield-risk.onrender.com/docs>. The free instance
sleeps when idle, so the first request may take 30–60 seconds.

> **Scope.** This predicts the recorded *product test result* for a historical
> dataset. It does not predict equipment breakdowns, does not explain why a failure
> happened, and is not a production decision system. The score is an uncalibrated
> ranking signal for closer review, and the evidence for the signal is weak — see
> [Results on the held-out test set](#results-on-the-held-out-test-set).

---

## Problem

A semiconductor line records hundreds of sensor readings for every production
example. Only about 6.6% of examples then fail the in-house test. Finding the
failures matters, but the measurements are anonymous, noisy, incomplete, and far
outnumbered by passes.

The question this project asks: **do these process measurements rank failing
examples above passing ones, well enough to be useful for deciding what to review
first?**

That framing sets the metric. Accuracy is useless here — a model that always says
"pass" scores 93.4% and catches nothing. What matters is how well the model ranks
rows, and what it costs to flag them.

---

## Data

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) — 1,567 production
examples, 590 real-valued process measurements each, plus a timestamp and a
pass/fail label.

- **Negative class:** `-1` (passed) → encoded `0`; 1,463 examples (93.4%).
- **Positive class:** `1` (failed) → encoded `1`; **104 examples (6.6%)**.
- **Features:** 590 anonymous sensors (`0` … `589`). They have no interpretable
  names, so this project never claims to know what any sensor measures.
- **Timestamp:** loaded but deliberately **not used as a feature**. It encodes when
  a row was produced, which would leak temporal information and let a model learn
  drift instead of process signal.

### Getting the files

The raw files are not committed (`secom.data` alone is 54 MB). Download them once
into `data/` — see [`data/README.md`](data/README.md) for the URL, the expected
filenames, byte sizes and SHA-256 checksums. The notebook and tests both expect:

```text
data/secom.data          measurements, 1,567 x 590, `NaN` for missing
data/secom_labels.data   label and timestamp, 1,567 rows
data/secom.names         variable descriptions
```

### What the data looks like

- **Missingness:** 41,951 of 924,530 measurement cells are missing (4.54%). It is
  concentrated: 52 features are complete, 28 are over 50% missing, 4 are over 90%
  missing, and no feature is entirely empty. The median row has 24 missing values;
  the worst has 152.
- **Little variation:** 116 features are constant across the dataset and 6 more are
  near-constant. 112 of the constants contain only `0` or `NaN`.
- **Weak per-feature signal:** no single sensor separates the classes. The
  best-separated feature differs by 0.59 standard deviations between pass and fail
  groups, the next by 0.57, and their distributions still overlap heavily.

---

## Approach

### Splits (seed 42, stratified)

| Split | Rows | Failures | Failure rate |
|---|---:|---:|---:|
| Train | 1,002 | 66 | 6.59% |
| Validation | 251 | 17 | 6.77% |
| Test | 314 | 21 | 6.69% |

The validation set drives model comparison, threshold exploration and the final
selection. The test set is used exactly once, at the end. Stratification keeps the
failure rate aligned across splits, and no row appears in two splits.

### Preprocessing (inside the pipeline, fitted per fold)

```text
median imputation  ->  constant-column removal  ->  standardisation  ->  classifier
```

Every step is a scikit-learn `Pipeline` stage, so it is fitted on training rows only.
That matters for the variance filter in particular: it removes **122** columns from
the 1,002 training rows (more than the 116 constants in the full dataset), because a
few features happen to hold a single value inside those rows. Validation and test
data are transformed with the training medians, means and scales, never refitted.

In early chronological folds, 16 features had no observed value at all and were
dropped by the imputer, so those folds used 452 columns instead of 468.

---

## Model comparison

Five candidates, all scored on the 251-row validation set at the default 0.5 cut.
Failure is the positive class; "caught" counts failures the model flagged.

| Model | Caught / 17 | False alarms | Precision | Recall | F1 | Bal. acc. | ROC-AUC | Avg. precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (most frequent) | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | 0.068 |
| **Logistic regression** | **5** | **30** | **0.143** | **0.294** | **0.192** | **0.583** | 0.661 | **0.220** |
| Random forest | 0 | 0 | 0.000 | 0.000 | 0.000 | 0.500 | **0.746** | 0.158 |
| Decision tree | 2 | 12 | 0.143 | 0.118 | 0.129 | 0.533 | 0.533 | 0.077 |
| Hist gradient boosting | 0 | 2 | 0.000 | 0.000 | 0.000 | 0.496 | 0.665 | 0.155 |

Reading this honestly:

- **The dummy sets the floor.** Its average precision (0.068) *is* the failure rate,
  which is the level any real ranking must beat.
- **Only logistic regression produces usable flags at 0.5**, and even then roughly
  six of its seven flags are false alarms.
- **The forest ranks best (ROC-AUC 0.746) but flags nobody at 0.5**, because its
  highest validation probability is 0.48. Its ranking is real; its operating point is
  simply misplaced at that cut.
- **The single tree is the weakest real model** — depth 35, 52 leaves, average
  precision 0.077, barely above no-skill. It overfits 1,002 rows.

A threshold sweep (0.1–0.5) on all four real candidates confirms the cut is doing
work, not the model: the forest catches all 17 failures at 0.1 but raises 163 false
alarms out of 180 flags. Threshold choice cannot manufacture signal that is not there.

No hyperparameter search was run. Defaults were kept deliberately: with 17 validation
failures, a grid search would select settings that fit those 17 cases, and the
time-aware results below show how thin the underlying signal is. The comparison is
across model families and preprocessing variants instead.

---

## Time-aware validation

A random split flatters this dataset, because production rows drift over time. So the
`X_train` portion was re-split into four expanding chronological folds (200 validation
rows each, 6–15 failures per fold) and the candidates re-compared. Average precision
(AP) per fold:

| Approach | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean AP | Std |
|---|---:|---:|---:|---:|---:|---:|
| Validation failure rate | 0.075 | 0.030 | 0.035 | 0.060 | 0.050 | — |
| Constant baseline | 0.075 | 0.030 | 0.035 | 0.060 | 0.050 | 0.021 |
| **LR baseline** | 0.083 | 0.036 | 0.040 | 0.159 | **0.080** | 0.057 |
| LR + missing indicators | 0.087 | 0.052 | 0.051 | 0.139 | 0.082 | 0.041 |
| LR + correlation pruning | 0.081 | 0.040 | 0.039 | 0.185 | 0.086 | 0.069 |

- Every model beats the constant baseline in every fold, but the margin is tiny:
  0.080 against 0.050.
- **Fold 4 carries the apparent skill.** Excluding it, the LR baseline falls to 0.053
  against a 0.050 baseline, and the ordering of the three variants changes.
- **Correlation pruning was rejected**, despite the highest mean AP. Its rule required
  it to win in a majority of folds (it won 2 of 4) and to be no less stable than the
  plain pipeline (std 0.069 against 0.057). It failed both, and its advantage came
  almost entirely from one fold.

The plain logistic regression was therefore selected: it is the simplest option, and
the more complex ones do not survive a consistency check.

---

## Results on the held-out test set

Scored once, with the artifact that is shipped in this repository. That artifact was
fitted on the 1,253 train + validation rows (83 failures) after the model and the
selection rule were fixed.

| Metric | Value | Reference |
|---|---:|---|
| Average precision (PR-AUC) | **0.124** | 0.067 = flag nobody |
| ROC-AUC | **0.644** | 0.500 = no ranking ability |
| Balanced accuracy at 0.5 | 0.541 | 0.500 = coin flip |
| Accuracy at 0.5 | 0.844 | 0.933 = flag nobody |

Confusion matrix at 0.5 (314 rows, 21 failures): 261 true passes, 32 false alarms,
17 missed failures, 4 caught. Failure-class precision 0.111, recall 0.190.

Test threshold sweep — reported for interpretation, not selected from these results:

| Threshold | Flagged | Flagged % | Caught | Missed | False alarms | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 61 | 19.4 | 6 | 15 | 55 | 0.098 | 0.286 |
| 0.20 | 52 | 16.6 | 5 | 16 | 47 | 0.096 | 0.238 |
| 0.30 | 47 | 15.0 | 5 | 16 | 42 | 0.106 | 0.238 |
| 0.40 | 41 | 13.1 | 5 | 16 | 36 | 0.122 | 0.238 |
| 0.50 | 36 | 11.5 | 4 | 17 | 32 | 0.111 | 0.190 |

**The honest summary.** The score ranks rows better than chance — average precision
is about 1.9× the failure rate — and that is all. No threshold turns it into a usable
filter: at 0.1 the model catches 6 of 21 failures while producing 55 false alarms; at
0.5 it catches 4 and misses 17. Validation gave AP 0.220 on 17 failures and the
chronological folds gave a mean of 0.080, so the test figure of 0.124 sits inside the
range those two estimates implied.

Because the scores are uncalibrated and no operating point is defensible from 21
failures, **no decision threshold is stored with the model**. The service returns
rankings, and choosing a cut is left to whoever consumes them.

---

## Running the service

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# development: reload on changes under src/
python -m uvicorn src.app:app --port 8000 --reload --reload-dir src
```

On macOS or Linux, replace the activation line with `source .venv/bin/activate`.

`--reload-dir src` narrows watching to the service code. Without it, `--reload`
watches the whole repository and re-executing the notebook restarts the API.

For a production-style run, drop both flags:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t secom-risk .
docker run --rm -p 8000:8000 secom-risk
```

The image installs only `requirements.txt`, contains just `src/`, `models/` and the
artifact, runs as an unprivileged user, and honours `$PORT` when a platform sets it.
`.dockerignore` keeps the 54 MB data directory and the notebook out of the build
context.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Landing payload pointing at the other routes |
| `GET` | `/health` | Service status, model name, expected feature count |
| `POST` | `/predict` | Risk score per row of measurements |
| `GET` | [`/docs`](https://secom-yield-risk.onrender.com/docs) | Interactive OpenAPI documentation (live) |

The live `/docs` page is the quickest way to see the contract: expand `POST /predict`,
choose *Try it out*, and send a body. The request body needs a full row of 590 numbers,
so paste a row rather than typing one.

A request is a list of rows. Each row must contain **exactly 590 measurements in the
original column order**, with `null` for a missing reading. Nothing should be imputed
or scaled beforehand: the served pipeline does that with the values it learned during
training.

```powershell
$row  = ((Get-Content data\secom.data -TotalCount 1) -split '\s+')
$rec  = @($row | ForEach-Object { if ($_ -eq 'NaN') { $null } else { [double]$_ } })
$body = @{ records = @(, $rec) } | ConvertTo-Json -Depth 5 -Compress

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType 'application/json' -Body $body
```

```json
{"risk_scores": [5.444228934534924e-08]}
```

One score per row, in the order the rows were sent. Scores run from 0 to 1, are
**uncalibrated**, and are only meaningful relative to each other.

Invalid input fails before it reaches the model, with a 422 and a message naming the
problem: an empty `records` list, a row that is not 590 values long, a missing
`records` key, or malformed JSON.

```json
{"detail": "Row 0 has 589 measurements, but exactly 590 are required."}
```

A row of all `null`s is accepted and scored, because the pipeline imputes medians.

---

## Deploying

> **Hosted demo:** <https://secom-yield-risk.onrender.com>
>
> Deployed from commit `c55084d` on Render's free tier (Docker runtime, health check
> `/health`). `GET /`, `GET /health`, `GET /docs` and `POST /predict` are open and
> unauthenticated. The free instances sleep after roughly 15 minutes idle, so the
> first request after a quiet period can take 30–60 seconds; later requests answer in
> well under a second.
>
> The deployed service was checked against the committed artifact: for the first
> three rows of `secom.data` it returns the same scores to floating-point precision
> (two bit-identical, one differing in the last digit through BLAS summation order),
> and the same validation errors.

`render.yaml` describes a free-tier Docker web service with a health check on
`/health`. On a platform that builds from the repository, the steps are: connect the
repo, choose the Docker runtime, set the health check path to `/health`, and let the
platform set `PORT` — the container command already reads it.

---

## Reproducing from scratch

```powershell
git clone https://github.com/la-kshita/secom-yield-risk-prediction.git
cd secom-yield-risk-prediction

# 1. data: follow data/README.md, ending with data/secom.data present

# 2. environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 3. checks
pytest

# 4. retrain and rewrite the artifact (optional: the artifact is committed)
python -m nbconvert --to notebook --execute --inplace notebooks/train.ipynb
```

The notebook reproduces every number in this README. It is seeded throughout, and
re-running it rewrites `models/secom_logistic_regression_pipeline.joblib` with a
byte-identical file.

`pytest` runs 24 tests: the API contract, the artifact's contents, and the raw data's
schema. The data tests skip automatically when the files are not present, which is the
normal state of a fresh clone.

### Version note

`requirements.txt` pins `scikit-learn==1.9.1`, `pandas==3.1.1`, `numpy==2.5.4`,
`joblib==1.6.0`, `fastapi==0.162.6`, `uvicorn==0.54.0` and `pydantic==2.14.3` on
Python 3.13. These match the versions recorded inside the artifact, and a test fails
if the environment drifts from them.

---

## Limitations

- **Weak signal, stated plainly.** Average precision is 0.124 on the holdout against
  0.067 for flagging nobody. A chronological split gives 0.080 against 0.050, and
  dropping one favourable fold brings it to 0.053. This is a weak ranking aid, not a
  filter.
- **Tiny positive class.** 104 failures in total, 21 in the test set, 6–15 per
  chronological fold. One example changing place moves recall by several points.
- **Uncalibrated scores.** `class_weight="balanced"` means the outputs are not
  probabilities of failure. A score of 0.9 does not mean a 90% chance of failing; it
  means the row is ranked highly relative to others.
- **No stored threshold.** No cut is defensible from this evidence, so none is
  shipped. The API returns rankings, not decisions.
- **Historical, single line.** One dataset collected in 2008 from one process. No
  evidence this transfers to another line, another product, or today.
- **No causal claim.** A high score means a row resembles past failures, not that any
  sensor caused a failure. Acting on it means "review this", not "the process is
  broken".
- **No drift monitoring.** The service scores whatever it is given. The time-aware
  analysis suggests drift matters; nothing in the deployment detects it.
- **Unprotected endpoint.** The demo has no authentication, rate limiting or batch
  cap. Appropriate for a course demo, not for real data.
- **Accuracy is misleading here** and is reported only to be dismissed: flagging
  nobody scores 93.3% on the test set, and this model scores 84.4%.

---

## Repository layout

```text
README.md                     this file
data/README.md                how to download the raw data (not committed)
data/secom*.data              raw SECOM files (local only)
notebooks/train.ipynb         EDA, model comparison, threshold analysis,
                              time-aware validation, final holdout evaluation
src/app.py                    FastAPI service: /, /health, /predict
models/                       committed artifact, other artifacts ignored
tests/                        API contract, artifact contents, data schema
Dockerfile, .dockerignore     container image for the service
render.yaml                   deployment configuration
requirements.txt              pinned runtime dependencies
requirements-dev.txt          notebook, plotting and test dependencies
pyproject.toml                pytest configuration
```

---

## Data license and citation

SECOM is published by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This project uses the data under that licence and makes no claim of ownership.

> McCann, M. & Johnston, A. (2008). SECOM. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C54305>

The code in this repository is the author's own work, written for the DataTalks.Club
Machine Learning Zoomcamp midterm.
