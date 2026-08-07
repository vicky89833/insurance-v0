# Employee Insurance Enrollment Prediction

An end-to-end ML pipeline that predicts whether an employee will enroll in a voluntary insurance product, from census-style HR data.

> **Read this first.** The provided dataset's target turned out to be **fully deterministic** — `enrolled` is exactly *"at least 3 of {age > 30, salary > $60,000, employment_type == Full-time, has_dependents == Yes}"*, verified against all 10,000 rows with zero mismatches. The models therefore reach **1.000 ROC-AUC on held-out data**, which says something about the data generator rather than about the modelling. Detecting this is the most useful result in the project. Full analysis in **[report.md](report.md)**.

---

## Quickstart

```bash
make setup    # create .venv and install pinned dependencies
make train    # train, tune, evaluate, and write artifacts
make test     # run the test suite (76 tests)
```

Without `make`:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.train
.venv/bin/python -m pytest tests/ -q
```

Requires Python 3.11+. Training takes ~90 seconds on a laptop.

### All commands

| Command | What it does |
|---|---|
| `make setup` | Create the virtualenv and install `requirements.txt` |
| `make eda` | Data-quality report, generating-rule discovery, EDA figures |
| `make train` | Tune 6 candidates, select, evaluate on held-out data, persist artifacts |
| `make test` | Run the test suite |
| `make serve` | Start the prediction API on `http://127.0.0.1:8000` |
| `make all` | `eda` → `train` → `test` |
| `make clean` | Remove generated artifacts and caches |

---

## What the pipeline does

1. **Load and validate** (`src/data.py`) — schema check, data-quality report, minimal cleaning. Problems are *reported*, not silently patched.
2. **Preprocess** (`src/features.py`) — imputation, encoding and scaling as pipeline steps, so they are refitted inside every CV fold and cannot leak.
3. **Train and tune** (`src/train.py`, `src/models.py`) — six candidates, `RandomizedSearchCV` over 5-fold stratified CV, scored on ROC-AUC.
4. **Select** — one-standard-error rule: the *simplest* model within one SE of the best score wins, so complexity is never bought with noise.
5. **Evaluate** (`src/evaluate.py`) — held-out metrics, calibration, cost-based threshold selection, permutation importance, partial dependence, per-subgroup slices.
6. **Serve** (`src/api.py`) — FastAPI service loading the persisted pipeline.

### Results summary

| | |
|---|---|
| Selected model | Decision tree (depth 4, entropy, balanced weights) |
| Test ROC-AUC / PR-AUC | 1.0000 / 1.0000 |
| Test accuracy | 1.0000 (vs 61.7% majority-class baseline) |
| Signal features | `has_dependents`, `salary`, `employment_type`, `age` |
| Zero-importance features | `gender`, `region`, `marital_status`, `tenure_years` |

A depth-4 tree was chosen over a random forest that scored 1.0000: the 0.0003 AUC difference is far below what is worth paying for in interpretability and latency.

---

## Prediction API

```bash
make serve   # then open http://127.0.0.1:8000/docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness; reports whether a model is loaded |
| `GET /model/info` | Model card: what is deployed, and its known limitations |
| `POST /predict` | Score one employee |
| `POST /predict/batch` | Score up to 1,000 employees in one call |

Both prediction endpoints accept an optional `?threshold=` override.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 41, "salary": 72500, "tenure_years": 6.5,
       "gender": "Female", "marital_status": "Married",
       "employment_type": "Full-time", "region": "West", "has_dependents": "Yes"}'
```

```json
{
  "enrollment_probability": 1.0,
  "will_enroll": true,
  "threshold": 0.501,
  "model_name": "decision_tree"
}
```

Input is validated by Pydantic, so malformed requests get a precise `422` rather than a `500`. If no model has been trained yet, the service still starts, reports `degraded` on `/health`, and returns `503` from the prediction endpoints.

---

## Project structure

```
├── README.md                 # this file
├── report.md                 # findings, model rationale, results, next steps
├── docs/design.md            # design decisions made before implementation
├── requirements.txt          # pinned dependencies
├── Makefile                  # every command in this README
├── pyproject.toml            # pytest and lint configuration
├── data/
│   └── employee_data.csv     # 10,000 employee records
├── src/
│   ├── config.py             # schema, paths, constants — single source of truth
│   ├── data.py               # loading, validation, cleaning, splitting
│   ├── features.py           # feature engineering + preprocessing pipelines
│   ├── models.py             # candidate registry and search spaces
│   ├── train.py              # CLI: tuning, selection, evaluation, persistence
│   ├── evaluate.py           # metrics, threshold economics, plots
│   ├── eda.py                # exploratory analysis + generating-rule discovery
│   └── api.py                # FastAPI prediction service
├── tests/                    # 76 tests
└── artifacts/
    ├── figures/              # all report figures
    ├── metrics/              # metrics.json, eda_summary.json, sweeps
    └── models/               # persisted pipeline (gitignored — rebuild with `make train`)
```

`artifacts/metrics/` and `artifacts/figures/` are tracked in git on purpose: they are the evidence behind `report.md` and should be reviewable in a diff. The `.joblib` binary is not tracked — it is reproducible from a seeded run.

---

## Training options

```bash
python -m src.train --help
```

| Flag | Purpose |
|---|---|
| `--models logistic decision_tree` | Train a subset of candidates |
| `--n-iter 60` | Randomised-search candidates per model (default 30) |
| `--cv 10` | Cross-validation folds (default 5) |
| `--seed 7` | Random seed (default 42) |
| `--no-feature-engineering` | Ablation: train on raw columns only |
| `--no-plots` | Skip figure generation |

Runs are deterministic given a seed.

---

## Testing

```bash
make test
```

76 tests covering the data contract and quality checks, feature engineering edge cases (unseen categories, missing values, division-by-zero on `tenure_ratio`), pipeline structure and leakage prevention, the model-selection rule, threshold economics, and every API endpoint including failure modes.

---

## Notes and limitations

- **The dataset is synthetic and noiseless.** No threshold, coefficient or performance figure here transfers to real enrollment data. See [report.md §5](report.md) for what a production version would need.
- **The cost model is an assumption.** `COST_FALSE_POSITIVE = $25` and `COST_FALSE_NEGATIVE = $300` in `src/config.py` are stated placeholders that drive the recommended threshold. Replace them with measured campaign economics before relying on the operating point.
- **`gender` is used as a feature** because it was supplied. It carries zero measured importance here, and its use in insurance pricing or targeting would need legal review before any real deployment.
