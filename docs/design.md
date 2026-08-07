# Design decisions

Written before implementation, amended once during it. Kept in the repo because the *reasoning* behind a pipeline is usually harder to recover later than the code.

---

## 1. Problem framing

Binary classification on ~10,000 census-style employee records, target `enrolled`.

The business use is not "who is enrolled" but **who to target with outreach**. Two consequences:

- Model selection is scored on a **ranking** metric (ROC-AUC), which is threshold independent.
- The **decision threshold is a separate, business-owned parameter**, tuned after training against an explicit cost model.

Accuracy is deliberately demoted. At a 61.7% base rate, "predict everyone enrolls" scores 61.7% — so accuracy is reported only alongside the dummy baseline that contextualises it.

## 2. Splitting and leakage

- 20% stratified hold-out, touched **once**, at the end.
- 5-fold stratified CV on the remaining 80% for tuning.
- All preprocessing lives **inside** the `Pipeline`. Fitting an imputer or scaler before splitting is the classic silent leak; putting them in the pipeline makes it structurally impossible. `tests/test_train.py::test_pipeline_preprocesses_before_classifying` pins this.
- `employee_id` is dropped — a surrogate key is an invitation to memorise rows.

## 3. Preprocessing

Three numeric treatments, selected per model family:

| Kind | Numeric arm | Used by |
|---|---|---|
| `linear` | median impute → standardise | Logistic regression |
| `linear_binned` | median impute → 8-bin quantile one-hot | Binned logistic |
| `tree` | median impute only | Trees, forests, boosting |

Trees are invariant to monotonic rescaling, so standardising for them would cost time and change nothing.

Categoricals: constant-fill (`"Missing"` as its own level, since missingness is informative) → one-hot with `handle_unknown="ignore"`. The `ignore` is a **serving requirement**: an unseen `region` arriving at the API must produce an all-zero block, not a 500.

**Why `linear_binned` exists.** It is not there to win. It answers one question — when the linear baseline trails the trees, is that a *capacity* limit or a *representation* limit? Same estimator, different features, so the difference isolates the cause. It earned its place: 0.9691 → 0.9991.

## 4. Feature engineering — deliberately minimal

Only two additions, each with a reason:

- `log_salary` — salary is right-skewed; helps linear models, harmless to trees.
- `tenure_ratio = tenure_years / max(age − 18, 1)` — encodes the known data defect (385 rows with impossible working histories) as a feature instead of discarding rows.

**Rejected, with reasons:**

| Rejected | Why |
|---|---|
| SMOTE / resampling | 62/38 is mild; resampling distorts the calibration the cost analysis depends on. `class_weight` was in the search space instead. |
| Target encoding | Every categorical has ≤ 4 levels. One-hot is simpler and has no leakage surface. |
| Dropping the noise features | Let regularisation and permutation importance demonstrate they are noise, rather than assuming it. |
| Hand-coding `age > 30`, `salary > 60000` | These *are* the answer (see §8). Encoding them would leak the analysis into the features and prove nothing. Better to let trees find the cuts and show the recovery in partial-dependence plots. |

Feature engineering is gated behind `--no-feature-engineering` so the contribution is measured, not assumed. Measured result: +0.0026 AUC for plain logistic, ~0 for everything else. Reported honestly in `report.md §2.4`.

## 5. Model candidates

Six, ordered simplest to most flexible: dummy → logistic → binned logistic → decision tree → random forest → histogram gradient boosting. Each carries a `rationale` string in the registry; `tests/test_train.py` asserts none is blank, because a candidate with no stated reason to exist should not be in the report.

Tuning: `RandomizedSearchCV`, 30 candidates per model, 5-fold stratified, ROC-AUC. Random rather than grid search — at this budget it covers the space better per unit of compute.

**Rejected:** XGBoost/LightGBM. On 10,000 rows with 8 low-cardinality features they add a dependency without plausibly beating `HistGradientBoostingClassifier`.

## 6. Selection rule

**One-standard-error rule**: among candidates within 1 SE of the best CV score, take the one with the lowest `complexity_rank`. Complexity should have to *earn* its place against noise, and a difference smaller than fold-to-fold variance is not evidence.

The dummy baseline is excluded from selection — on a degenerate dataset it could otherwise win on simplicity.

**Amendment made during implementation.** The top models score identically on every fold, so their standard error is exactly 0 and the rule collapses to demanding a perfect tie — which would always crown the most complex model. Tolerance is therefore floored at `SELECTION_MIN_DELTA = 0.0005` ROC-AUC. This constant is a judgement call, named in `config.py` rather than buried, and it is the one number in this project I would most expect a reviewer to push back on.

## 7. Evaluation

- **Ranking:** ROC-AUC and PR-AUC.
- **Calibration:** Brier score and a reliability curve. If scores feed outreach budgeting, the probabilities must mean something.
- **Threshold:** cost-based, from stated assumptions ($25 wasted outreach vs $300 forgone margin — only the 12:1 ratio matters). Reported at both 0.5 and the tuned point.
- **Explainability:** permutation importance, *not* impurity importance, which is biased toward continuous and high-cardinality features. Plus partial dependence for the continuous drivers.
- **Robustness:** metrics sliced by `gender`, `region`, `employment_type`.

Ties in the cost sweep are resolved to the **midpoint of the tied plateau**, not the first tied value — under perfect separation every cut in the score gap is equally optimal, and the midpoint sits furthest from either edge.

## 8. The finding that changed the plan

EDA was supposed to be routine. It was not.

An unconstrained decision tree fitted on the whole dataset reaches **100% training accuracy at depth 5 with 13 leaves**. That means no label noise and no irreducible error — with real behavioural data, effectively impossible. A rule search then recovered the generator exactly, on all 10,000 rows:

```
enrolled = 1  ⟺  at least 3 of 4:  age > 30 · salary > 60,000
                                    employment_type == "Full-time" · has_dependents == "Yes"
```

**What changed as a result:**

- `discover_generating_rule` became a permanent part of `src/eda.py`, not a one-off script. It is the check that distinguishes "my model is excellent" from "my data is synthetic", and it is cheap enough to run always.
- The decision tree's search space was widened (`min_samples_leaf` down to 1, depth to 8). Over-regularising would have hidden the fact that a small tree fits the rule exactly.
- `report.md` leads with the finding rather than with the metrics. A 1.000 AUC presented without that context would be misleading, whatever the accompanying caveats.
- The recommendation changed: if this data were real, the honest advice is to **ship the rule as four lines of business logic** — no ML required to evaluate a 3-of-4 threshold rule.

## 9. Repo layout

A Python package with a CLI, not notebooks. The assignment scores code quality and documentation as two of five criteria, and a notebook carries stale cell outputs that cannot be trusted or diffed. `make eda` regenerates every figure and a JSON summary from source, so every number in the report is checkable.

`artifacts/metrics/` and `artifacts/figures/` are tracked in git (they are the report's evidence); the `.joblib` model is not (it is reproducible from a seeded run).

## 10. Scope

**Built:** full pipeline, hyperparameter tuning, evaluation suite, FastAPI service, 76 tests.

**Deliberately not built:** MLflow / Weights & Biases tracking. A grader running `make train` should not need a tracking server, and `artifacts/metrics/metrics.json` already captures the full run record — parameters, leaderboard, metrics, data-quality report — in a diffable format. The integration point is obvious if it is ever wanted.
