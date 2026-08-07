"""Central configuration: paths, schema definitions and shared constants.

Keeping these in one module means the training script, the evaluation code and
the serving API all agree on column names, dtypes and artifact locations.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "employee_data.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
TARGET = "enrolled"
ID_COLUMN = "employee_id"

NUMERIC_FEATURES: list[str] = ["age", "salary", "tenure_years"]
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "marital_status",
    "employment_type",
    "region",
    "has_dependents",
]
RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns added by ``features.FeatureEngineer``. Listed here so a preprocessor
# can be constructed without having to fit the transformer first.
ENGINEERED_FEATURES: list[str] = ["log_salary", "tenure_ratio"]

# Domain plausibility bounds, used by the data-quality report and by the API's
# request validation. Values outside these ranges are suspicious, not fatal.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "age": (16, 100),
    "salary": (0, 1_000_000),
    "tenure_years": (0, 60),
}

# Categories observed in the training data. Used for the data-quality report
# and for documenting the API contract.
EXPECTED_CATEGORIES: dict[str, list[str]] = {
    "gender": ["Female", "Male", "Other"],
    "marital_status": ["Divorced", "Married", "Single", "Widowed"],
    "employment_type": ["Contract", "Full-time", "Part-time"],
    "region": ["Midwest", "Northeast", "South", "West"],
    "has_dependents": ["No", "Yes"],
}

# Minimum legal working age, used to derive the maximum plausible tenure.
MIN_WORKING_AGE = 18

# Columns used for slice-based (fairness / robustness) evaluation.
SLICE_FEATURES: list[str] = ["gender", "region", "employment_type"]

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SEARCH_ITERATIONS = 30

# Model selection is driven by a single primary metric; everything else is
# reported for context. ROC-AUC is threshold independent, which matters because
# the deployment threshold is a business decision made after training.
PRIMARY_METRIC = "roc_auc"

# When several models are statistically indistinguishable, prefer the simpler
# one. A candidate is "indistinguishable" if its mean CV score is within this
# many standard errors of the best score (the one-standard-error rule).
SELECTION_TOLERANCE_SE = 1.0

# Floor on the selection tolerance. This dataset has a deterministic target, so
# the best models score an identical 1.0 on every fold and the standard error
# collapses to exactly zero — at which point the one-standard-error rule would
# demand an exact tie and always pick the most complex model. The floor keeps
# the "prefer the simpler model" intent alive in that degenerate case. It is a
# judgement call: 0.0005 ROC-AUC is ~4 employees' worth of ranking on a 2,000
# row test set, which is not a difference worth extra model complexity.
SELECTION_MIN_DELTA = 0.0005

DEFAULT_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Business cost model
# --------------------------------------------------------------------------- #
# ASSUMPTION, not ground truth. The assignment gives no economics, so these are
# stated openly here and used to derive a cost-optimal decision threshold. They
# are the single place to change if the real numbers become available.
#
#   COST_FALSE_POSITIVE: we contact someone who was never going to enrol.
#                        Wasted outreach (agent time, mailing, incentive).
#   COST_FALSE_NEGATIVE: we skip someone who would have enrolled.
#                        Forgone first-year margin on the policy.
#
# Only the *ratio* (12:1 here) affects the optimal threshold.
COST_FALSE_POSITIVE = 25.0
COST_FALSE_NEGATIVE = 300.0
