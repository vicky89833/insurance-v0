"""Feature engineering and preprocessing.

Everything here is expressed as scikit-learn transformers so that it can live
*inside* the model pipeline. That is what makes cross-validation honest: the
imputer, the scaler and the encoder are fitted on each training fold only, so
no information from the validation fold leaks into the fitted parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from src import config

# Preprocessing variants. "linear_binned" exists to answer a specific question:
# when a linear model trails the trees, is that a capacity limit or merely a
# feature-representation limit? Binning the numerics answers it directly.
PREPROCESSOR_KINDS = ("linear", "linear_binned", "tree")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add two domain-motivated features and normalise categorical dtypes.

    Added columns:
        ``log_salary``
            ``log1p(salary)``. Salary is right-skewed; the log makes the
            relationship closer to linear for the regression models. Tree
            models are invariant to it, so it is harmless there.
        ``tenure_ratio``
            ``tenure_years / max(age - MIN_WORKING_AGE, 1)``, the share of a
            plausible working life spent at the company. Values above 1.0 are
            impossible, so this also surfaces the known data-quality defect
            (see ``data.build_quality_report``) as a signal the model can use
            rather than as rows to discard.

    The transformer is stateless: ``fit`` only records the input schema, which
    keeps it safe to apply to a single row at inference time.
    """

    def __init__(self, add_features: bool = True) -> None:
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        X = pd.DataFrame(X).copy()

        # Cast numerics to float up front. `age` arrives as int64, and several
        # scikit-learn tools (partial dependence in particular) reject integer
        # columns because grid values would be silently rounded.
        for column in config.NUMERIC_FEATURES:
            if column in X.columns:
                X[column] = X[column].astype(float)

        if self.add_features:
            X["log_salary"] = np.log1p(X["salary"].clip(lower=0))
            working_years = (X["age"] - config.MIN_WORKING_AGE).clip(lower=1)
            X["tenure_ratio"] = X["tenure_years"] / working_years

        # Normalise categoricals to plain object dtype with numpy NaN. pandas
        # nullable dtypes (StringDtype/pd.NA) do not round-trip cleanly through
        # SimpleImputer, and this is the one place to fix that for good.
        for column in config.CATEGORICAL_FEATURES:
            if column in X.columns:
                X[column] = X[column].astype(object).where(X[column].notna(), np.nan)

        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(input_features if input_features is not None else self.feature_names_in_)
        if self.add_features:
            names += config.ENGINEERED_FEATURES
        return np.asarray(names, dtype=object)


def _numeric_features(add_features: bool) -> list[str]:
    """Numeric columns the preprocessor should expect, engineering included."""
    extra = config.ENGINEERED_FEATURES if add_features else []
    return config.NUMERIC_FEATURES + extra


def _numeric_branch(kind: str) -> Pipeline:
    """Build the numeric arm of the ColumnTransformer for a given model family.

    Args:
        kind: One of ``PREPROCESSOR_KINDS``.

    - ``linear``: impute, then standardise. Scaling matters for regularised
      logistic regression, where the penalty is applied on the coefficient
      scale.
    - ``linear_binned``: impute, then quantile-bin into one-hot indicators.
      This lets a linear model express step functions.
    - ``tree``: impute only. Trees are invariant to monotonic rescaling, so
      standardising would add cost without changing a single split.
    """
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]

    if kind == "linear":
        steps.append(("scaler", StandardScaler()))
    elif kind == "linear_binned":
        steps.append(
            (
                "binner",
                KBinsDiscretizer(n_bins=8, encode="onehot-dense", strategy="quantile"),
            )
        )
    elif kind != "tree":
        raise ValueError(f"Unknown preprocessor kind: {kind!r}")

    return Pipeline(steps)


def build_preprocessor(kind: str = "tree", add_features: bool = True) -> Pipeline:
    """Assemble the full preprocessing pipeline.

    Args:
        kind: Which numeric treatment to use; see ``_numeric_branch``.
        add_features: Whether to run ``FeatureEngineer`` first.

    Returns:
        An unfitted ``Pipeline`` mapping the raw feature frame to a dense
        numeric matrix.
    """
    if kind not in PREPROCESSOR_KINDS:
        raise ValueError(f"kind must be one of {PREPROCESSOR_KINDS}, got {kind!r}")

    categorical_branch = Pipeline(
        [
            # A missing category is itself informative, so it is encoded as its
            # own level rather than imputed to the mode.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            # handle_unknown="ignore" is a serving requirement: an unseen region
            # arriving at the API must yield an all-zero block, not an exception.
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(kind), _numeric_features(add_features)),
            ("categorical", categorical_branch, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        [
            ("engineer", FeatureEngineer(add_features=add_features)),
            ("columns", column_transformer),
        ]
    )


def get_feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Read output column names from a fitted preprocessor, for plots/reports."""
    return list(fitted_preprocessor.named_steps["columns"].get_feature_names_out())
