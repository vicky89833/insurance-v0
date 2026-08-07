"""Tests for feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import FeatureEngineer, build_preprocessor, get_feature_names


def test_engineer_adds_expected_columns(sample_xy):
    X, _ = sample_xy
    out = FeatureEngineer().fit_transform(X)
    assert set(config.ENGINEERED_FEATURES) <= set(out.columns)


def test_engineer_can_be_disabled(sample_xy):
    X, _ = sample_xy
    out = FeatureEngineer(add_features=False).fit_transform(X)
    assert not set(config.ENGINEERED_FEATURES) & set(out.columns)


def test_log_salary_is_monotonic_in_salary(sample_xy):
    X, _ = sample_xy
    out = FeatureEngineer().fit_transform(X)
    assert (out.sort_values("salary")["log_salary"].diff().dropna() >= 0).all()


def test_tenure_ratio_exceeds_one_for_impossible_history():
    """The engineered ratio is what makes the known data defect visible."""
    frame = pd.DataFrame(
        [{"age": 25, "salary": 50_000.0, "tenure_years": 20.0,
          "gender": "Male", "marital_status": "Single",
          "employment_type": "Full-time", "region": "West", "has_dependents": "No"}]
    )
    out = FeatureEngineer().fit_transform(frame)
    # 20 years of service by age 25 means 20 / (25 - 18) > 1.
    assert out.loc[0, "tenure_ratio"] == pytest.approx(20 / 7)
    assert out.loc[0, "tenure_ratio"] > 1.0


def test_tenure_ratio_never_divides_by_zero():
    """Age at or below the minimum working age must not produce inf/NaN."""
    frame = pd.DataFrame(
        [{"age": 18, "salary": 30_000.0, "tenure_years": 1.0,
          "gender": "Male", "marital_status": "Single",
          "employment_type": "Contract", "region": "South", "has_dependents": "No"}]
    )
    out = FeatureEngineer().fit_transform(frame)
    assert np.isfinite(out.loc[0, "tenure_ratio"])


def test_numeric_columns_are_cast_to_float(sample_xy):
    """Integer `age` breaks partial dependence, so it is cast up front."""
    X, _ = sample_xy
    out = FeatureEngineer().fit_transform(X)
    for column in config.NUMERIC_FEATURES:
        assert out[column].dtype == np.float64


@pytest.mark.parametrize("kind", ["linear", "linear_binned", "tree"])
def test_preprocessor_output_is_finite_and_dense(sample_xy, kind):
    X, _ = sample_xy
    matrix = build_preprocessor(kind).fit_transform(X)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == len(X)
    assert np.isfinite(matrix).all()


def test_preprocessor_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        build_preprocessor("magic")


def test_unseen_category_does_not_raise(sample_xy):
    """A region absent from training must score, not 500 the API."""
    X, _ = sample_xy
    preprocessor = build_preprocessor("tree").fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["region"] = "Atlantis"
    matrix = preprocessor.transform(unseen)

    assert matrix.shape[1] == preprocessor.transform(X.iloc[[0]]).shape[1]
    assert np.isfinite(matrix).all()


def test_missing_values_are_imputed(sample_xy):
    X, _ = sample_xy
    preprocessor = build_preprocessor("tree").fit(X)

    with_gaps = X.iloc[[0]].copy()
    with_gaps["salary"] = np.nan
    with_gaps["region"] = None

    assert np.isfinite(preprocessor.transform(with_gaps)).all()


def test_feature_names_match_matrix_width(sample_xy):
    X, _ = sample_xy
    preprocessor = build_preprocessor("tree").fit(X)
    assert len(get_feature_names(preprocessor)) == preprocessor.transform(X).shape[1]


def test_transform_is_stateless_across_row_counts(sample_xy):
    """Scoring one row must give the same result as scoring it in a batch."""
    X, _ = sample_xy
    preprocessor = build_preprocessor("linear").fit(X)

    batch = preprocessor.transform(X.iloc[:5])
    single = preprocessor.transform(X.iloc[[0]])
    np.testing.assert_allclose(batch[0], single[0])
