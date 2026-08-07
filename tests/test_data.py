"""Tests for loading, validation, cleaning and splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, data


def test_load_raw_reads_expected_schema(sample_csv):
    df = data.load_raw(sample_csv)
    assert len(df) == 400
    assert set(config.RAW_FEATURES) <= set(df.columns)
    assert config.TARGET in df.columns


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        data.load_raw(tmp_path / "nope.csv")


def test_load_raw_missing_column_raises(sample_frame, tmp_path):
    path = tmp_path / "incomplete.csv"
    sample_frame.drop(columns=["salary"]).to_csv(path, index=False)
    with pytest.raises(data.DataValidationError, match="salary"):
        data.load_raw(path)


def test_load_raw_normalises_column_names(sample_frame, tmp_path):
    path = tmp_path / "messy_headers.csv"
    renamed = sample_frame.rename(columns={"age": " AGE ", "salary": "Salary"})
    renamed.to_csv(path, index=False)
    assert {"age", "salary"} <= set(data.load_raw(path).columns)


def test_quality_report_counts_implausible_tenure(sample_frame):
    # A 25-year-old cannot have 20 years of service.
    sample_frame.loc[0, ["age", "tenure_years"]] = [25, 20.0]
    report = data.build_quality_report(sample_frame)
    assert report.implausible_tenure >= 1


def test_quality_report_flags_out_of_range_and_unknown_categories(sample_frame):
    sample_frame.loc[0, "age"] = 150
    sample_frame.loc[1, "region"] = "Atlantis"
    report = data.build_quality_report(sample_frame)
    assert report.out_of_range["age"] == 1
    assert report.unexpected_categories["region"] == ["Atlantis"]


def test_quality_report_detects_duplicates(sample_frame):
    duplicated = pd.concat([sample_frame, sample_frame.iloc[[0]]], ignore_index=True)
    report = data.build_quality_report(duplicated)
    assert report.duplicate_records == 1


def test_quality_report_target_distribution_sums_to_one(sample_frame):
    report = data.build_quality_report(sample_frame)
    assert sum(report.target_distribution.values()) == pytest.approx(1.0, abs=1e-3)


def test_clean_strips_whitespace_and_drops_duplicates(sample_frame):
    sample_frame.loc[0, "region"] = "  West  "
    duplicated = pd.concat([sample_frame, sample_frame.iloc[[5]]], ignore_index=True)
    cleaned = data.clean(duplicated)

    assert cleaned.loc[0, "region"] == "West"
    assert len(cleaned) == len(sample_frame)


def test_clean_keeps_implausible_rows(sample_frame):
    """Implausible tenure is reported, never silently dropped or overwritten."""
    sample_frame.loc[0, ["age", "tenure_years"]] = [25, 20.0]
    cleaned = data.clean(sample_frame)
    assert len(cleaned) == len(sample_frame)
    assert cleaned.loc[0, "tenure_years"] == 20.0


def test_split_features_target_excludes_id(sample_frame):
    X, y = data.split_features_target(sample_frame)
    assert config.ID_COLUMN not in X.columns
    assert list(X.columns) == config.RAW_FEATURES
    assert y.name == config.TARGET


def test_stratified_split_preserves_class_balance(sample_xy):
    X, y = sample_xy
    X_train, X_test, y_train, y_test = data.stratified_split(X, y, test_size=0.25)

    assert len(X_train) + len(X_test) == len(X)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)
    # No row may appear on both sides of the split.
    assert not set(X_train.index) & set(X_test.index)


def test_load_dataset_end_to_end(sample_csv):
    X, y, report = data.load_dataset(sample_csv)
    assert list(X.columns) == config.RAW_FEATURES
    assert set(y.unique()) <= {0, 1}
    assert report.n_rows == 400
