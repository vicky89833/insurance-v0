"""Shared pytest fixtures.

The synthetic frame is generated from the same rule that governs the real
dataset (``at least 3 of 4`` conditions — see ``src/eda.py``). Tests therefore
exercise data with genuine structure while staying fast and self-contained: no
test depends on ``data/employee_data.csv`` being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def apply_generating_rule(df: pd.DataFrame) -> pd.Series:
    """The recovered rule: enrolled iff at least 3 of 4 conditions hold."""
    conditions = (
        (df["age"] > 30).astype(int)
        + (df["salary"] > 60_000).astype(int)
        + (df["employment_type"] == "Full-time").astype(int)
        + (df["has_dependents"] == "Yes").astype(int)
    )
    return (conditions >= 3).astype(int)


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small dataset with the same schema and structure as the real one."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 400

    df = pd.DataFrame(
        {
            config.ID_COLUMN: range(10_001, 10_001 + n),
            "age": rng.integers(22, 65, n),
            "gender": rng.choice(["Male", "Female", "Other"], n),
            "marital_status": rng.choice(["Married", "Single", "Divorced", "Widowed"], n),
            "salary": rng.normal(65_000, 15_000, n).round(2).clip(1_000),
            "employment_type": rng.choice(["Full-time", "Part-time", "Contract"], n),
            "region": rng.choice(["West", "South", "Midwest", "Northeast"], n),
            "has_dependents": rng.choice(["Yes", "No"], n),
            "tenure_years": rng.uniform(0, 20, n).round(1),
        }
    )
    df[config.TARGET] = apply_generating_rule(df)
    return df


@pytest.fixture
def sample_xy(sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and target, id column dropped."""
    return sample_frame[config.RAW_FEATURES].copy(), sample_frame[config.TARGET].copy()


@pytest.fixture
def sample_csv(sample_frame: pd.DataFrame, tmp_path):
    """The sample frame written to a temporary CSV."""
    path = tmp_path / "employee_data.csv"
    sample_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def valid_employee() -> dict:
    """A well-formed API request body."""
    return {
        "age": 41,
        "salary": 72_500.0,
        "tenure_years": 6.5,
        "gender": "Female",
        "marital_status": "Married",
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": "Yes",
    }
