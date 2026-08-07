"""Data loading, validation and splitting.

The raw file is census-style employee data. This module is the only place that
reads the CSV directly; everything downstream works with validated frames.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the input data cannot be used for training."""


@dataclass
class DataQualityReport:
    """Structured summary of the checks run against the raw frame."""

    n_rows: int
    n_columns: int
    missing_values: dict[str, int]
    duplicate_ids: int
    duplicate_records: int
    target_distribution: dict[str, float]
    out_of_range: dict[str, int] = field(default_factory=dict)
    implausible_tenure: int = 0
    unexpected_categories: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(path: Path | str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and assert that the expected schema is present.

    Args:
        path: Location of the employee CSV.

    Returns:
        The raw dataframe, with column names normalised to lowercase.

    Raises:
        FileNotFoundError: If the CSV is missing.
        DataValidationError: If required columns are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See the README for download instructions."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = set(config.RAW_FEATURES) | {config.TARGET}
    missing -= set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    logger.info("Loaded %d rows x %d columns from %s", len(df), df.shape[1], path)
    return df


def build_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """Run data-quality checks and return them as a structured report.

    The checks are deliberately non-destructive: they describe the data so that
    cleaning decisions can be made explicitly and documented, rather than being
    buried in a chain of silent transformations.
    """
    out_of_range: dict[str, int] = {}
    for column, (low, high) in config.VALID_RANGES.items():
        if column in df.columns:
            n_bad = int((~df[column].between(low, high)).sum())
            if n_bad:
                out_of_range[column] = n_bad

    # Tenure cannot exceed the number of years a person could have worked.
    max_plausible_tenure = (df["age"] - config.MIN_WORKING_AGE).clip(lower=0)
    implausible_tenure = int((df["tenure_years"] > max_plausible_tenure).sum())

    unexpected: dict[str, list[str]] = {}
    for column, allowed in config.EXPECTED_CATEGORIES.items():
        if column in df.columns:
            seen = set(df[column].dropna().astype(str).unique()) - set(allowed)
            if seen:
                unexpected[column] = sorted(seen)

    target_share = df[config.TARGET].value_counts(normalize=True)

    return DataQualityReport(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        missing_values={c: int(n) for c, n in df.isna().sum().items() if n > 0},
        duplicate_ids=(
            int(df[config.ID_COLUMN].duplicated().sum())
            if config.ID_COLUMN in df.columns
            else 0
        ),
        duplicate_records=int(
            df.drop(columns=[config.ID_COLUMN], errors="ignore").duplicated().sum()
        ),
        target_distribution={
            str(k): round(float(v), 4) for k, v in target_share.items()
        },
        out_of_range=out_of_range,
        implausible_tenure=implausible_tenure,
        unexpected_categories=unexpected,
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal, defensible cleaning steps.

    Two operations only, both safe to repeat at inference time and neither
    using information from the target:

    1. Drop exact duplicate records (ignoring the surrogate id).
    2. Strip whitespace from categorical values, so that ``"West "`` and
       ``"West"`` do not become two categories.

    Implausible-but-possible values (e.g. tenure exceeding working life) are
    *reported*, not dropped or overwritten: there is no ground truth to correct
    them to, and ``features.FeatureEngineer`` exposes the anomaly to the model
    as ``tenure_ratio`` instead. See ``report.md``.
    """
    subset = [c for c in df.columns if c != config.ID_COLUMN]
    cleaned = df.drop_duplicates(subset=subset).copy()

    for column in config.CATEGORICAL_FEATURES:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()

    n_dropped = len(df) - len(cleaned)
    if n_dropped:
        logger.info("Dropped %d duplicate records during cleaning", n_dropped)
    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the model matrix from the target, dropping the surrogate id.

    ``employee_id`` is excluded on purpose: it is a row identifier with no
    predictive meaning, and including it would let the model memorise rows.
    """
    X = df[config.RAW_FEATURES].copy()
    y = df[config.TARGET].astype(int)
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified hold-out split, so both sides keep the same class balance."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_dataset(
    path: Path | str = config.RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series, DataQualityReport]:
    """Convenience entry point: load, report, clean, and split into X/y."""
    raw = load_raw(path)
    report = build_quality_report(raw)
    X, y = split_features_target(clean(raw))
    return X, y, report
