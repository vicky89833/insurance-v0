"""Tests for the model registry, the selection rule and the training run."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src import config, train
from src.models import ModelSpec, build_pipeline, build_registry


@pytest.fixture
def registry():
    return build_registry()


def test_registry_contains_baseline_and_challengers(registry):
    assert "dummy" in registry
    assert len(registry) >= 5
    assert all(isinstance(spec, ModelSpec) for spec in registry.values())


def test_every_spec_documents_its_rationale(registry):
    """A candidate with no stated reason to exist should not be in the report."""
    for spec in registry.values():
        assert spec.rationale.strip()


def test_complexity_ranks_are_unique(registry):
    """The selection tie-break needs a strict ordering to be deterministic."""
    ranks = [spec.complexity_rank for spec in registry.values()]
    assert len(ranks) == len(set(ranks))


@pytest.mark.parametrize("name", ["dummy", "logistic", "decision_tree", "hist_gradient_boosting"])
def test_pipeline_fits_and_predicts_probabilities(sample_xy, registry, name):
    X, y = sample_xy
    pipeline = build_pipeline(registry[name]).fit(X, y)

    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_preprocesses_before_classifying(sample_xy, registry):
    """Preprocessing must live inside the pipeline, not be applied beforehand.

    This is what makes cross-validation honest: the imputer, scaler and encoder
    are refitted on each training fold, so no validation-fold statistics leak
    into them.
    """
    pipeline = build_pipeline(registry["logistic"])
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "classifier"]


def test_cross_validation_runs_without_leaking(sample_xy, registry):
    X, y = sample_xy
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(build_pipeline(registry["decision_tree"]), X, y,
                             cv=cv, scoring="roc_auc")
    assert len(scores) == 3
    assert scores.min() > 0.5


def test_search_space_keys_address_real_pipeline_params(sample_xy, registry):
    """Guards against typos in the parameter grids, which search reports as errors."""
    pipeline = build_pipeline(registry["logistic_binned"])
    valid = set(pipeline.get_params())
    assert set(registry["logistic_binned"].param_distributions) <= valid


def _result(name: str, mean: float, std: float, rank: int) -> train.SearchResult:
    spec = ModelSpec(name=name, estimator=None, preprocessor_kind="tree",
                     complexity_rank=rank, rationale="test")
    return train.SearchResult(
        spec=spec, estimator=None, cv_mean=mean, cv_std=std,
        cv_scores=np.full(5, mean), best_params={}, fit_seconds=0.1,
    )


def test_selection_prefers_simpler_model_within_tolerance():
    """The one-standard-error rule should not buy complexity with noise."""
    results = [
        _result("simple", 0.9997, 0.0003, rank=1),
        _result("complex", 1.0000, 0.0000, rank=5),
    ]
    assert train.select_best(results).spec.name == "simple"


def test_selection_takes_the_better_model_when_the_gap_is_real():
    results = [
        _result("simple", 0.80, 0.01, rank=1),
        _result("complex", 0.95, 0.01, rank=5),
    ]
    assert train.select_best(results).spec.name == "complex"


def test_selection_never_returns_the_dummy_baseline():
    """On a degenerate dataset the dummy could otherwise win on simplicity."""
    results = [
        _result("dummy", 0.50, 0.0, rank=0),
        _result("logistic", 0.50, 0.0, rank=1),
    ]
    assert train.select_best(results).spec.name == "logistic"


def test_selection_requires_candidates():
    with pytest.raises(ValueError, match="No candidate models"):
        train.select_best([_result("dummy", 0.5, 0.0, rank=0)])


def test_leaderboard_is_sorted_by_score():
    leaderboard = train.build_leaderboard(
        [_result("a", 0.80, 0.01, 1), _result("b", 0.95, 0.01, 2)]
    )
    assert list(leaderboard["model"]) == ["b", "a"]


def test_training_run_writes_all_artifacts(sample_csv, tmp_path, monkeypatch):
    """End-to-end smoke test: a short run must produce a loadable model."""
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "MODEL_CARD_PATH", tmp_path / "models" / "model_card.json")

    args = train.parse_args(
        [
            "--data", str(sample_csv),
            "--model-path", str(tmp_path / "models" / "model.joblib"),
            "--models", "dummy", "decision_tree",
            "--n-iter", "2", "--cv", "3", "--no-plots",
        ]
    )
    payload = train.run(args)

    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "metrics" / "metrics.json").exists()
    assert payload["selected_model"]["name"] == "decision_tree"
    assert 0.0 <= payload["test_metrics"]["at_default_threshold"]["roc_auc"] <= 1.0

    # The metrics file must be valid JSON, not just present.
    saved = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    assert saved["data_quality"]["n_rows"] == 400


def test_no_feature_engineering_flag_is_respected(sample_csv, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_CARD_PATH", tmp_path / "models" / "model_card.json")

    args = train.parse_args(
        [
            "--data", str(sample_csv),
            "--model-path", str(tmp_path / "models" / "model.joblib"),
            "--models", "decision_tree",
            "--n-iter", "2", "--cv", "3", "--no-plots", "--no-feature-engineering",
        ]
    )
    payload = train.run(args)
    assert payload["run"]["feature_engineering"] is False


def test_unknown_model_name_exits_with_a_message(sample_csv, tmp_path):
    args = train.parse_args(
        [
            "--data", str(sample_csv),
            "--model-path", str(tmp_path / "model.joblib"),
            "--models", "not_a_model", "--no-plots",
        ]
    )
    with pytest.raises(SystemExit, match="Unknown model"):
        train.run(args)
