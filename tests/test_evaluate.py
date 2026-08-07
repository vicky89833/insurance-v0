"""Tests for metrics and threshold economics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, evaluate


@pytest.fixture
def perfect_scores():
    """A perfectly separated problem: scores 0.1 for negatives, 0.9 for positives."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_proba = np.where(y_true == 1, 0.9, 0.1)
    return y_true, y_proba


def test_metrics_on_a_perfect_classifier(perfect_scores):
    y_true, y_proba = perfect_scores
    metrics = evaluate.compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["cost_per_employee"] == 0.0


def test_metrics_on_a_random_classifier():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 2_000)
    metrics = evaluate.compute_metrics(y_true, rng.random(2_000), threshold=0.5)
    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.05)


def test_confusion_counts_sum_to_sample_size(perfect_scores):
    y_true, y_proba = perfect_scores
    m = evaluate.compute_metrics(y_true, y_proba)
    total = m["true_positives"] + m["true_negatives"] + m["false_positives"] + m["false_negatives"]
    assert total == len(y_true)


def test_threshold_changes_the_precision_recall_balance():
    """Raising the threshold must not increase recall."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 1_000)
    y_proba = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 1_000), 0, 1)

    low = evaluate.compute_metrics(y_true, y_proba, threshold=0.2)
    high = evaluate.compute_metrics(y_true, y_proba, threshold=0.8)
    assert low["recall"] >= high["recall"]


def test_expected_cost_weights_false_negatives_more_heavily():
    """The stated cost model says a missed enrolment hurts more than wasted outreach."""
    y_true = np.array([0, 0, 1, 1])
    contact_everyone = np.array([0.9, 0.9, 0.9, 0.9])  # 2 false positives
    contact_no_one = np.array([0.1, 0.1, 0.1, 0.1])    # 2 false negatives

    cost_fp = evaluate.expected_cost(y_true, contact_everyone, 0.5)
    cost_fn = evaluate.expected_cost(y_true, contact_no_one, 0.5)

    assert cost_fp == 2 * config.COST_FALSE_POSITIVE
    assert cost_fn == 2 * config.COST_FALSE_NEGATIVE
    assert cost_fn > cost_fp


def test_optimal_threshold_minimises_cost_over_the_sweep():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 1_000)
    y_proba = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 1_000), 0, 1)

    threshold, sweep = evaluate.find_optimal_threshold(y_true, y_proba)

    assert 0.0 <= threshold <= 1.0
    best_cost = evaluate.expected_cost(y_true, y_proba, threshold)
    assert best_cost == pytest.approx(sweep["total_cost"].min())


def test_asymmetric_costs_push_the_threshold_below_a_half():
    """With FN costlier than FP, the model should contact more people, not fewer."""
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 2_000)
    y_proba = np.clip(y_true * 0.4 + rng.normal(0.3, 0.25, 2_000), 0, 1)

    threshold, _ = evaluate.find_optimal_threshold(
        y_true, y_proba, cost_fp=1.0, cost_fn=20.0
    )
    assert threshold < 0.5


def test_optimal_threshold_picks_the_middle_of_a_tied_plateau(perfect_scores):
    """Perfect separation makes every cut in the gap equal; take the midpoint."""
    y_true, y_proba = perfect_scores
    threshold, _ = evaluate.find_optimal_threshold(y_true, y_proba)
    assert 0.1 < threshold < 0.9


def test_slice_metrics_cover_every_level(sample_xy):
    X, y = sample_xy
    rng = np.random.default_rng(4)
    y_proba = np.clip(y * 0.5 + rng.normal(0.25, 0.15, len(y)), 0, 1)

    slices = evaluate.slice_metrics(X, y, y_proba, threshold=0.5, columns=["region"])

    assert set(slices["value"]) == set(X["region"].unique())
    assert slices["n"].sum() == len(X)


def test_slice_metrics_handle_single_class_groups():
    """A slice with one class has no defined ROC-AUC; report NaN, do not crash."""
    X = pd.DataFrame({"region": ["West"] * 5 + ["South"] * 5})
    y = pd.Series([1] * 5 + [0, 1, 0, 1, 0])
    y_proba = np.linspace(0.1, 0.9, 10)

    slices = evaluate.slice_metrics(X, y, y_proba, threshold=0.5, columns=["region"])
    west = slices.loc[slices["value"] == "West", "roc_auc"].iloc[0]
    assert np.isnan(west)


def test_figures_are_written_to_disk(perfect_scores, tmp_path):
    y_true, y_proba = perfect_scores
    roc_path = evaluate.plot_roc_pr(y_true, y_proba, out_dir=tmp_path)
    calibration_path = evaluate.plot_calibration(y_true, y_proba, out_dir=tmp_path)

    assert roc_path.exists() and roc_path.stat().st_size > 0
    assert calibration_path.exists() and calibration_path.stat().st_size > 0
