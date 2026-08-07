"""Evaluation: metrics, threshold economics, diagnostic plots, explainability.

Two ideas drive this module:

1. **Ranking quality and the decision threshold are separate concerns.** The
   model produces a score; converting that score into "contact / do not
   contact" is a business decision with an explicit cost trade-off.
2. **Accuracy is the least useful number here.** With 61.7% of employees
   enrolled, predicting "everyone enrols" already scores 61.7%.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

logger = logging.getLogger(__name__)

PALETTE = {
    "primary": "#2b6cb0",
    "secondary": "#dd6b20",
    "muted": "#a0aec0",
    "positive": "#2f855a",
    "negative": "#c53030",
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the full metric suite at a given decision threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probability of the positive class.
        threshold: Score above which a prediction counts as positive.

    Returns:
        Threshold-free metrics (ROC-AUC, PR-AUC, Brier, log loss) alongside
        threshold-dependent ones (accuracy, precision, recall, F1).
    """
    y_true = np.asarray(y_true)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "cost_per_employee": float(
            (fp * config.COST_FALSE_POSITIVE + fn * config.COST_FALSE_NEGATIVE)
            / max(len(y_true), 1)
        ),
    }


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fp: float = config.COST_FALSE_POSITIVE,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
) -> float:
    """Total misclassification cost, in currency units, at one threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * cost_fp + fn * cost_fn)


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    cost_fp: float = config.COST_FALSE_POSITIVE,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    n_steps: int = 501,
) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds and return the cost-minimising one.

    For a perfectly calibrated model the optimum sits at
    ``cost_fp / (cost_fp + cost_fn)``; the empirical sweep is used instead
    because it does not assume calibration and it produces the curve that goes
    into the report.

    Returns:
        The best threshold and the full sweep as a dataframe.
    """
    y_true = np.asarray(y_true)
    thresholds = np.linspace(0.0, 1.0, n_steps)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "total_cost": float(fp * cost_fp + fn * cost_fn),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "contact_rate": float(y_pred.mean()),
            }
        )

    sweep = pd.DataFrame(rows)

    # Several thresholds often tie at the minimum cost — always so when the
    # classes are perfectly separated, where any cut inside the score gap is
    # equally good. Taking the midpoint of the tied plateau puts the operating
    # point as far as possible from either edge, instead of arbitrarily landing
    # on the lowest tied threshold.
    tied = sweep.loc[sweep["total_cost"] == sweep["total_cost"].min(), "threshold"]
    best_threshold = float(tied.median())
    theoretical = cost_fp / (cost_fp + cost_fn)
    logger.info(
        "Cost-optimal threshold: %.3f (theoretical for a calibrated model: %.3f)",
        best_threshold,
        theoretical,
    )
    return best_threshold, sweep


def slice_metrics(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-subgroup performance, as a robustness and fairness check.

    A model can look healthy overall while failing a subgroup. Reporting
    ROC-AUC and recall per slice makes that visible; a flat table is a real
    result and worth stating.
    """
    columns = columns or config.SLICE_FEATURES
    rows = []
    for column in columns:
        for value, index in X.groupby(column, observed=True).groups.items():
            mask = X.index.isin(index)
            y_slice = np.asarray(y_true)[mask]
            p_slice = y_proba[mask]
            # ROC-AUC is undefined when a slice contains a single class.
            single_class = len(np.unique(y_slice)) < 2
            rows.append(
                {
                    "feature": column,
                    "value": str(value),
                    "n": int(mask.sum()),
                    "positive_rate": float(y_slice.mean()),
                    "roc_auc": (
                        float("nan") if single_class else float(roc_auc_score(y_slice, p_slice))
                    ),
                    "recall": float(
                        recall_score(
                            y_slice, (p_slice >= threshold).astype(int), zero_division=0
                        )
                    ),
                    "precision": float(
                        precision_score(
                            y_slice, (p_slice >= threshold).astype(int), zero_division=0
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Write a figure to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


def plot_roc_pr(y_true, y_proba, out_dir: Path = config.FIGURES_DIR) -> Path:
    """ROC and precision-recall curves side by side."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0].plot(fpr, tpr, color=PALETTE["primary"], lw=2,
                 label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="Random")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title="ROC curve")
    axes[0].legend(loc="lower right", frameon=False)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision, color=PALETTE["secondary"], lw=2,
                 label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    axes[1].axhline(y_true.mean(), ls="--", color=PALETTE["muted"],
                    label=f"Baseline = {y_true.mean():.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="lower left", frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "roc_pr_curves.png")


def plot_calibration(y_true, y_proba, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Reliability diagram: do predicted probabilities mean what they say?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    axes[0].plot(prob_pred, prob_true, "o-", color=PALETTE["primary"], label="Model")
    axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["muted"], label="Perfect calibration")
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed frequency",
                title=f"Calibration (Brier = {brier_score_loss(y_true, y_proba):.4f})")
    axes[0].legend(frameon=False)

    axes[1].hist(y_proba, bins=40, color=PALETTE["primary"], alpha=0.8)
    axes[1].set(xlabel="Predicted probability", ylabel="Employees",
                title="Score distribution")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "calibration.png")


def plot_threshold_analysis(
    sweep: pd.DataFrame,
    best_threshold: float,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Operating-point trade-offs and the cost curve that picks the threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(sweep["threshold"], sweep["precision"], color=PALETTE["primary"], label="Precision")
    axes[0].plot(sweep["threshold"], sweep["recall"], color=PALETTE["secondary"], label="Recall")
    axes[0].plot(sweep["threshold"], sweep["f1"], color=PALETTE["positive"], label="F1")
    axes[0].axvline(best_threshold, ls="--", color=PALETTE["negative"],
                    label=f"Cost-optimal = {best_threshold:.3f}")
    axes[0].set(xlabel="Decision threshold", ylabel="Score",
                title="Operating-point trade-offs")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(sweep["threshold"], sweep["total_cost"], color=PALETTE["negative"], lw=2)
    axes[1].axvline(best_threshold, ls="--", color=PALETTE["muted"])
    axes[1].set(
        xlabel="Decision threshold",
        ylabel="Total cost on test set (currency units)",
        title=(
            f"Cost curve (FP = {config.COST_FALSE_POSITIVE:.0f}, "
            f"FN = {config.COST_FALSE_NEGATIVE:.0f})"
        ),
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "threshold_analysis.png")


def plot_confusion_matrices(
    y_true,
    y_proba,
    thresholds: dict[str, float],
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Confusion matrices at each named operating point, side by side."""
    y_true = np.asarray(y_true)
    fig, axes = plt.subplots(1, len(thresholds), figsize=(5.5 * len(thresholds), 4.5))
    axes = np.atleast_1d(axes)

    for ax, (label, threshold) in zip(axes, thresholds.items()):
        cm = confusion_matrix(y_true, (y_proba >= threshold).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if value > cm.max() / 2 else "black", fontsize=12)
        ax.set(
            title=f"{label} (t = {threshold:.3f})",
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["Pred: no", "Pred: yes"],
            yticklabels=["True: no", "True: yes"],
        )
    return save_figure(fig, out_dir / "confusion_matrices.png")


def plot_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dir: Path = config.FIGURES_DIR,
    n_repeats: int = 10,
    random_state: int = config.RANDOM_STATE,
) -> tuple[Path, pd.DataFrame]:
    """Permutation importance on the held-out set.

    Preferred over impurity-based importance, which is biased towards
    high-cardinality and continuous features. Permutation importance measures
    what the metric actually loses when a column is shuffled, on data the model
    has never seen.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats,
        random_state=random_state, scoring="roc_auc", n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(importance) + 1.6))
    ax.barh(importance["feature"], importance["importance_mean"],
            xerr=importance["importance_std"], color=PALETTE["primary"], alpha=0.9)
    ax.axvline(0, color=PALETTE["muted"], lw=1)
    ax.set(xlabel="Drop in ROC-AUC when the column is shuffled",
           title="Permutation importance (test set)")
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "permutation_importance.png"), importance.iloc[::-1]


def plot_partial_dependence(
    model,
    X_test: pd.DataFrame,
    features: list[str] | None = None,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Partial dependence for the continuous drivers.

    This is where the learned thresholds become visible: if the model has found
    the step in enrolment at roughly age 30 and roughly $62k salary, the curves
    show it without anyone having hard-coded a cut point.
    """
    features = features or ["age", "salary", "tenure_years"]

    # scikit-learn refuses integer columns here (grid points would be rounded),
    # and `age` is int64 in the raw data. The pipeline casts internally, but
    # partial dependence inspects the frame it is handed, so cast a copy too.
    X_test = X_test.copy()
    for column in config.NUMERIC_FEATURES:
        if column in X_test.columns:
            X_test[column] = X_test[column].astype(float)

    fig, axes = plt.subplots(1, len(features), figsize=(4.6 * len(features), 4))
    PartialDependenceDisplay.from_estimator(
        model, X_test, features=features, ax=np.atleast_1d(axes),
        line_kw={"color": PALETTE["primary"], "lw": 2},
    )
    for ax in np.atleast_1d(axes):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Partial dependence: modelled effect on enrolment probability", y=1.02)
    return save_figure(fig, out_dir / "partial_dependence.png")


def plot_model_comparison(
    leaderboard: pd.DataFrame, out_dir: Path = config.FIGURES_DIR
) -> Path:
    """Cross-validated ROC-AUC per candidate, with error bars."""
    df = leaderboard.sort_values("cv_roc_auc_mean")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.6))
    ax.barh(df["model"], df["cv_roc_auc_mean"], xerr=df["cv_roc_auc_std"],
            color=PALETTE["primary"], alpha=0.9)
    ax.set(xlabel="Cross-validated ROC-AUC (mean +/- s.d. over folds)",
           title="Model comparison", xlim=(0.4, 1.0))
    ax.axvline(0.5, ls="--", color=PALETTE["muted"], lw=1)
    for y, value in enumerate(df["cv_roc_auc_mean"]):
        ax.text(value + 0.012, y, f"{value:.4f}", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "model_comparison.png")
