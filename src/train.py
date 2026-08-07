"""Training entry point: candidate search, selection, evaluation, persistence.

Run with ``python -m src.train`` (see ``--help`` for options). The script is
deterministic given a seed, and every artifact it produces — model, metrics,
figures, model card — is written under ``artifacts/``.

Selection protocol
------------------
1. Split off a 20% stratified test set and *do not touch it* until step 4.
2. Tune each candidate with randomised search over 5-fold stratified CV on the
   training set, scored by ROC-AUC.
3. Choose the winner with the one-standard-error rule: among models within one
   standard error of the best CV score, take the simplest. This prevents
   crowning a complex model on a difference that is indistinguishable from
   fold-to-fold noise.
4. Evaluate the winner once on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src import config, data, evaluate
from src.models import ModelSpec, build_pipeline, build_registry

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Outcome of tuning a single candidate."""

    spec: ModelSpec
    estimator: object
    cv_mean: float
    cv_std: float
    cv_scores: np.ndarray
    best_params: dict
    fit_seconds: float

    @property
    def standard_error(self) -> float:
        """Standard error of the CV mean, used by the selection rule."""
        return float(self.cv_std / np.sqrt(len(self.cv_scores)))


def tune_candidate(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    n_iter: int,
    add_features: bool,
    random_state: int,
) -> SearchResult:
    """Tune one candidate and return its cross-validated performance.

    Models with no search space (the dummy baseline) are cross-validated
    directly, so that every candidate is scored under the identical protocol.
    """
    pipeline = build_pipeline(spec, add_features=add_features)
    started = time.perf_counter()

    if spec.is_tunable:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spec.param_distributions,
            n_iter=n_iter,
            scoring=config.PRIMARY_METRIC,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_index = search.best_index_
        cv_scores = np.array(
            [
                search.cv_results_[f"split{i}_test_score"][best_index]
                for i in range(cv.get_n_splits())
            ]
        )
        estimator, best_params = search.best_estimator_, search.best_params_
    else:
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, scoring=config.PRIMARY_METRIC, cv=cv, n_jobs=-1
        )
        estimator = pipeline.fit(X_train, y_train)
        best_params = {}

    result = SearchResult(
        spec=spec,
        estimator=estimator,
        cv_mean=float(np.mean(cv_scores)),
        cv_std=float(np.std(cv_scores)),
        cv_scores=cv_scores,
        best_params={k: _jsonable(v) for k, v in best_params.items()},
        fit_seconds=time.perf_counter() - started,
    )
    logger.info(
        "%-24s CV %s = %.4f +/- %.4f  (%.1fs)",
        spec.name, config.PRIMARY_METRIC, result.cv_mean, result.cv_std, result.fit_seconds,
    )
    return result


def select_best(results: list[SearchResult]) -> SearchResult:
    """Apply the one-standard-error rule: simplest model within 1 SE of the best.

    The dummy baseline is excluded from selection — it exists only to anchor
    the reader's expectations, and on a degenerate dataset it could otherwise
    win by this rule.
    """
    candidates = [r for r in results if r.spec.name != "dummy"]
    if not candidates:
        raise ValueError("No candidate models were trained.")

    top = max(candidates, key=lambda r: r.cv_mean)
    tolerance = max(
        config.SELECTION_TOLERANCE_SE * top.standard_error, config.SELECTION_MIN_DELTA
    )
    cutoff = top.cv_mean - tolerance
    within_noise = [r for r in candidates if r.cv_mean >= cutoff]
    winner = min(within_noise, key=lambda r: r.spec.complexity_rank)

    if winner.spec.name != top.spec.name:
        logger.info(
            "Selected %r over the top scorer %r: within 1 SE (%.4f >= %.4f) and simpler.",
            winner.spec.name, top.spec.name, winner.cv_mean, cutoff,
        )
    return winner


def build_leaderboard(results: list[SearchResult]) -> pd.DataFrame:
    """Tabulate every candidate, best CV score first."""
    return pd.DataFrame(
        [
            {
                "model": r.spec.name,
                "cv_roc_auc_mean": round(r.cv_mean, 4),
                "cv_roc_auc_std": round(r.cv_std, 4),
                "cv_roc_auc_se": round(r.standard_error, 4),
                "fit_seconds": round(r.fit_seconds, 1),
                "complexity_rank": r.spec.complexity_rank,
                "best_params": r.best_params,
            }
            for r in results
        ]
    ).sort_values("cv_roc_auc_mean", ascending=False, ignore_index=True)


def _jsonable(value):
    """Coerce numpy scalars so ``json.dump`` does not choke on them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def run(args: argparse.Namespace) -> dict:
    """Execute the full training run and return the metrics payload."""
    for directory in (config.MODEL_DIR, config.METRICS_DIR, config.FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    X, y, quality = data.load_dataset(args.data)
    logger.info("Dataset: %d rows, positive rate %.3f", len(X), y.mean())
    if quality.implausible_tenure:
        logger.warning(
            "%d rows have tenure exceeding plausible working life; kept and "
            "encoded as tenure_ratio (see report.md).",
            quality.implausible_tenure,
        )

    X_train, X_test, y_train, y_test = data.stratified_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    logger.info("Train: %d rows | Test: %d rows (held out)", len(X_train), len(X_test))

    registry = build_registry(random_state=args.seed)
    selected = args.models or list(registry)
    unknown = set(selected) - set(registry)
    if unknown:
        raise SystemExit(f"Unknown model(s): {sorted(unknown)}. Available: {list(registry)}")

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    results = [
        tune_candidate(
            registry[name], X_train, y_train, cv,
            n_iter=args.n_iter,
            add_features=not args.no_feature_engineering,
            random_state=args.seed,
        )
        for name in selected
    ]

    leaderboard = build_leaderboard(results)
    print("\n=== Cross-validated model comparison (training set) ===")
    print(leaderboard.drop(columns=["best_params"]).to_string(index=False))

    best = select_best(results)
    model = best.estimator
    logger.info("Selected model: %s", best.spec.name)

    # ---- The held-out test set is used from here on, exactly once. ---------
    y_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, sweep = evaluate.find_optimal_threshold(y_test, y_proba)

    metrics_default = evaluate.compute_metrics(y_test, y_proba, config.DEFAULT_THRESHOLD)
    metrics_optimal = evaluate.compute_metrics(y_test, y_proba, optimal_threshold)
    slices = evaluate.slice_metrics(X_test, y_test, y_proba, optimal_threshold)

    print("\n=== Held-out test performance ===")
    print(f"ROC-AUC {metrics_default['roc_auc']:.4f} | "
          f"PR-AUC {metrics_default['pr_auc']:.4f} | "
          f"Brier {metrics_default['brier_score']:.4f}")
    print(f"@ t=0.500  accuracy {metrics_default['accuracy']:.4f}  "
          f"precision {metrics_default['precision']:.4f}  recall {metrics_default['recall']:.4f}")
    print(f"@ t={optimal_threshold:.3f}  accuracy {metrics_optimal['accuracy']:.4f}  "
          f"precision {metrics_optimal['precision']:.4f}  recall {metrics_optimal['recall']:.4f}")
    print(f"Cost per employee: {metrics_default['cost_per_employee']:.2f} -> "
          f"{metrics_optimal['cost_per_employee']:.2f} at the tuned threshold")

    figures = {}
    if not args.no_plots:
        figures["model_comparison"] = str(evaluate.plot_model_comparison(leaderboard))
        figures["roc_pr"] = str(evaluate.plot_roc_pr(y_test, y_proba))
        figures["calibration"] = str(evaluate.plot_calibration(y_test, y_proba))
        figures["threshold_analysis"] = str(
            evaluate.plot_threshold_analysis(sweep, optimal_threshold)
        )
        figures["confusion_matrices"] = str(
            evaluate.plot_confusion_matrices(
                y_test, y_proba,
                {"Default": config.DEFAULT_THRESHOLD, "Cost-optimal": optimal_threshold},
            )
        )
        figures["partial_dependence"] = str(evaluate.plot_partial_dependence(model, X_test))
        importance_path, importance = evaluate.plot_permutation_importance(
            model, X_test, y_test, random_state=args.seed
        )
        figures["permutation_importance"] = str(importance_path)
        print("\n=== Permutation importance (test set) ===")
        print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    payload = {
        "run": {
            "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "cv_folds": args.cv,
            "search_iterations": args.n_iter,
            "feature_engineering": not args.no_feature_engineering,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "data_quality": quality.to_dict(),
        "leaderboard": _jsonable(leaderboard.to_dict(orient="records")),
        "selected_model": {
            "name": best.spec.name,
            "rationale": best.spec.rationale,
            "best_params": best.best_params,
            "cv_roc_auc_mean": round(best.cv_mean, 4),
            "cv_roc_auc_std": round(best.cv_std, 4),
        },
        "test_metrics": {
            "at_default_threshold": metrics_default,
            "at_cost_optimal_threshold": metrics_optimal,
        },
        "cost_model": {
            "cost_false_positive": config.COST_FALSE_POSITIVE,
            "cost_false_negative": config.COST_FALSE_NEGATIVE,
            "optimal_threshold": optimal_threshold,
            "note": "Costs are a stated assumption, not measured; see report.md.",
        },
        "slice_metrics": _jsonable(slices.round(4).to_dict(orient="records")),
        "figures": figures,
    }

    (config.METRICS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    sweep.to_csv(config.METRICS_DIR / "threshold_sweep.csv", index=False)
    slices.to_csv(config.METRICS_DIR / "slice_metrics.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "model_name": best.spec.name,
            "threshold": optimal_threshold,
            "features": config.RAW_FEATURES,
            "trained_at": payload["run"]["timestamp"],
            "sklearn_pipeline_steps": [name for name, _ in model.steps],
        },
        args.model_path,
    )
    config.MODEL_CARD_PATH.write_text(
        json.dumps(
            {
                "model_name": best.spec.name,
                "rationale": best.spec.rationale,
                "trained_at": payload["run"]["timestamp"],
                "training_rows": int(len(X_train)),
                "features": config.RAW_FEATURES,
                "engineered_features": (
                    config.ENGINEERED_FEATURES if not args.no_feature_engineering else []
                ),
                "primary_metric": config.PRIMARY_METRIC,
                "test_roc_auc": metrics_default["roc_auc"],
                "recommended_threshold": optimal_threshold,
                "known_limitations": [
                    "Trained on synthetic data; coefficients should not be read "
                    "as real-world effects.",
                    "tenure_years carries no signal in this dataset and may in "
                    "reality; revisit before relying on it.",
                    "Cost assumptions (FP/FN) are placeholders and drive the "
                    "recommended threshold.",
                ],
            },
            indent=2,
        )
    )
    logger.info("Model -> %s | Metrics -> %s", args.model_path, config.METRICS_DIR / "metrics.json")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the insurance-enrolment model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH,
                        help="Path to the employee CSV.")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH,
                        help="Where to write the fitted pipeline.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of candidates to train (default: all).")
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--cv", type=int, default=config.CV_FOLDS)
    parser.add_argument("--n-iter", type=int, default=config.SEARCH_ITERATIONS,
                        help="Randomised-search candidates per tunable model.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--no-feature-engineering", action="store_true",
                        help="Ablation: train on raw columns only.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation (faster CI runs).")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)


if __name__ == "__main__":
    main()
