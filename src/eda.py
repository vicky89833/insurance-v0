"""Exploratory data analysis, as a reproducible script rather than a notebook.

Run with ``python -m src.eda``. Writes figures to ``artifacts/figures/`` and a
machine-readable summary to ``artifacts/metrics/eda_summary.json``, so the
numbers quoted in ``report.md`` can always be regenerated and checked. A
notebook would carry stale cell outputs; this cannot.

The most consequential section is ``discover_generating_rule``. Before trusting
any model score on this dataset, it is worth asking whether the target contains
any noise at all — and here it does not.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src import config, data
from src.evaluate import PALETTE, save_figure

logger = logging.getLogger(__name__)

# Candidate binary conditions tested by the rule-recovery search. Each is a
# threshold or level suggested by the univariate enrolment rates below.
CANDIDATE_CONDITIONS: dict[str, str] = {
    "age > 30": "age > 30",
    "salary > 60000": "salary > 60000",
    "employment_type == 'Full-time'": "employment_type == 'Full-time'",
    "has_dependents == 'Yes'": "has_dependents == 'Yes'",
}


def univariate_rates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Enrolment rate by category level and by numeric quintile.

    This is the cheapest possible signal check, and it is what first exposes
    the step changes in ``age`` and ``salary``.
    """
    tables: dict[str, pd.DataFrame] = {}

    for column in config.CATEGORICAL_FEATURES:
        tables[column] = (
            df.groupby(column, observed=True)[config.TARGET]
            .agg(enrolment_rate="mean", n="size")
            .round(4)
            .reset_index()
            .rename(columns={column: "level"})
            .assign(feature=column)
        )

    for column in config.NUMERIC_FEATURES:
        bins = pd.qcut(df[column], 5, duplicates="drop")
        tables[column] = (
            df.groupby(bins, observed=True)[config.TARGET]
            .agg(enrolment_rate="mean", n="size")
            .round(4)
            .reset_index()
            .rename(columns={column: "level"})
            .assign(feature=column, level=lambda t: t["level"].astype(str))
        )

    return tables


def discover_generating_rule(df: pd.DataFrame) -> dict:
    """Test whether the target is a deterministic function of the features.

    Method:

    1. Fit an unconstrained decision tree on the *entire* dataset. If it
       reaches 100% training accuracy with a shallow tree and few leaves, the
       label carries no noise — there is no irreducible (Bayes) error, and a
       perfect test score is a property of the data, not evidence of a good
       model or of leakage.
    2. If so, search "at least k of these n conditions" rules over the
       candidate conditions and report any that reproduce the label exactly.

    This matters because it changes how every downstream metric should be
    read. It is reported in full in ``report.md``.
    """
    X = pd.get_dummies(df[config.RAW_FEATURES])
    y = df[config.TARGET]

    tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE).fit(X, y)
    is_deterministic = bool(np.isclose(tree.score(X, y), 1.0))

    findings: dict = {
        "unconstrained_tree_train_accuracy": round(float(tree.score(X, y)), 6),
        "unconstrained_tree_depth": int(tree.get_depth()),
        "unconstrained_tree_leaves": int(tree.get_n_leaves()),
        "target_is_deterministic": is_deterministic,
        "exact_rules": [],
    }

    if not is_deterministic:
        logger.info("Target is not perfectly separable; skipping rule search.")
        return findings

    logger.warning(
        "Target is a DETERMINISTIC function of the features "
        "(unconstrained tree: 100%% train accuracy, depth %d, %d leaves).",
        tree.get_depth(), tree.get_n_leaves(),
    )

    condition_matrix = pd.DataFrame(
        {name: df.eval(expr).astype(int) for name, expr in CANDIDATE_CONDITIONS.items()}
    )

    # Search "at least k of any subset of size n" rules, smallest first.
    for n in range(2, len(CANDIDATE_CONDITIONS) + 1):
        for subset in combinations(CANDIDATE_CONDITIONS, n):
            satisfied = condition_matrix[list(subset)].sum(axis=1)
            for k in range(1, n + 1):
                if ((satisfied >= k).astype(int) == y).all():
                    rule = f"at least {k} of {n}: " + " | ".join(subset)
                    findings["exact_rules"].append(
                        {"k": k, "n": n, "conditions": list(subset), "description": rule}
                    )
                    logger.warning("Exact rule recovered -> %s", rule)

    findings["decision_tree_text"] = export_text(
        DecisionTreeClassifier(max_depth=4, random_state=config.RANDOM_STATE).fit(X, y),
        feature_names=list(X.columns),
        decimals=1,
    )
    return findings


def plot_target_relationships(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Enrolment rate across every feature, in one grid."""
    features = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
    n_cols = 4
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    overall = df[config.TARGET].mean()
    for ax, column in zip(axes, features):
        if column in config.NUMERIC_FEATURES:
            grouped = df.groupby(pd.qcut(df[column], 10, duplicates="drop"), observed=True)
            rates = grouped[config.TARGET].mean()
            ax.plot(range(len(rates)), rates.values, "o-", color=PALETTE["primary"])
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels([f"{i.mid:,.0f}" for i in rates.index], rotation=45, fontsize=7)
            ax.set_xlabel(f"{column} (decile midpoint)")
        else:
            rates = df.groupby(column, observed=True)[config.TARGET].mean().sort_values()
            ax.bar(rates.index.astype(str), rates.values, color=PALETTE["primary"], alpha=0.9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_xlabel(column)

        ax.axhline(overall, ls="--", color=PALETTE["negative"], lw=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Enrolment rate")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Enrolment rate by feature (dashed line = overall rate, {overall:.1%})",
        y=1.01, fontsize=13,
    )
    return save_figure(fig, out_dir / "eda_target_relationships.png")


def plot_distributions(df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR) -> Path:
    """Numeric distributions split by outcome, to show where the classes sit."""
    fig, axes = plt.subplots(1, len(config.NUMERIC_FEATURES),
                             figsize=(4.6 * len(config.NUMERIC_FEATURES), 3.8))
    for ax, column in zip(np.atleast_1d(axes), config.NUMERIC_FEATURES):
        for label, colour, name in [
            (0, PALETTE["muted"], "Not enrolled"),
            (1, PALETTE["primary"], "Enrolled"),
        ]:
            ax.hist(df.loc[df[config.TARGET] == label, column], bins=40,
                    alpha=0.65, color=colour, label=name)
        ax.set(xlabel=column, ylabel="Employees", title=f"{column} by outcome")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    return save_figure(fig, out_dir / "eda_distributions.png")


def run(path: Path = config.RAW_DATA_PATH) -> dict:
    """Run the full EDA and persist figures plus a JSON summary."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw = data.load_raw(path)
    quality = data.build_quality_report(raw)
    df = data.clean(raw)

    print("=== Data quality ===")
    print(json.dumps(quality.to_dict(), indent=2))

    tables = univariate_rates(df)
    print("\n=== Enrolment rate by feature ===")
    for name, table in tables.items():
        print(f"\n-- {name} --")
        print(table[["level", "enrolment_rate", "n"]].to_string(index=False))

    rule = discover_generating_rule(df)
    print("\n=== Generating-rule check ===")
    print(f"Target is deterministic: {rule['target_is_deterministic']}")
    for exact in rule["exact_rules"]:
        print(f"  EXACT RULE: {exact['description']}")

    figures = {
        "target_relationships": str(plot_target_relationships(df)),
        "distributions": str(plot_distributions(df)),
    }

    summary = {
        "data_quality": quality.to_dict(),
        "univariate_rates": {
            name: table.to_dict(orient="records") for name, table in tables.items()
        },
        "generating_rule": rule,
        "figures": figures,
    }
    (config.METRICS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("EDA summary -> %s", config.METRICS_DIR / "eda_summary.json")
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
