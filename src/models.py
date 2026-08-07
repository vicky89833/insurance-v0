"""Model registry: the candidate estimators and their search spaces.

Each candidate is described once, declaratively, so that ``train.py`` stays a
thin orchestration layer and adding a model is a single entry in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.features import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one candidate model.

    Attributes:
        name: Registry key, also used in filenames and report tables.
        estimator: Unfitted estimator instance.
        preprocessor_kind: Which numeric treatment this family needs.
        param_distributions: Search space, keys prefixed ``classifier__``.
        complexity_rank: Tie-breaker for model selection — lower is simpler.
            Used by the one-standard-error rule so that a model is only
            promoted over a simpler one when the gain is larger than noise.
        rationale: Why this candidate is in the comparison at all.
    """

    name: str
    estimator: Any
    preprocessor_kind: str
    complexity_rank: int
    rationale: str
    param_distributions: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tunable(self) -> bool:
        return bool(self.param_distributions)


def build_registry(random_state: int = config.RANDOM_STATE) -> dict[str, ModelSpec]:
    """Return every candidate model, keyed by name.

    Ordered from simplest to most flexible, which is also the order the report
    walks through them in.
    """
    specs = [
        ModelSpec(
            name="dummy",
            estimator=DummyClassifier(strategy="prior"),
            preprocessor_kind="tree",
            complexity_rank=0,
            rationale=(
                "Predicts the majority class for everyone. Establishes the "
                "accuracy floor (~61.7%), which stops a mediocre model from "
                "looking good on accuracy alone."
            ),
        ),
        ModelSpec(
            name="logistic",
            estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            preprocessor_kind="linear",
            complexity_rank=1,
            rationale=(
                "Interpretable, well-calibrated baseline. Expected to underfit: "
                "age and salary act on enrolment as step functions, which a "
                "linear term in the log-odds cannot represent."
            ),
            param_distributions={
                # L2 is the default penalty; scikit-learn >= 1.8 deprecates
                # setting `penalty` explicitly in favour of C / l1_ratio.
                "classifier__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="logistic_binned",
            estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            preprocessor_kind="linear_binned",
            complexity_rank=2,
            rationale=(
                "Same estimator, quantile-binned numerics. Isolates whether the "
                "linear baseline is limited by model capacity or purely by "
                "feature representation."
            ),
            param_distributions={
                "classifier__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
                "preprocessor__columns__numeric__binner__n_bins": [5, 8, 12, 20],
            },
        ),
        ModelSpec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(random_state=random_state),
            preprocessor_kind="tree",
            complexity_rank=3,
            rationale=(
                "Depth-limited single tree. Kept for explanation rather than "
                "accuracy: the printed splits show the learned thresholds in a "
                "form a non-technical stakeholder can read."
            ),
            param_distributions={
                # min_samples_leaf reaches down to 1 and depth to 8 on purpose:
                # the target turns out to be a deterministic rule (see
                # notes/rule_discovery in the EDA), so a small tree can fit it
                # exactly and over-regularising would hide that.
                "classifier__max_depth": [3, 4, 5, 6, 8, None],
                "classifier__min_samples_leaf": [1, 2, 5, 10, 25, 50],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=random_state, n_jobs=-1, n_estimators=300
            ),
            preprocessor_kind="tree",
            complexity_rank=4,
            rationale=(
                "Bagged trees: a strong, low-maintenance non-linear reference "
                "that needs little tuning to be competitive."
            ),
            param_distributions={
                "classifier__n_estimators": [200, 300, 500],
                "classifier__max_depth": [None, 6, 10, 16],
                "classifier__min_samples_leaf": [1, 5, 10, 25],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(random_state=random_state),
            preprocessor_kind="tree",
            complexity_rank=5,
            rationale=(
                "Boosted histogram trees. Expected top performer: threshold "
                "effects are exactly what axis-aligned splits capture, and it "
                "trains in seconds at this data size."
            ),
            param_distributions={
                "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "classifier__max_leaf_nodes": [7, 15, 31, 63],
                "classifier__min_samples_leaf": [10, 20, 50],
                "classifier__l2_regularization": [0.0, 0.1, 1.0],
                "classifier__max_iter": [200, 400],
            },
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_pipeline(spec: ModelSpec, add_features: bool = True) -> Pipeline:
    """Compose preprocessing and estimator into one fittable pipeline.

    The estimator is cloned by scikit-learn during search/CV, so the shared
    registry instance is never mutated.
    """
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(spec.preprocessor_kind, add_features=add_features),
            ),
            ("classifier", spec.estimator),
        ]
    )
