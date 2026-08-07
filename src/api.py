"""FastAPI service exposing the trained model.

Run with::

    uvicorn src.api:app --reload

Interactive documentation is then served at ``/docs``.

Design notes
------------
* The model is loaded **once** at startup, not per request. Loading a joblib
  bundle inside the handler would dominate latency.
* The stored artifact carries the cost-optimal threshold chosen at training
  time, so the service defaults to the same operating point that was evaluated
  in the report. Callers can override it per request.
* Requests are validated by Pydantic before they reach the model, which turns
  malformed input into a 422 with a precise message instead of an opaque 500.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 1000

# Loaded at startup by the lifespan handler below.
_state: dict = {"bundle": None}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class EmployeeFeatures(BaseModel):
    """One employee record.

    Categorical fields are typed as literals so that the allowed values appear
    in the OpenAPI schema and bad input is rejected with a clear message. The
    underlying pipeline also encodes unknown categories safely, so relaxing
    these to plain strings would degrade gracefully rather than fail.
    """

    age: Annotated[int, Field(ge=16, le=100, description="Age in years.")]
    salary: Annotated[float, Field(ge=0, le=1_000_000, description="Annual salary.")]
    tenure_years: Annotated[
        float, Field(ge=0, le=60, description="Years of service at the company.")
    ]
    gender: Literal["Female", "Male", "Other"]
    marital_status: Literal["Divorced", "Married", "Single", "Widowed"]
    employment_type: Literal["Contract", "Full-time", "Part-time"]
    region: Literal["Midwest", "Northeast", "South", "West"]
    has_dependents: Literal["No", "Yes"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 41,
                    "salary": 72500.0,
                    "tenure_years": 6.5,
                    "gender": "Female",
                    "marital_status": "Married",
                    "employment_type": "Full-time",
                    "region": "West",
                    "has_dependents": "Yes",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Model output for a single employee."""

    enrollment_probability: float = Field(
        ..., description="Predicted probability of enrolling, in [0, 1]."
    )
    will_enroll: bool = Field(
        ..., description="Probability compared against the decision threshold."
    )
    threshold: float = Field(..., description="Threshold used for this response.")
    model_name: str

    # Pydantic reserves the `model_` prefix; opt out so `model_name` is allowed.
    model_config = {"protected_namespaces": ()}


class BatchPredictionResponse(BaseModel):
    """Model output for a batch of employees, in the order submitted."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    trained_at: str | None = None

    model_config = {"protected_namespaces": ()}


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and release it on shutdown.

    A missing artifact is logged rather than raised: the service still starts
    and reports ``degraded`` on ``/health``, which is friendlier to an
    orchestrator than a crash loop.
    """
    try:
        _state["bundle"] = joblib.load(config.MODEL_PATH)
        logger.info(
            "Loaded model %r (trained %s) from %s",
            _state["bundle"]["model_name"],
            _state["bundle"]["trained_at"],
            config.MODEL_PATH,
        )
    except FileNotFoundError:
        _state["bundle"] = None
        logger.error(
            "No model at %s. Run `python -m src.train` first; "
            "prediction endpoints will return 503 until then.",
            config.MODEL_PATH,
        )
    yield
    _state["bundle"] = None


app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description=(
        "Scores employees on their likelihood of enrolling in the voluntary "
        "insurance product. See report.md for how the model was selected and "
        "for the (important) caveats about the training data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_bundle() -> dict:
    """Return the loaded model bundle, or fail with 503 if unavailable."""
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train` and restart the service.",
        )
    return bundle


def _score(bundle: dict, records: list[EmployeeFeatures], threshold: float) -> list[PredictionResponse]:
    """Score records as a single batch.

    One vectorised call to ``predict_proba`` handles the whole batch; looping
    per row would multiply the pipeline overhead by the batch size.
    """
    frame = pd.DataFrame([record.model_dump() for record in records])[config.RAW_FEATURES]
    probabilities = bundle["model"].predict_proba(frame)[:, 1]
    return [
        PredictionResponse(
            enrollment_probability=round(float(p), 6),
            will_enroll=bool(p >= threshold),
            threshold=threshold,
            model_name=bundle["model_name"],
        )
        for p in probabilities
    ]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness and readiness check."""
    bundle = _state.get("bundle")
    if bundle is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=bundle["model_name"],
        trained_at=bundle.get("trained_at"),
    )


@app.get("/model/info", tags=["ops"])
def model_info() -> dict:
    """Model card: what is deployed, how it was trained, and its limitations."""
    bundle = _require_bundle()
    card = {}
    if config.MODEL_CARD_PATH.exists():
        import json

        card = json.loads(config.MODEL_CARD_PATH.read_text())
    return {
        "model_name": bundle["model_name"],
        "trained_at": bundle.get("trained_at"),
        "features": bundle.get("features", config.RAW_FEATURES),
        "pipeline_steps": bundle.get("sklearn_pipeline_steps", []),
        "default_threshold": bundle.get("threshold", config.DEFAULT_THRESHOLD),
        "model_card": card,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(
    employee: Annotated[EmployeeFeatures, Body()],
    threshold: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="Override the default decision threshold."),
    ] = None,
) -> PredictionResponse:
    """Score a single employee."""
    bundle = _require_bundle()
    effective = threshold if threshold is not None else bundle.get(
        "threshold", config.DEFAULT_THRESHOLD
    )
    return _score(bundle, [employee], effective)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(
    employees: Annotated[list[EmployeeFeatures], Body()],
    threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> BatchPredictionResponse:
    """Score up to ``MAX_BATCH_SIZE`` employees in one request."""
    if not employees:
        raise HTTPException(status_code=422, detail="Request body must contain at least one employee.")
    if len(employees) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size {len(employees)} exceeds the limit of {MAX_BATCH_SIZE}.",
        )

    bundle = _require_bundle()
    effective = threshold if threshold is not None else bundle.get(
        "threshold", config.DEFAULT_THRESHOLD
    )
    predictions = _score(bundle, employees, effective)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
