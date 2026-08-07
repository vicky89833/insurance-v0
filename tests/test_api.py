"""Tests for the FastAPI prediction service."""

from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from src import api, config
from src.models import build_pipeline, build_registry


@pytest.fixture
def client(sample_xy, tmp_path, monkeypatch):
    """A client backed by a small model trained on the synthetic fixture."""
    X, y = sample_xy
    model = build_pipeline(build_registry()["decision_tree"]).fit(X, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": "decision_tree",
            "threshold": 0.5,
            "features": config.RAW_FEATURES,
            "trained_at": "2026-01-01T00:00:00",
            "sklearn_pipeline_steps": ["preprocessor", "classifier"],
        },
        model_path,
    )
    monkeypatch.setattr(config, "MODEL_PATH", model_path)

    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    """A client started with no model artifact on disk."""
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "missing.joblib")
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "decision_tree"


def test_health_is_degraded_without_a_model(client_without_model):
    """A missing artifact must not stop the service from starting."""
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded", "model_loaded": False,
        "model_name": None, "trained_at": None,
    }


def test_prediction_returns_503_without_a_model(client_without_model, valid_employee):
    response = client_without_model.post("/predict", json=valid_employee)
    assert response.status_code == 503
    assert "python -m src.train" in response.json()["detail"]


def test_predict_returns_a_calibrated_shaped_response(client, valid_employee):
    response = client.post("/predict", json=valid_employee)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["enrollment_probability"] <= 1.0
    assert isinstance(body["will_enroll"], bool)
    assert body["model_name"] == "decision_tree"


def test_predict_matches_the_generating_rule(client, valid_employee):
    """41 / 72.5k / Full-time / dependents satisfies all four conditions."""
    body = client.post("/predict", json=valid_employee).json()
    assert body["will_enroll"] is True

    # Flip three conditions off: under 30, low salary, part-time.
    unlikely = valid_employee | {"age": 25, "salary": 30_000.0, "employment_type": "Part-time"}
    assert client.post("/predict", json=unlikely).json()["will_enroll"] is False


def test_threshold_override_changes_the_decision(client, valid_employee):
    """A per-request threshold must override the one stored with the model.

    The comparison is ``probability >= threshold``, matching scikit-learn's
    convention. The tested employee scores near zero under the default
    threshold and is only flagged once the threshold drops to zero.
    """
    unlikely = valid_employee | {"age": 25, "salary": 30_000.0, "employment_type": "Part-time"}

    default = client.post("/predict", json=unlikely).json()
    lenient = client.post("/predict?threshold=0.0", json=unlikely).json()

    assert default["threshold"] == 0.5
    assert default["will_enroll"] is False
    assert lenient["threshold"] == 0.0
    assert lenient["will_enroll"] is True


def test_out_of_range_threshold_is_rejected(client, valid_employee):
    assert client.post("/predict?threshold=1.5", json=valid_employee).status_code == 422


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"age": 5}, "age below the allowed minimum"),
        ({"age": "forty"}, "non-numeric age"),
        ({"salary": -100}, "negative salary"),
        ({"tenure_years": 999}, "tenure beyond a working life"),
        ({"region": "Atlantis"}, "unknown region"),
        ({"employment_type": "Freelance"}, "unknown employment type"),
    ],
)
def test_invalid_input_is_rejected_with_422(client, valid_employee, override, reason):
    response = client.post("/predict", json=valid_employee | override)
    assert response.status_code == 422, reason


def test_missing_field_is_rejected(client, valid_employee):
    del valid_employee["salary"]
    assert client.post("/predict", json=valid_employee).status_code == 422


def test_batch_prediction_preserves_order_and_count(client, valid_employee):
    unlikely = valid_employee | {"age": 25, "salary": 30_000.0, "employment_type": "Part-time"}
    response = client.post("/predict/batch", json=[valid_employee, unlikely, valid_employee])
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 3
    assert [p["will_enroll"] for p in body["predictions"]] == [True, False, True]


def test_batch_matches_single_prediction(client, valid_employee):
    """Batching must not change a score - a common source of serving skew."""
    single = client.post("/predict", json=valid_employee).json()
    batched = client.post("/predict/batch", json=[valid_employee]).json()["predictions"][0]
    assert single["enrollment_probability"] == batched["enrollment_probability"]


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json=[]).status_code == 422


def test_oversized_batch_is_rejected(client, valid_employee):
    response = client.post("/predict/batch", json=[valid_employee] * (api.MAX_BATCH_SIZE + 1))
    assert response.status_code == 413
    assert str(api.MAX_BATCH_SIZE) in response.json()["detail"]


def test_model_info_exposes_the_serving_contract(client):
    body = client.get("/model/info").json()
    assert body["model_name"] == "decision_tree"
    assert body["features"] == config.RAW_FEATURES
    assert body["default_threshold"] == 0.5


def test_openapi_schema_documents_allowed_categories(client):
    """The allowed values should be discoverable without reading the source."""
    schema = client.get("/openapi.json").json()
    employee = schema["components"]["schemas"]["EmployeeFeatures"]["properties"]
    assert set(employee["region"]["enum"]) == set(config.EXPECTED_CATEGORIES["region"])
