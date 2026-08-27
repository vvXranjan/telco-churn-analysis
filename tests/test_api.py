"""API tests using FastAPI TestClient (no network, no OpenAI)."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def _client():
    return TestClient(app)


def test_health():
    with _client() as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info():
    with _client() as client:
        resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "XGBoost"
    assert body["threshold"] == 0.525
    assert body["feature_count"] == 45


def test_predict_example_customer(customer_38_features):
    with _client() as client:
        resp = client.post("/predict", json=customer_38_features)
    assert resp.status_code == 200
    body = resp.json()
    assert np.isclose(body["predicted_probability"], 0.7301, atol=1e-4)
    assert body["threshold"] == 0.525
    assert body["predicted_class"] == "Churn"
    assert len(body["shap_evidence"]) == 45
    # Evidence schema matches Notebook 04/05.
    first = body["shap_evidence"][0]
    assert set(first) == {"feature", "original_feature", "value", "value_display",
                          "shap_value", "direction", "abs_shap"}
    # No API key configured -> no LLM call, graceful response.
    assert body["llm_status"] == "not_configured"
    assert body["explanation"] is None
    assert body["verification"] is None


def test_predict_evidence_matches_notebook04(customer_38_features, evidence_json):
    with _client() as client:
        resp = client.post("/predict", json=customer_38_features)
    body = resp.json()
    api_by_feature = {e["feature"]: e["shap_value"] for e in body["shap_evidence"]}
    nb_by_feature = {e["feature"]: e["shap_value"] for e in evidence_json["evidence"]}
    assert api_by_feature == nb_by_feature


def test_predict_invalid_missing_field(customer_38_features):
    payload = dict(customer_38_features)
    payload.pop("Contract")
    with _client() as client:
        resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_negative_tenure(customer_38_features):
    payload = dict(customer_38_features)
    payload["tenure"] = -5
    with _client() as client:
        resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_invalid_senior_citizen(customer_38_features):
    payload = dict(customer_38_features)
    payload["SeniorCitizen"] = 7
    with _client() as client:
        resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_unknown_category_still_works(customer_38_features):
    payload = dict(customer_38_features)
    payload["InternetService"] = "UnknownProvider"
    with _client() as client:
        resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["predicted_probability"] <= 1.0


def test_model_loaded_not_retrained():
    # The service loads the persisted artifact; it must not contain fit code.
    assert ".fit(" not in open("app/services/model_service.py").read()