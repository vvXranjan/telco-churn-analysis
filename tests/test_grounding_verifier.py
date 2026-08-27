"""Deterministic grounding verifier tests.

Reproduces the Notebook 05 valid-mock (PASS) and adversarial (FLAG) scenarios,
plus individual failure cases. No LLM is involved.
"""

import numpy as np

from app.services.evidence import reduce_evidence, to_canonical
from app.services.verifier import verify_explanation


def _authoritative(evidence_json):
    return to_canonical(evidence_json)


def _supplied_features(evidence_json):
    positive, negative = reduce_evidence(evidence_json["evidence"], k=5)
    return {e["feature"] for e in positive + negative}


def _valid_mock(evidence_json):
    """Evidence-compliant deterministic response (mirrors Notebook 05)."""
    canonical = _authoritative(evidence_json)
    positive, negative = reduce_evidence(evidence_json["evidence"], k=5)
    expected_class = "Churn" if canonical["predicted_class"] == 1 else "Retained"
    supporting = [
        {
            "feature": e["feature"],
            "direction": e["direction"],
            "shap_value": e["shap_value"],
            "explanation": (
                f"The model's output was pushed toward churn by {e['feature']}."
            ),
        }
        for e in positive
    ]
    countervailing = [
        {
            "feature": e["feature"],
            "direction": e["direction"],
            "shap_value": e["shap_value"],
            "explanation": (
                f"The model's output was pushed away from churn by {e['feature']}."
            ),
        }
        for e in negative
    ]
    return {
        "risk_summary": (
            f"The model assigns this customer a "
            f"{canonical['predicted_probability'] * 100:.2f}% churn probability, "
            f"above the {canonical['threshold'] * 100:.1f}% decision threshold."
        ),
        "predicted_probability": canonical["predicted_probability"],
        "threshold": canonical["threshold"],
        "predicted_class": expected_class,
        "supporting_evidence": supporting,
        "countervailing_evidence": countervailing,
        "limitations": "This explanation is derived from the supplied model evidence.",
    }


def _adversarial(evidence_json):
    """Intentionally fabricated response (mirrors Notebook 05)."""
    canonical = _authoritative(evidence_json)
    mcm = next(f for f in canonical["features"]
               if f["feature"] == "numerical__MonthlyCharges")
    return {
        "risk_summary": "High monthly charges led to churn for this customer.",
        "predicted_probability": 0.9999,
        "threshold": 0.50,
        "predicted_class": "Churn",
        "supporting_evidence": [
            {
                "feature": "customer_income",
                "direction": "toward churn",
                "shap_value": 0.9,
                "explanation": "High customer income caused churn.",
            },
            {
                "feature": mcm["feature"],
                "direction": "away from churn",
                "shap_value": 0.1234,
                "explanation": "The customer's monthly charge pushed the model away from churn.",
            },
        ],
        "countervailing_evidence": [],
        "limitations": "The customer actually churned, so this is the ground truth.",
    }


def test_valid_mock_passes(evidence_json):
    result = verify_explanation(
        _valid_mock(evidence_json), _authoritative(evidence_json),
        _supplied_features(evidence_json),
    )
    assert result["status"] == "PASS"
    assert all(result["checks"].values())
    assert result["flags"] == []


def test_adversarial_flags(evidence_json):
    result = verify_explanation(
        _adversarial(evidence_json), _authoritative(evidence_json),
        _supplied_features(evidence_json),
    )
    assert result["status"] == "FLAG"
    assert result["checks"]["features_grounded"] is False
    assert result["checks"]["no_unsupported_evidence"] is False
    assert result["checks"]["shap_values_grounded"] is False
    assert result["checks"]["directions_consistent"] is False
    assert result["checks"]["probability_grounded"] is False
    assert result["checks"]["threshold_grounded"] is False
    assert result["checks"]["no_causal_claim"] is False
    assert result["checks"]["no_ground_truth_leakage"] is False


def _base_response(evidence_json):
    valid = _valid_mock(evidence_json)
    return {
        "risk_summary": "The model assigns a churn probability above the threshold.",
        "predicted_probability": valid["predicted_probability"],
        "threshold": valid["threshold"],
        "predicted_class": valid["predicted_class"],
        "supporting_evidence": [valid["supporting_evidence"][0]],
        "countervailing_evidence": [valid["countervailing_evidence"][0]],
        "limitations": "Derived from supplied evidence.",
    }


def test_fabricated_feature_flags(evidence_json):
    resp = _base_response(evidence_json)
    resp["supporting_evidence"] = [
        {"feature": "not_a_real_feature", "direction": "toward churn",
         "shap_value": 0.1, "explanation": "x"}
    ]
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["features_grounded"] is False
    assert result["status"] == "FLAG"


def test_fabricated_shap_flags(evidence_json):
    resp = _base_response(evidence_json)
    item = resp["supporting_evidence"][0]
    item["shap_value"] = item["shap_value"] + 0.2
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["shap_values_grounded"] is False
    assert result["status"] == "FLAG"


def test_reversed_direction_flags(evidence_json):
    resp = _base_response(evidence_json)
    item = resp["supporting_evidence"][0]
    item["direction"] = "away from churn"
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["directions_consistent"] is False
    assert result["status"] == "FLAG"


def test_fabricated_probability_flags(evidence_json):
    resp = _base_response(evidence_json)
    resp["predicted_probability"] = 0.99
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["probability_grounded"] is False
    assert result["status"] == "FLAG"


def test_fabricated_threshold_flags(evidence_json):
    resp = _base_response(evidence_json)
    resp["threshold"] = 0.40
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["threshold_grounded"] is False
    assert result["status"] == "FLAG"


def test_causal_claim_flags(evidence_json):
    resp = _base_response(evidence_json)
    resp["risk_summary"] = "The contract caused churn."
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["no_causal_claim"] is False
    assert result["status"] == "FLAG"


def test_ground_truth_leakage_flags(evidence_json):
    resp = _base_response(evidence_json)
    resp["limitations"] = "The customer actually churned."
    result = verify_explanation(resp, _authoritative(evidence_json),
                                _supplied_features(evidence_json))
    assert result["checks"]["no_ground_truth_leakage"] is False
    assert result["status"] == "FLAG"


def test_unsupported_feature_flags(evidence_json):
    resp = _base_response(evidence_json)
    # A feature that exists in the full evidence but was NOT supplied to the LLM.
    full_features = {f["feature"] for f in _authoritative(evidence_json)["features"]}
    supplied = _supplied_features(evidence_json)
    unsupplied = next(f for f in sorted(full_features) if f not in supplied)
    resp["supporting_evidence"] = [
        {"feature": unsupplied, "direction": "toward churn",
         "shap_value": 0.05, "explanation": "x"}
    ]
    result = verify_explanation(resp, _authoritative(evidence_json), supplied)
    assert result["checks"]["no_unsupported_evidence"] is False
    assert result["status"] == "FLAG"