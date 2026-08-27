"""Evidence JSON schema and internal-consistency tests."""

import numpy as np

from app.services.evidence import to_canonical, validate_evidence


def test_evidence_json_schema_valid(evidence_json):
    errors = validate_evidence(evidence_json)
    assert errors == [], f"evidence schema errors: {errors}"


def test_evidence_top_level_fields(evidence_json):
    assert evidence_json["customer_index"] == 38
    assert 0.0 <= evidence_json["predicted_probability"] <= 1.0
    assert evidence_json["threshold"] == 0.525
    assert evidence_json["predicted_class"] == 1


def test_evidence_nonempty_and_45_items(evidence_json):
    assert len(evidence_json["evidence"]) == 45
    assert len({e["feature"] for e in evidence_json["evidence"]}) == 45


def test_evidence_internally_consistent(evidence_json):
    for item in evidence_json["evidence"]:
        assert np.isfinite(item["shap_value"])
        assert abs(item["abs_shap"] - abs(item["shap_value"])) < 1e-9
        expected = "toward churn" if item["shap_value"] >= 0 else "away from churn"
        assert item["direction"] == expected


def test_evidence_matches_nb05_schema(evidence_json):
    for item in evidence_json["evidence"]:
        assert set(item) == {
            "feature",
            "original_feature",
            "value",
            "value_display",
            "shap_value",
            "direction",
            "abs_shap",
        }


def test_canonical_conversion(evidence_json):
    canonical = to_canonical(evidence_json)
    assert canonical["predicted_class"] == 1
    assert len(canonical["features"]) == 45
    assert canonical["features"][0]["feature"] == evidence_json["evidence"][0]["feature"]