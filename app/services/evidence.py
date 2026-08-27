"""Evidence schema validation and canonical representation.

The authoritative evidence schema is defined by Notebook 04 and consumed by
Notebook 05. This module reuses that exact schema so the API, tests, and drift
checks all speak the same language as the notebooks.
"""

import numpy as np
import pandas as pd


def validate_evidence(data):
    """Return a list of schema/consistency errors (empty list means valid).

    Mirrors the validation implemented in Notebook 05.
    """
    errors = []

    required_top = ["customer_index", "predicted_probability", "threshold",
                    "predicted_class", "evidence"]
    for key in required_top:
        if key not in data:
            errors.append(f"missing top-level key: {key}")

    if errors:
        return errors

    prob = data["predicted_probability"]
    thr = data["threshold"]
    if not (isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0):
        errors.append(f"predicted_probability not in [0,1]: {prob!r}")
    if not (isinstance(thr, (int, float)) and 0.0 <= thr <= 1.0):
        errors.append(f"threshold not in [0,1]: {thr!r}")

    evidence = data["evidence"]
    if not isinstance(evidence, list) or len(evidence) == 0:
        errors.append("evidence is missing or empty")

    for i, item in enumerate(evidence):
        for key in ["feature", "value", "shap_value", "direction"]:
            if key not in item:
                errors.append(f"evidence[{i}] missing key: {key}")
                continue
        sv = item.get("shap_value")
        if not (isinstance(sv, (int, float)) and np.isfinite(sv)):
            errors.append(f"evidence[{i}] shap_value is not a finite number: {sv!r}")
        expected = "toward churn" if sv >= 0 else "away from churn"
        if item.get("direction") != expected:
            errors.append(
                f"evidence[{i}] direction {item.get('direction')!r} inconsistent "
                f"with SHAP sign (expected {expected!r})"
            )
    return errors


def to_canonical(data):
    """Build the canonical authoritative evidence dict used by the verifier."""
    return {
        "customer_index": data["customer_index"],
        "predicted_probability": float(data["predicted_probability"]),
        "threshold": float(data["threshold"]),
        "predicted_class": int(data["predicted_class"]),
        "base_value": float(data.get("base_value", 0.0)),
        "features": [
            {
                "feature": item["feature"],
                "value": item["value"],
                "value_display": item.get("value_display", ""),
                "shap_value": float(item["shap_value"]),
                "direction": item["direction"],
            }
            for item in data["evidence"]
        ],
    }


def reduce_evidence(evidence, k=5):
    """Return (positive, negative) strongest contributors, each sorted by
    absolute SHAP value descending, matching Notebook 05's evidence reduction."""
    pos = [e for e in evidence if e["shap_value"] > 0]
    neg = [e for e in evidence if e["shap_value"] < 0]
    pos = sorted(pos, key=lambda e: abs(e["shap_value"]), reverse=True)[:k]
    neg = sorted(neg, key=lambda e: abs(e["shap_value"]), reverse=True)[:k]
    return pos, neg