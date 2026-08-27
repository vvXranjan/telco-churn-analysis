"""SHAP evidence generation for single customers.

Reuses the exact evidence schema produced by Notebook 04
(feature, original_feature, value, value_display, shap_value, direction,
abs_shap) so the API output stays compatible with Notebook 05.
"""

import numpy as np


def decode_feature(feature_name):
    """Split a transformed feature name into (kind, original_column[, category])."""
    if feature_name.startswith("numerical__"):
        return ("numerical", feature_name.split("__", 1)[1], None)
    if feature_name.startswith("categorical__"):
        _, rest = feature_name.split("__", 1)
        col, _, category = rest.rpartition("_")
        return ("categorical", col, category)
    return ("unknown", feature_name, None)


def build_evidence(features_dict, proba, X_row, pipeline):
    """Build the per-customer SHAP evidence list (Notebook 04 schema).

    Parameters
    ----------
    features_dict : dict
        Raw customer features (original values).
    proba : float
        Predicted churn probability for this customer.
    X_row : ndarray
        Transformed single-row feature matrix.
    pipeline : sklearn Pipeline
        The loaded artifact pipeline (preprocess + classifier).
    """
    from app.services import model_service

    feature_names = model_service.get_transformed_feature_names(pipeline)
    explainer = model_service.get_explainer(pipeline)
    row_shap = np.asarray(explainer.shap_values(X_row))[0]

    evidence = []
    for i, fname in enumerate(feature_names):
        kind, col, category = decode_feature(fname)
        value = float(X_row[0, i])
        if kind == "numerical":
            value_display = str(features_dict[col])
        elif kind == "categorical":
            value_display = category if value == 1.0 else f"not {category}"
        else:
            value_display = str(value)
        shap_value = float(row_shap[i])
        direction = "toward churn" if shap_value >= 0 else "away from churn"
        evidence.append(
            {
                "feature": fname,
                "original_feature": col,
                "value": value,
                "value_display": value_display,
                "shap_value": shap_value,
                "direction": direction,
                "abs_shap": abs(shap_value),
            }
        )

    evidence.sort(key=lambda e: e["abs_shap"], reverse=True)
    return evidence