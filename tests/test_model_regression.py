"""Regression tests: the loaded artifact must reproduce the validated
Notebook 03 holdout results and Notebook 04 SHAP outputs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.services import model_service


def _recompute_holdout(cleaned_csv_path, random_state=42):
    df = pd.read_csv(cleaned_csv_path).drop_duplicates(keep="first")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    pipeline = model_service.load_model()
    proba = pipeline.predict_proba(X_hold)[:, 1]
    threshold = model_service.get_threshold()
    pred = (proba >= threshold).astype(int)
    return (
        roc_auc_score(y_hold, proba),
        f1_score(y_hold, pred),
        confusion_matrix(y_hold, pred).ravel(),
    )


def test_artifact_reproduces_nb03_holdout(cleaned_csv_path, model_metadata):
    roc, f1, cm = _recompute_holdout(cleaned_csv_path)
    meta = model_metadata["holdout_evaluation"]
    assert np.isclose(roc, meta["roc_auc"], atol=1e-4)
    assert np.isclose(f1, meta["f1"], atol=1e-4)
    assert list(cm) == [int(x) for x in meta["confusion_matrix"]]
    # Documented Notebook 03 values.
    assert abs(roc - 0.8530) < 1e-4
    assert abs(f1 - 0.6365) < 1e-4
    assert list(cm) == [585, 188, 61, 218]


def test_example_customer_probability(customer_38_features):
    proba, cls, _ = model_service.predict_probability(customer_38_features)
    assert np.isclose(proba, 0.7301, atol=1e-4)
    assert cls == 1


def test_example_customer_local_shap(customer_38_features, evidence_json):
    result = model_service.predict_and_explain(customer_38_features)
    evidence = result["evidence"]
    assert len(evidence) == 45
    by_feature = {e["feature"]: e["shap_value"] for e in evidence}
    expected = {e["feature"]: e["shap_value"] for e in evidence_json["evidence"]}
    assert by_feature == expected  # identical SHAP values to Notebook 04
    # Top positive contributors remain consistent.
    assert np.isclose(by_feature["numerical__MonthlyCharges"], 0.5016, atol=1e-4)
    assert np.isclose(
        by_feature["categorical__Contract_Month-to-month"], 0.3872, atol=1e-4
    )
    # Top negative contributors remain consistent.
    assert np.isclose(by_feature["numerical__tenure"], -0.4771, atol=1e-4)
    assert np.isclose(by_feature["numerical__TotalCharges"], -0.4243, atol=1e-4)