"""Persistence, checksum, and prediction logic for the validated XGBoost pipeline.

The model artifact is the authoritative predictive model. It is loaded once at
startup and is never retrained here. The artifact SHA-256 checksum is verified
before the pipeline is used.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services import shap_service

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODEL_DIR / "telco_churn_xgboost.joblib"
METADATA_PATH = MODEL_DIR / "telco_churn_xgboost_metadata.json"
HASH_PATH = MODEL_DIR / "telco_churn_xgboost.joblib.sha256"

_pipeline = None
_metadata = None
_explainer = None


def compute_file_hash(path, algorithm="sha256"):
    """Return the hex digest of a file, streaming to bound memory use."""
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact_checksum():
    """Verify the persisted artifact against its checksum file and metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not HASH_PATH.exists():
        raise FileNotFoundError(f"Checksum file not found: {HASH_PATH}")
    stored = HASH_PATH.read_text().strip()
    current = compute_file_hash(MODEL_PATH)
    if stored != current:
        raise RuntimeError(
            "Model artifact hash mismatch: the file may have been replaced."
        )
    metadata = load_metadata()
    if stored != metadata.get("sha256"):
        raise RuntimeError("Checksum does not match the metadata sha256 value.")
    return current


def load_metadata():
    """Load and cache the artifact metadata JSON."""
    global _metadata
    if _metadata is None:
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    return _metadata


def load_model():
    """Load the persisted pipeline once; fail loudly if integrity fails."""
    global _pipeline
    if _pipeline is None:
        verify_artifact_checksum()
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_pipeline():
    return load_model()


def get_metadata():
    return load_metadata()


def get_threshold():
    return float(load_metadata()["decision_threshold"])


def get_raw_feature_names(pipeline=None):
    """Raw (untransformed) feature names in the order the ColumnTransformer expects."""
    pipeline = pipeline or load_model()
    preprocessor = pipeline.named_steps["preprocess"]
    numerical = list(preprocessor.transformers_[0][2])
    categorical = list(preprocessor.transformers_[1][2])
    return numerical + categorical


def get_transformed_feature_names(pipeline=None):
    """Transformed feature names exactly as the model sees them."""
    pipeline = pipeline or load_model()
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def get_explainer(pipeline=None):
    """SHAP TreeExplainer for the persisted XGBoost estimator (cached)."""
    global _explainer
    if _explainer is None:
        import shap

        from xgboost import XGBClassifier

        pipeline = pipeline or load_model()
        estimator = pipeline.named_steps["classifier"]
        assert isinstance(estimator, XGBClassifier), "Unexpected estimator type"
        _explainer = shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
    return _explainer


def predict_probability(features_dict, pipeline=None):
    """Predict the churn probability for one raw customer feature dict.

    Returns (probability, predicted_class_int, transformed_row).
    """
    pipeline = pipeline or load_model()
    raw_names = get_raw_feature_names(pipeline)
    frame = pd.DataFrame([features_dict])[raw_names]
    X_row = pipeline.named_steps["preprocess"].transform(frame)
    proba = float(pipeline.named_steps["classifier"].predict_proba(X_row)[0, 1])
    threshold = get_threshold()
    predicted_class = int(proba >= threshold)
    return proba, predicted_class, X_row


def predict_and_explain(features_dict):
    """Full single-customer pipeline: predict + SHAP evidence.

    Returns a dict with probability, threshold, predicted_class (int),
    evidence (list, Notebook 04 schema), base_value.
    """
    pipeline = load_model()
    proba, predicted_class, X_row = predict_probability(features_dict, pipeline)
    evidence = shap_service.build_evidence(features_dict, proba, X_row, pipeline)
    return {
        "probability": proba,
        "threshold": get_threshold(),
        "predicted_class": predicted_class,
        "evidence": evidence,
        "base_value": float(get_explainer(pipeline).expected_value),
    }