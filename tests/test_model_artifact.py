"""Model artifact existence, integrity, and structure tests."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services import model_service

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "telco_churn_xgboost.joblib"
HASH_PATH = ROOT / "models" / "telco_churn_xgboost.joblib.sha256"
METADATA_PATH = ROOT / "models" / "telco_churn_xgboost_metadata.json"


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_artifact_exists():
    assert MODEL_PATH.exists(), "model artifact missing"


def test_artifact_loads():
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict_proba")


def test_sha256_matches():
    stored = HASH_PATH.read_text().strip()
    current = _file_hash(MODEL_PATH)
    assert current == stored, "artifact hash differs from checksum file"
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    assert stored == metadata["sha256"], "artifact hash differs from metadata sha256"


def test_metadata_valid(model_metadata):
    assert model_metadata["model_name"] == "XGBoost"
    assert model_metadata["random_state"] == 42
    assert abs(float(model_metadata["decision_threshold"]) - 0.525) < 1e-6
    assert model_metadata["training_row_count"] == 4907
    assert model_metadata["dev_row_count"] == 1051
    assert model_metadata["holdout_row_count"] == 1052
    assert model_metadata["feature_count_before_preprocessing"] == 19
    assert model_metadata["transformed_feature_count"] == 45
    assert "selected_hyperparameters" in model_metadata
    hp = model_metadata["selected_hyperparameters"]
    assert hp["learning_rate"] == 0.05 and hp["max_depth"] == 3
    assert hp["n_estimators"] == 200 and hp["subsample"] == 0.8
    assert "sha256" in model_metadata


def test_pipeline_contains_preprocessing_and_xgboost():
    from sklearn.compose import ColumnTransformer
    from xgboost import XGBClassifier

    pipeline = model_service.load_model()
    assert "preprocess" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["preprocess"], ColumnTransformer)
    assert isinstance(pipeline.named_steps["classifier"], XGBClassifier)


def test_transformed_feature_count_is_45():
    names = model_service.get_transformed_feature_names()
    assert len(names) == 45
    assert len(set(names)) == 45


def test_threshold_is_0525(model_metadata):
    assert float(model_metadata["decision_threshold"]) == 0.525


def test_predictions_are_valid(cleaned_csv_path):
    df = pd.read_csv(cleaned_csv_path).drop_duplicates(keep="first")
    sample = df.head(10)
    pipeline = model_service.load_model()
    for _, row in sample.iterrows():
        features = row.drop(labels=["Churn"]).to_dict()
        proba, cls, _ = model_service.predict_probability(features, pipeline)
        assert np.isfinite(proba) and 0.0 <= proba <= 1.0
        assert cls in (0, 1)