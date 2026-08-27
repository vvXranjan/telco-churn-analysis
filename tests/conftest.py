"""Shared pytest fixtures.

The autouse fixture guarantees tests never hit the network: no OPENAI_API_KEY
is present during the test session, so the API/verifier follow the
"not_configured" path and no LLM call is made.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "telco_churn_xgboost.joblib"
METADATA_PATH = MODEL_DIR / "telco_churn_xgboost_metadata.json"
HASH_PATH = MODEL_DIR / "telco_churn_xgboost.joblib.sha256"
CLEANED_CSV = ROOT / "data" / "cleaned" / "telco_churn_clean.csv"
EVIDENCE_JSON = ROOT / "data" / "evidence" / "example_customer_evidence.json"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)


@pytest.fixture(scope="session")
def project_root():
    return ROOT


@pytest.fixture(scope="session")
def model_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def evidence_json():
    with open(EVIDENCE_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def cleaned_csv_path():
    return CLEANED_CSV


@pytest.fixture(scope="session")
def customer_38_features(cleaned_csv_path):
    df = pd.read_csv(cleaned_csv_path).drop_duplicates(keep="first")
    row = df.loc[38]
    return row.drop(labels=["Churn"]).to_dict()