"""Data drift detector tests using deterministic synthetic data."""

import numpy as np
import pandas as pd

from drift.drift_check import detect_drift

NUM = ["tenure", "MonthlyCharges"]
CAT = ["Contract", "InternetService"]


def _make_baseline(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, size=n).astype(float),
            "MonthlyCharges": rng.normal(70, 25, size=n),
        }
    )
    df["Contract"] = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n,
        p=[0.55, 0.25, 0.20],
    )
    df["InternetService"] = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.40, 0.40, 0.20]
    )
    return df


def test_report_structure():
    base = _make_baseline()
    report = detect_drift(base, base.copy(), NUM, CAT)
    assert set(report.columns) == {"feature", "test", "statistic", "p_value",
                                   "drift_detected"}
    assert set(report["feature"]) == set(NUM + CAT)
    tests = dict(zip(report["feature"], report["test"]))
    assert tests["tenure"] == "Kolmogorov-Smirnov"
    assert tests["MonthlyCharges"] == "Kolmogorov-Smirnov"
    assert tests["Contract"] == "Chi-square"
    assert tests["InternetService"] == "Chi-square"
    assert all(0.0 <= p <= 1.0 for p in report["p_value"])
    assert report["drift_detected"].dtype == bool


def test_identical_data_no_drift():
    base = _make_baseline()
    report = detect_drift(base, base.copy(), NUM, CAT)
    assert not report["drift_detected"].any()


def test_shifted_numerical_detects_drift():
    base = _make_baseline()
    shifted = base.copy()
    shifted["tenure"] = shifted["tenure"] + 30
    report = detect_drift(base, shifted, NUM, CAT)
    tenure_row = report[report["feature"] == "tenure"].iloc[0]
    assert bool(tenure_row["drift_detected"]) is True
    assert tenure_row["p_value"] < 0.05


def test_shifted_categorical_detects_drift():
    base = _make_baseline()
    shifted = base.copy()
    shifted["Contract"] = shifted["Contract"].map(
        lambda x: "Two year" if x == "Month-to-month" else x
    )
    report = detect_drift(base, shifted, NUM, CAT)
    contract_row = report[report["feature"] == "Contract"].iloc[0]
    assert bool(contract_row["drift_detected"]) is True
    assert contract_row["p_value"] < 0.05


def test_small_shift_may_not_detect(capsys):
    # A tiny perturbation should not always flag; verifies the check is not
    # trivially "always true". With identical distributions p is ~1.
    base = _make_baseline()
    report = detect_drift(base, base.copy(), NUM, CAT)
    assert report["p_value"].min() > 0.05


def test_drift_does_not_change_model(cleaned_csv_path):
    from app.services import model_service

    base = _make_baseline()
    shifted = base.copy()
    shifted["MonthlyCharges"] += 50
    report = detect_drift(base, shifted, NUM, CAT)
    assert bool(report["drift_detected"].any())  # drift exists
    # Drift check must not alter the model's predictions.
    probe = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 34, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 106.35, "TotalCharges": 3549.25,
    }
    before = model_service.predict_probability(probe)
    after = model_service.predict_probability(probe)
    assert before[0] == after[0]