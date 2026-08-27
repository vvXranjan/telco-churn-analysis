"""Lightweight data drift detection.

Compares a current (inference-time) batch of customer features against a
baseline (training) distribution.

- Numerical features: two-sample Kolmogorov-Smirnov test.
- Categorical features: chi-square test on category counts between baseline
  and current.

The check produces a structured report with ``feature``, ``test``,
``statistic``, ``p_value`` and ``drift_detected``.

Important distinction: statistical significance is NOT the same as operational
importance. A low p-value means the distributions differ more than expected by
chance; with large samples even tiny, practically irrelevant shifts become
"significant". The report should therefore be read alongside the magnitude of
the statistic and business context, not treated as an automatic alarm.

The drift check never retrains the model and never alters predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

DEFAULT_ALPHA = 0.05


def detect_drift(baseline, current, numerical_features, categorical_features,
                 alpha=DEFAULT_ALPHA):
    """Return a DataFrame report of per-feature drift tests.

    Parameters
    ----------
    baseline : pandas.DataFrame
        Reference distribution (e.g. the training set).
    current : pandas.DataFrame
        Current / incoming batch.
    numerical_features : list of str
    categorical_features : list of str
    alpha : float
        Significance threshold for ``drift_detected``.

    Returns
    -------
    pandas.DataFrame
        Columns: feature, test, statistic, p_value, drift_detected.
    """
    rows = []

    for feat in numerical_features:
        if feat not in baseline.columns or feat not in current.columns:
            continue
        base = pd.to_numeric(baseline[feat], errors="coerce").dropna()
        curr = pd.to_numeric(current[feat], errors="coerce").dropna()
        if len(base) == 0 or len(curr) == 0:
            continue
        statistic, p_value = ks_2samp(base, curr)
        rows.append(
            {
                "feature": feat,
                "test": "Kolmogorov-Smirnov",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < alpha),
            }
        )

    for feat in categorical_features:
        if feat not in baseline.columns or feat not in current.columns:
            continue
        base = baseline[feat].astype(str).fillna("__missing__")
        curr = current[feat].astype(str).fillna("__missing__")
        categories = sorted(set(base).union(set(curr)))
        base_counts = base.value_counts().reindex(categories, fill_value=0)
        current_counts = curr.value_counts().reindex(categories, fill_value=0)
        table = pd.DataFrame({"baseline": base_counts, "current": current_counts})
        if table["current"].sum() == 0:
            continue
        # Expected counts are never zero here (every row/column total > 0).
        statistic, p_value, dof, expected = chi2_contingency(table.values)
        rows.append(
            {
                "feature": feat,
                "test": "Chi-square",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(p_value < alpha),
            }
        )

    return pd.DataFrame(rows)


def get_training_baseline(cleaned_csv_path, random_state=42):
    """Re-derive the exact training feature set used by Notebook 03.

    Reuses the validated methodology: load the cleaned CSV, drop the 22 exact
    duplicate rows from a modeling copy, then apply the same stratified
    70/15/15 split with ``random_state=42``. Returns the 70% training features.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(cleaned_csv_path)
    df_model = df.drop_duplicates(keep="first")
    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state
    )
    return X_train