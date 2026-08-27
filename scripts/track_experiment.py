"""MLflow experiment tracking for the validated XGBoost model.

This script does NOT retrain the model. It loads the persisted artifact and its
metadata (produced by Notebook 03) and logs the validated configuration and
holdout metrics to a local MLflow tracking store so the experiment is
inspectable rather than existing only as notebook outputs.

No API keys, secrets, customer PII, or raw datasets are logged.

Usage:
    python scripts/track_experiment.py
"""

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "telco_churn_xgboost.joblib"
METADATA_PATH = MODEL_DIR / "telco_churn_xgboost_metadata.json"
MLRUNS_DIR = ROOT / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)
TRACKING_URI = "sqlite:///" + str(MLRUNS_DIR / "mlflow.db")

DEFAULT_EXPERIMENT = "telco_churn_xgboost"


def main(experiment_name=DEFAULT_EXPERIMENT):
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    holdout = metadata["holdout_evaluation"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgboost-validated") as run:
        mlflow.log_params(
            {
                "model": metadata["model_name"],
                "model_type": metadata["model_type"],
                "random_state": metadata["random_state"],
                "learning_rate": metadata["selected_hyperparameters"]["learning_rate"],
                "max_depth": metadata["selected_hyperparameters"]["max_depth"],
                "n_estimators": metadata["selected_hyperparameters"]["n_estimators"],
                "subsample": metadata["selected_hyperparameters"]["subsample"],
                "scale_pos_weight": metadata["scale_pos_weight"],
                "decision_threshold": metadata["decision_threshold"],
                "train_row_count": metadata["training_row_count"],
                "dev_row_count": metadata["dev_row_count"],
                "holdout_row_count": metadata["holdout_row_count"],
                "feature_count_before_preprocessing": metadata[
                    "feature_count_before_preprocessing"
                ],
                "transformed_feature_count": metadata["transformed_feature_count"],
                "xgboost_version": metadata["package_versions"]["xgboost"],
                "scikit_learn_version": metadata["package_versions"]["scikit-learn"],
            }
        )
        mlflow.log_metrics(
            {
                "roc_auc": holdout["roc_auc"],
                "pr_auc": holdout["pr_auc"],
                "precision": holdout["precision"],
                "recall": holdout["recall"],
                "f1": holdout["f1"],
                "accuracy": holdout["accuracy"],
            }
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="model")

    print(f"Logged run {run.info.run_id} to {TRACKING_URI}")
    print("Experiment:", experiment_name)
    print("Tracked holdout metrics: ROC-AUC=%.4f F1=%.4f PR-AUC=%.4f"
          % (holdout["roc_auc"], holdout["f1"], holdout["pr_auc"]))


if __name__ == "__main__":
    main()