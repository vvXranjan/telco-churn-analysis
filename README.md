# Telco Churn — Evidence-Grounded Prediction & Explanation System

An evidence-grounded telco churn prediction system that combines leakage-safe
XGBoost modeling, SHAP-based model evidence, and a constrained LLM explanation
layer with deterministic verification.

The pipeline takes customer attributes, predicts churn with a persisted,
reproducible XGBoost pipeline, attributes the prediction with SHAP, and — when
an LLM is configured — verbalizes that evidence under strict constraints. A
deterministic verifier independently checks every generated claim against the
authoritative SHAP evidence (PASS / FLAG). The LLM is never the source of truth.

## Architecture

```
Cleaned data
   → Notebook 02 — hypothesis-driven EDA (statistical evidence)
   → Notebook 03 — leakage-safe modeling (train/dev/holdout)
        → persists full pipeline artifact (models/)
   → Notebook 04 — SHAP explainability → structured evidence JSON
   → Notebook 05 — evidence-grounded LLM explanation
   → deterministic verifier → PASS / FLAG
        ↓
   App layer: FastAPI /predict + Gradio UI (same evidence path)
```

Trust hierarchy (never reversed):

1. **Model artifact** — authoritative predictive model
2. **SHAP** — authoritative model attribution evidence
3. **Evidence JSON** — structured evidence representation
4. **LLM** — natural-language verbalization layer
5. **Deterministic verifier** — validation gate

## Demo / API

Not deployed. Run locally:

```bash
# API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Gradio UI
python app/ui.py

# MLflow tracking (local SQLite store)
python scripts/track_experiment.py
```

`GET /health`, `GET /model-info`, and `POST /predict` are available. Without
`OPENAI_API_KEY`, `/predict` still returns the model prediction and SHAP
evidence with `"llm_status": "not_configured"` and `"explanation": null`.

## Deployment notes (verified locally)

A Docker image builds and runs the inference API locally (no cloud deployment
has been performed):

```bash
docker build -t telco-churn-api:latest .
docker run -p 8000:8000 telco-churn-api:latest
```

- The container loads the persisted model artifact at startup and verifies its
  SHA-256 checksum; it never retrains.
- `PORT` is respected as an environment variable (default 8000) and the server
  binds to `0.0.0.0`.
- `requirements-app.txt` holds the minimal runtime dependencies (the image does
  not install JupyterLab/Gradio/MLflow).
- Secrets are passed as environment variables (`OPENAI_API_KEY`,
  `OPENAI_MODEL`); none are baked into the image.
- A `HEALTHCHECK` polls `/health`.

Verified locally: `GET /health` → 200, `GET /model-info` → 200,
`POST /predict` (customer 38) → probability 0.7301 / class Churn / 45 SHAP
evidence items with `llm_status: not_configured`, invalid payload → 422.

## Status of LLM testing

No real OpenAI call has been executed in this repository yet
(`OPENAI_API_KEY` was unavailable). The full LLM path is implemented and gated
by a deterministic verifier; without a key, tests exercise a clearly labelled
deterministic mock (valid → PASS, fabricated adversarial → FLAG).

## Final holdout metrics

XGBoost at threshold **0.525**, holdout evaluated exactly once (1,052 rows):

| ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| 0.8530 | 0.6571 | 0.5369 | 0.7814 | 0.6365 | 0.7633 |

Confusion matrix: TN=585, FP=188, FN=61, TP=218 (FPR 0.2432, FNR 0.2186).
A majority-class baseline reaches 73.55% dev accuracy while detecting zero
churners — accuracy alone is not the goal.

## Evidence-grounded explanation architecture

- **SHAP** (`shap.TreeExplainer`, `tree_path_dependent`, 45 transformed
  features) generates per-customer attribution on the model's log-odds scale.
- The LLM receives only the **top 5 positive + top 5 negative** SHAP
  contributors, never the ground-truth label and never model-performance
  metrics.
- A **deterministic verifier** checks probability, threshold, feature names,
  SHAP values, directions, unsupported evidence, causal language, and
  ground-truth leakage. Valid responses → PASS; fabricated adversarial
  responses → FLAG (all violation classes are detected).
- Without an API key the full pipeline runs on a clearly labelled
  deterministic mock; no fabricated LLM output is ever produced.

## Repository structure

```
.
├── app/                    # FastAPI app + Gradio UI + services
│   ├── main.py
│   ├── schemas.py
│   ├── ui.py
│   └── services/           # model, shap, explainer, verifier, evidence
├── data/
│   ├── cleaned/telco_churn_clean.csv
│   └── evidence/example_customer_evidence.json
├── drift/                  # lightweight data-drift detection
├── models/                 # persisted pipeline + metadata + checksum
├── notebooks/              # 01 … 05 (cleaning → modeling → explanation)
├── scripts/track_experiment.py   # MLflow tracking of the validated run
├── tests/                  # pytest suite
├── .github/workflows/tests.yml
├── Dockerfile
├── MODEL_CARD.md
├── requirements.txt        # full local-development dependencies
├── requirements-app.txt    # minimal runtime dependencies (API image / CI)
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt   # full stack (notebooks, UI, MLflow)
python -m pytest -q               # run the test suite
uvicorn app.main:app --reload --port 8000
```

`POST /predict` accepts the original (untransformed) customer fields, e.g.:

```json
{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 34, "PhoneService": "Yes", "MultipleLines": "Yes",
  "InternetService": "Fiber optic", "OnlineSecurity": "No",
  "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "No",
  "StreamingTV": "Yes", "StreamingMovies": "Yes",
  "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 106.35,
  "TotalCharges": 3549.25
}
```

Returns probability, threshold, predicted class, the 45-feature SHAP evidence,
and (if configured) the verified LLM explanation.

## Testing

`pytest -q` runs 42 tests covering: artifact existence/integrity (SHA-256),
model-loading without retraining, reproduction of the Notebook 03 holdout
metrics, 45-feature transformed space, evidence schema, the deterministic
verifier (valid → PASS, adversarial → FLAG, plus each failure class), API
endpoints (health / model-info / predict / 422 handling), and drift detection.
Tests never call OpenAI and need no `OPENAI_API_KEY`.

CI (`.github/workflows/tests.yml`) runs the suite on push and pull_request with
Python 3.13. It installs `requirements-app.txt` plus `pytest` (the tests do not
require JupyterLab, Gradio, or MLflow), then runs `pytest -q` and a secrets
scan.

## MLflow

`scripts/track_experiment.py` logs the validated model configuration and
holdout metrics to a local SQLite MLflow store (`mlruns/mlflow.db`) so model
experiments are inspectable rather than existing only as notebook outputs.
MLflow tracks the configuration and metrics; it does not by itself provide
model governance.

## Limitations

- The deterministic verifier reduces and detects classes of unsupported
  claims; it does not mathematically guarantee hallucination-free language.
- Findings come from a single 7,032-customer dataset and may not generalize to
  another operator, market, period, or population.
- Predictive relationships do not establish causality.
- No demographic fairness analysis was possible (the data lacks appropriate
  demographic attributes); see `MODEL_CARD.md`.

## Notebooks

- **01** Data cleaning
- **02** Hypothesis-driven EDA (statistical evidence)
- **03** Leakage-safe modeling + persisted artifact
- **04** SHAP explainability → evidence JSON
- **05** Evidence-grounded LLM explanation + deterministic verification