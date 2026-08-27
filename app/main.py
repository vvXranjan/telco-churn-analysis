"""Telco churn inference API.

Trust hierarchy (never reversed):
1. model artifact = authoritative predictive model
2. SHAP = authoritative model attribution evidence
3. evidence JSON / evidence list = structured evidence representation
4. LLM = natural-language verbalization layer
5. deterministic verifier = validation gate
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import CustomerFeatures
from app.services import explainer_service, model_service
from app.services.evidence import reduce_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("telco_churn.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail startup loudly if the artifact cannot be loaded or the checksum fails.
    model_service.load_model()
    logger.info("Model loaded and checksum verified: %s", model_service.MODEL_PATH.name)
    yield


app = FastAPI(
    title="Telco Churn Inference API",
    description=(
        "Evidence-grounded churn prediction: persisted XGBoost pipeline + SHAP "
        "evidence + optional constrained LLM explanation with deterministic "
        "verification."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=dict)
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=dict)
def model_info():
    metadata = model_service.get_metadata()
    return {
        "model": metadata["model_name"],
        "threshold": float(metadata["decision_threshold"]),
        "feature_count": int(metadata["transformed_feature_count"]),
    }


@app.post("/predict", response_model=dict)
def predict(features: CustomerFeatures):
    try:
        logger.info("Prediction request received")
        features_dict = features.model_dump()

        result = model_service.predict_and_explain(features_dict)
        probability = result["probability"]
        threshold = result["threshold"]
        predicted_class = result["predicted_class"]
        evidence = result["evidence"]

        predicted_class_label = "Churn" if predicted_class == 1 else "Retained"

        # Reduced evidence supplied to the LLM (strongest contributors only).
        positive, negative = reduce_evidence(evidence, k=5)
        supplied_features = {e["feature"] for e in positive + negative}

        authoritative = {
            "predicted_probability": probability,
            "threshold": threshold,
            "predicted_class": predicted_class,
            "features": [
                {
                    "feature": e["feature"],
                    "shap_value": e["shap_value"],
                    "direction": e["direction"],
                }
                for e in evidence
            ],
        }

        llm_result = explainer_service.generate_and_verify(
            probability,
            threshold,
            predicted_class_label,
            positive,
            negative,
            authoritative,
            supplied_features,
        )
        explanation = llm_result["explanation"]
        llm_status = llm_result["llm_status"]
        verification = llm_result["verification"]
        if llm_status == "configured":
            logger.info("Verification result: %s", verification["status"])

        response = {
            "predicted_probability": probability,
            "threshold": threshold,
            "predicted_class": predicted_class_label,
            "shap_evidence": evidence,
            "explanation": explanation,
            "llm_status": llm_status,
            "verification": verification,
        }
        logger.info("Prediction completed: class=%s proba=%.4f", predicted_class_label, probability)
        return response
    except Exception as exc:  # noqa: BLE001 - keep stack traces out of responses
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc