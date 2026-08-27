"""LLM explanation layer.

The LLM is only a constrained verbalization layer. It receives a reduced
evidence set and its structured output is passed through the deterministic
verifier. If OPENAI_API_KEY is unavailable, no call is made and the API
reports ``llm_status="not_configured"`` with ``explanation=None``.

No API key is ever printed, logged, or stored.
"""

import json
import os

from app.services.verifier import verify_explanation

SYSTEM_PROMPT = (
    "You are an evidence-grounded churn explanation assistant.\n"
    "Your job is to verbalize the supplied model evidence, not to generate new "
    "evidence.\n"
    "Rules:\n"
    "1. Use ONLY the evidence provided in the user message.\n"
    "2. Do not invent customer attributes.\n"
    "3. Do not invent numerical values.\n"
    "4. Do not invent SHAP values.\n"
    "5. Do not introduce reasons absent from the evidence.\n"
    "6. Do not infer unprovided demographic, financial, geographic, "
    "behavioral, or account information.\n"
    "7. Do not claim that any feature caused churn.\n"
    "8. Explain features as factors that pushed the MODEL prediction toward or "
    "away from churn.\n"
    "9. Preserve the direction of each SHAP contribution exactly as supplied.\n"
    "10. If evidence is insufficient, explicitly say so.\n"
    "11. Do not cite external knowledge.\n"
    "12. Do not mention information that is not present in the evidence.\n"
    "13. Report predicted_probability and threshold exactly as supplied.\n"
    "14. Respond ONLY with a JSON object of this exact shape: "
    '{"risk_summary": string, "predicted_probability": number, '
    '"threshold": number, "predicted_class": string, '
    '"supporting_evidence": [{"feature": string, "direction": string, '
    '"shap_value": number, "explanation": string}], '
    '"countervailing_evidence": [{"feature": string, "direction": string, '
    '"shap_value": number, "explanation": string}], '
    '"limitations": string}.'
)


def llm_configured():
    return bool(os.environ.get("OPENAI_API_KEY"))


def get_model_name():
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def build_llm_payload(probability, threshold, predicted_class, positive, negative):
    """Construct the user message from evidence only (no ground truth, no metrics)."""
    return {
        "task": (
            "Explain the model's churn prediction for this customer using "
            "ONLY the evidence below."
        ),
        "predicted_probability": probability,
        "threshold": threshold,
        "predicted_class": predicted_class,
        "supporting_evidence": positive,
        "countervailing_evidence": negative,
    }


def generate_and_verify(probability, threshold, predicted_class,
                        positive, negative, authoritative_evidence,
                        supplied_features):
    """Call the LLM (if configured) and verify the structured response.

    Returns a dict with ``explanation`` (dict or None), ``llm_status`` and
    ``verification`` (dict or None).
    """
    if not llm_configured():
        return {
            "explanation": None,
            "llm_status": "not_configured",
            "verification": None,
        }

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    payload = build_llm_payload(
        probability, threshold, predicted_class, positive, negative
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    explanation = json.loads(response.choices[0].message.content)
    verification = verify_explanation(
        explanation, authoritative_evidence, supplied_features
    )
    return {
        "explanation": explanation,
        "llm_status": "configured",
        "verification": verification,
    }