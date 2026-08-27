"""Minimal Gradio UI for the evidence-grounded churn explainer.

The UI clearly separates MODEL PREDICTION, SHAP EVIDENCE, LLM EXPLANATION and
VERIFICATION. LLM text is never presented as ground truth; when the API is not
configured the UI says the explanation is unavailable while keeping the model
and SHAP evidence available.
"""

import sys
from pathlib import Path

# Allow running directly as `python app/ui.py` (repo root on sys.path).
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import explainer_service, model_service, shap_service
from app.services.evidence import reduce_evidence


def build_report(features):
    """Build the Markdown report for a raw customer feature dict."""
    result = model_service.predict_and_explain(features)
    probability = result["probability"]
    threshold = result["threshold"]
    predicted_class = "Churn" if result["predicted_class"] == 1 else "Retained"
    evidence = result["evidence"]

    positive, negative = reduce_evidence(evidence, k=5)
    supplied_features = {e["feature"] for e in positive + negative}

    authoritative = {
        "predicted_probability": probability,
        "threshold": threshold,
        "predicted_class": result["predicted_class"],
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
        predicted_class,
        positive,
        negative,
        authoritative,
        supplied_features,
    )
    explanation = llm_result["explanation"]
    llm_status = llm_result["llm_status"]
    verification = llm_result["verification"]

    lines = []
    lines.append("## MODEL PREDICTION")
    lines.append(f"- Predicted churn probability: **{probability:.4f}** ({probability:.2%})")
    lines.append(f"- Decision threshold: **{threshold:.3f}**")
    lines.append(f"- Predicted class: **{predicted_class}**")
    lines.append("")
    lines.append("## SHAP EVIDENCE (top factors)")
    lines.append("| Feature | Value | SHAP | Direction |")
    lines.append("|---|---|---|---|")
    for e in evidence[:8]:
        lines.append(
            f"| {e['feature']} | {e['value_display']} | {e['shap_value']:+.4f} "
            f"| {e['direction']} |"
        )
    lines.append("")
    lines.append("### Factors pushing **toward** churn")
    for e in positive:
        lines.append(f"- {e['feature']} ({e['value_display']}): SHAP {e['shap_value']:+.4f}")
    lines.append("")
    lines.append("### Factors pushing **away** from churn")
    for e in negative:
        lines.append(f"- {e['feature']} ({e['value_display']}): SHAP {e['shap_value']:+.4f}")
    lines.append("")
    lines.append("## LLM EXPLANATION")
    if llm_status == "configured" and explanation is not None:
        lines.append(explanation.get("risk_summary", ""))
        for e in explanation.get("supporting_evidence", []):
            lines.append(f"- {e['feature']}: {e.get('explanation', '')}")
    else:
        lines.append("LLM explanation unavailable — model and SHAP evidence are still available.")
    lines.append("")
    lines.append("## VERIFICATION")
    if verification is None:
        lines.append("LLM not configured — no generated text to verify.")
    else:
        lines.append(f"Status: **{verification['status']}**")
        if verification["flags"]:
            for flag in verification["flags"]:
                lines.append(f"- {flag}")
    return "\n".join(lines)


def gradio_interface():
    """Build the Gradio interface (imports gradio lazily to keep imports light)."""
    import gradio as gr

    inputs = [
        gr.Textbox(label="gender"),
        gr.Number(label="SeniorCitizen", precision=0),
        gr.Textbox(label="Partner"),
        gr.Textbox(label="Dependents"),
        gr.Number(label="tenure", precision=0),
        gr.Textbox(label="PhoneService"),
        gr.Textbox(label="MultipleLines"),
        gr.Textbox(label="InternetService"),
        gr.Textbox(label="OnlineSecurity"),
        gr.Textbox(label="OnlineBackup"),
        gr.Textbox(label="DeviceProtection"),
        gr.Textbox(label="TechSupport"),
        gr.Textbox(label="StreamingTV"),
        gr.Textbox(label="StreamingMovies"),
        gr.Textbox(label="Contract"),
        gr.Textbox(label="PaperlessBilling"),
        gr.Textbox(label="PaymentMethod"),
        gr.Number(label="MonthlyCharges"),
        gr.Number(label="TotalCharges"),
    ]
    field_names = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]

    def predict_fn(*values):
        features = dict(zip(field_names, values))
        return build_report(features)

    return gr.Interface(
        fn=predict_fn,
        inputs=inputs,
        outputs=gr.Markdown(),
        title="Telco Churn — Evidence-Grounded Explainer",
        description=(
            "Enter a customer's attributes. The model prediction and SHAP "
            "evidence are always shown; the LLM explanation appears only when "
            "the API is configured."
        ),
    )


if __name__ == "__main__":
    gradio_interface().launch()