"""Deterministic verification of generated explanations against authoritative evidence.

This module is a direct extraction of the verifier defined in
``notebooks/05_evidence_grounded_churn_explainer.ipynb``. Behaviour is unchanged:
the LLM is only a constrained verbalization layer and this verifier is the final
gate that decides PASS / FLAG.

The verifier is intentionally lightweight. It validates structured claims and
known feature references; it does not provide formal semantic proof that every
sentence is truthful.
"""

import numpy as np

CAUSAL_PATTERNS = [
    "caused churn",
    "causes churn",
    "cause churn",
    "caused the customer to churn",
    "will cause churn",
    "led to churn",
    "because of",
]

LEAKAGE_PATTERNS = [
    "actually churned",
    "did in fact churn",
    "did churn",
    "true label",
    "ground truth",
    "was correct",
    "correctly predicted",
]

ALLOWED_DIRECTIONS = {"toward churn", "away from churn"}


def direction_of(shap_value):
    """Authoritative rule: SHAP > 0 -> toward churn; SHAP < 0 -> away from churn."""
    return "toward churn" if shap_value >= 0 else "away from churn"


def verify_explanation(generated, authoritative, supplied_features):
    """Deterministically check a generated structured explanation against the
    authoritative evidence.

    Parameters
    ----------
    generated : dict
        The structured response with keys ``risk_summary``,
        ``predicted_probability``, ``threshold``, ``predicted_class``,
        ``supporting_evidence``, ``countervailing_evidence``, ``limitations``.
        Each evidence item has ``feature``, ``direction``, ``shap_value`` and
        optionally ``explanation``.
    authoritative : dict
        Canonical evidence with ``predicted_probability``, ``threshold``,
        ``predicted_class`` and ``features`` (list of dicts with ``feature``
        and ``shap_value``).
    supplied_features : set
        The set of features actually supplied to the LLM.

    Returns
    -------
    dict
        {"status": "PASS" | "FLAG", "checks": {...}, "flags": [...]}
    """
    checks = {}
    flags = []

    def record(check_name, ok, reason):
        checks[check_name] = bool(ok)
        if not ok:
            flags.append(reason)

    auth_by_feature = {f["feature"]: f for f in authoritative["features"]}
    expected_class = "Churn" if authoritative["predicted_class"] == 1 else "Retained"

    # D. Probability grounding
    gen_prob = generated.get("predicted_probability")
    ok = (
        isinstance(gen_prob, (int, float))
        and np.isfinite(gen_prob)
        and np.isclose(float(gen_prob), authoritative["predicted_probability"], atol=1e-6)
    )
    record(
        "probability_grounded",
        ok,
        f"Fabricated probability: got {gen_prob!r}, authoritative "
        f"{authoritative['predicted_probability']}",
    )

    # E. Threshold grounding
    gen_thr = generated.get("threshold")
    ok = (
        isinstance(gen_thr, (int, float))
        and np.isfinite(gen_thr)
        and np.isclose(float(gen_thr), authoritative["threshold"], atol=1e-6)
    )
    record(
        "threshold_grounded",
        ok,
        f"Fabricated threshold: got {gen_thr!r}, authoritative "
        f"{authoritative['threshold']}",
    )

    # Predicted-class grounding
    ok = generated.get("predicted_class") == expected_class
    record(
        "predicted_class_grounded",
        ok,
        f"Predicted class mismatch: got {generated.get('predicted_class')!r}, "
        f"expected {expected_class!r}",
    )

    items = list(generated.get("supporting_evidence", [])) + list(
        generated.get("countervailing_evidence", [])
    )

    # A/F. Feature grounding + unsupported evidence
    features_grounded = True
    no_unsupported = True
    for item in items:
        fname = item.get("feature")
        if fname not in auth_by_feature:
            features_grounded = False
            flags.append(f"Unknown feature referenced: {fname!r}")
        if fname not in supplied_features:
            no_unsupported = False
            flags.append(
                f"Unsupported evidence: {fname!r} not in the supplied evidence set"
            )
    checks["features_grounded"] = bool(features_grounded)
    checks["no_unsupported_evidence"] = bool(no_unsupported)

    # B. SHAP value grounding
    shap_grounded = True
    for item in items:
        fname = item.get("feature")
        if fname not in auth_by_feature:
            continue
        auth = auth_by_feature[fname]
        gen_sv = item.get("shap_value")
        ok = (
            isinstance(gen_sv, (int, float))
            and np.isfinite(gen_sv)
            and np.isclose(float(gen_sv), auth["shap_value"], atol=1e-6)
        )
        if not ok:
            shap_grounded = False
            flags.append(
                f"Fabricated SHAP for {fname!r}: got {gen_sv!r}, "
                f"authoritative {auth['shap_value']:.6f}"
            )
    checks["shap_values_grounded"] = bool(shap_grounded)

    # C. Direction grounding
    directions_ok = True
    for item in items:
        fname = item.get("feature")
        if fname not in auth_by_feature:
            continue
        expected = direction_of(auth_by_feature[fname]["shap_value"])
        if item.get("direction") != expected:
            directions_ok = False
            flags.append(
                f"Wrong direction for {fname!r}: got {item.get('direction')!r}, "
                f"expected {expected!r}"
            )
    checks["directions_consistent"] = bool(directions_ok)

    # G. Causal language
    texts = [
        str(generated.get("risk_summary", "")),
        str(generated.get("limitations", "")),
    ]
    texts += [str(item.get("explanation", "")) for item in items]
    blob = " ".join(texts).lower()
    causal_hits = [p for p in CAUSAL_PATTERNS if p in blob]
    checks["no_causal_claim"] = not causal_hits
    if causal_hits:
        flags.append(f"Potential causal claim detected: {causal_hits}")

    # H. Ground-truth leakage
    leak_hits = [p for p in LEAKAGE_PATTERNS if p in blob]
    checks["no_ground_truth_leakage"] = not leak_hits
    if leak_hits:
        flags.append(f"Potential ground-truth leakage detected: {leak_hits}")

    status = "PASS" if not flags else "FLAG"
    return {"status": status, "checks": checks, "flags": flags}