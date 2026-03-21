"""
core/risk.py
Risk Aggregation Layer

Combines the anomaly detector score and fuzzy inference result
into a single final risk score and label, then writes it back
onto the Employee objects.
"""

from core.features import BehavioralFingerprint, build_fingerprints
from core.fuzzy    import fuzzy_inference, explain


# ---------------------------------------------------------------------------
# Weights (tune freely)
# ---------------------------------------------------------------------------

ANOMALY_WEIGHT = 0.4   # how much the raw IF score contributes
FUZZY_WEIGHT   = 0.6   # how much the fuzzy score contributes


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_risk(fingerprint: BehavioralFingerprint,
                 anomaly_score: float) -> dict:
    """
    Compute the full risk result for one employee snapshot.

    Parameters
    ----------
    fingerprint   : BehavioralFingerprint
    anomaly_score : float  [0, 1] — from AnomalyDetector

    Returns
    -------
    dict with keys:
        final_score   : float   [0, 1]
        risk_label    : str     "Low" | "Medium" | "High"
        anomaly_score : float
        fuzzy_score   : float
        fuzzy_label   : str
        fired_rules   : list
        memberships   : dict
        explanation   : str
    """
    fuzzy_result = fuzzy_inference(
        anomaly_score       = anomaly_score,
        deviation_score     = fingerprint.deviation_score,
    )

    final_score = (ANOMALY_WEIGHT * anomaly_score
                   + FUZZY_WEIGHT  * fuzzy_result["fuzzy_score"])

    # Use the fuzzy label as the primary label (more explainable)
    risk_label = fuzzy_result["risk_label"]

    return {
        "employee_id":  fingerprint.employee_id,
        "role":         fingerprint.role,
        "final_score":  round(final_score, 4),
        "risk_label":   risk_label,
        "anomaly_score": round(anomaly_score, 4),
        "fuzzy_score":  fuzzy_result["fuzzy_score"],
        "fuzzy_label":  fuzzy_result["risk_label"],
        "deviation_score": round(fingerprint.deviation_score, 4),
        "fired_rules":  fuzzy_result["fired_rules"],
        "memberships":  fuzzy_result["memberships"],
        "explanation":  explain(fuzzy_result),
        # Per-feature breakdown for radar chart
        "baseline":     fingerprint.baseline_dict,
        "current":      fingerprint.current_dict,
        "deviation":    fingerprint.deviation_dict,
    }


# ---------------------------------------------------------------------------
# Batch update — main entry point called each simulation step
# ---------------------------------------------------------------------------

def update_risk_scores(employees, detector) -> list[dict]:
    """
    For every employee:
      1. Build fingerprint from current behaviour
      2. Get anomaly score from detector
      3. Run risk aggregation
      4. Write risk_score + risk_label + risk_history back onto Employee

    Returns list of risk result dicts (one per employee).
    """
    fingerprints = build_fingerprints(employees)
    results = []

    for emp, fp in zip(employees, fingerprints):
        anomaly_score = detector.score(fp)
        risk_result   = compute_risk(fp, anomaly_score)

        # Write back onto the employee object
        emp.anomaly_score = anomaly_score
        emp.risk_score    = risk_result["final_score"]
        emp.risk_label    = risk_result["risk_label"]
        emp.risk_history.append(risk_result["final_score"])

        results.append(risk_result)

    return results


# ---------------------------------------------------------------------------
# Helper: threshold flags
# ---------------------------------------------------------------------------

ALERT_THRESHOLD = 0.65   # final_score above this triggers an alert

def get_alerts(risk_results: list[dict]) -> list[dict]:
    """Return only the employees whose final score exceeds the alert threshold."""
    return [r for r in risk_results if r["final_score"] >= ALERT_THRESHOLD]