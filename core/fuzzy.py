"""
core/fuzzy.py
Fuzzy Logic Inference Layer

Converts numerical anomaly score + deviation score into linguistic
risk categories using trapezoidal membership functions and a
Mamdani-style rule base.

No external fuzzy library required — pure NumPy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Membership functions (trapezoidal)
# ---------------------------------------------------------------------------

def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    Rises from 0→1 between a and b, stays 1 between b and c,
    falls from 1→0 between c and d.
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    # c < x < d
    return (d - x) / (d - c)


# ---------------------------------------------------------------------------
# Linguistic variable: Anomaly Score  (input 1)
# ---------------------------------------------------------------------------

def anomaly_low(score: float) -> float:
    return trapezoid(score, -0.1, 0.0, 0.25, 0.45)

def anomaly_medium(score: float) -> float:
    return trapezoid(score, 0.30, 0.45, 0.55, 0.70)

def anomaly_high(score: float) -> float:
    return trapezoid(score, 0.55, 0.70, 1.0, 1.1)


# ---------------------------------------------------------------------------
# Linguistic variable: Deviation Score  (input 2)
# ---------------------------------------------------------------------------

def deviation_low(score: float) -> float:
    return trapezoid(score, -0.1, 0.0, 0.10, 0.20)

def deviation_medium(score: float) -> float:
    return trapezoid(score, 0.10, 0.20, 0.30, 0.40)

def deviation_high(score: float) -> float:
    return trapezoid(score, 0.30, 0.40, 1.0, 1.1)


# ---------------------------------------------------------------------------
# Rule base
# Returns: (risk_label, fuzzy_score, fired_rules)
# ---------------------------------------------------------------------------

RULES = [
    # (anomaly_fn, deviation_fn, risk_label, crisp_output)
    (anomaly_low,    deviation_low,    "Low",    0.10),
    (anomaly_low,    deviation_medium, "Low",    0.20),
    (anomaly_low,    deviation_high,   "Medium", 0.45),
    (anomaly_medium, deviation_low,    "Low",    0.25),
    (anomaly_medium, deviation_medium, "Medium", 0.50),
    (anomaly_medium, deviation_high,   "High",   0.75),
    (anomaly_high,   deviation_low,    "Medium", 0.50),
    (anomaly_high,   deviation_medium, "High",   0.80),
    (anomaly_high,   deviation_high,   "High",   0.95),
]


def fuzzy_inference(anomaly_score: float,
                    deviation_score: float) -> dict:
    """
    Run Mamdani fuzzy inference.

    Parameters
    ----------
    anomaly_score   : float  [0, 1]  — from AnomalyDetector
    deviation_score : float  [0, 1]  — from BehavioralFingerprint

    Returns
    -------
    dict with keys:
        fuzzy_score   : float  [0, 1]  — weighted centroid output
        risk_label    : str            — "Low" | "Medium" | "High"
        fired_rules   : list[dict]     — for explainability
        memberships   : dict           — raw membership degrees
    """
    memberships = {
        "anomaly_low":      anomaly_low(anomaly_score),
        "anomaly_medium":   anomaly_medium(anomaly_score),
        "anomaly_high":     anomaly_high(anomaly_score),
        "deviation_low":    deviation_low(deviation_score),
        "deviation_medium": deviation_medium(deviation_score),
        "deviation_high":   deviation_high(deviation_score),
    }

    fired_rules = []
    weighted_sum = 0.0
    weight_total = 0.0

    for (a_fn, d_fn, label, crisp) in RULES:
        strength = min(a_fn(anomaly_score), d_fn(deviation_score))   # AND = min
        if strength > 0.0:
            fired_rules.append({
                "rule": f"IF anomaly={a_fn.__name__.split('_',1)[1].upper()} "
                        f"AND deviation={d_fn.__name__.split('_',1)[1].upper()} "
                        f"→ Risk={label}",
                "strength": round(strength, 3),
                "label": label,
            })
            weighted_sum += strength * crisp
            weight_total += strength

    fuzzy_score = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    # Map numeric score to categorical label
    if fuzzy_score < 0.35:
        risk_label = "Low"
    elif fuzzy_score < 0.65:
        risk_label = "Medium"
    else:
        risk_label = "High"

    return {
        "fuzzy_score":  round(fuzzy_score, 4),
        "risk_label":   risk_label,
        "fired_rules":  fired_rules,
        "memberships":  {k: round(v, 3) for k, v in memberships.items()},
    }


# ---------------------------------------------------------------------------
# Convenience: explain in plain English
# ---------------------------------------------------------------------------

def explain(result: dict) -> str:
    """
    Generate a one-paragraph plain-English explanation of a
    fuzzy inference result.
    """
    top_rule = max(result["fired_rules"], key=lambda r: r["strength"],
                   default=None)
    mem = result["memberships"]

    a_level = max(
        ["low", "medium", "high"],
        key=lambda l: mem.get(f"anomaly_{l}", 0)
    )
    d_level = max(
        ["low", "medium", "high"],
        key=lambda l: mem.get(f"deviation_{l}", 0)
    )

    lines = [
        f"Risk assessed as **{result['risk_label']}** "
        f"(fuzzy score: {result['fuzzy_score']:.2f}).",
        f"The anomaly model rates this behaviour as *{a_level}* anomaly "
        f"(strength {mem.get(f'anomaly_{a_level}', 0):.2f}).",
        f"Behavioural deviation from personal baseline is *{d_level}* "
        f"(strength {mem.get(f'deviation_{d_level}', 0):.2f}).",
    ]
    if top_rule:
        lines.append(
            f"Dominant rule: «{top_rule['rule']}» "
            f"with firing strength {top_rule['strength']:.2f}."
        )
    return "  \n".join(lines)