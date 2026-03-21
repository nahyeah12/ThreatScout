"""
core/features.py
Behavioral Fingerprint Layer

Converts raw employee behavior into structured feature vectors.
Computes baseline, current, and deviation vectors — forming the
"behavioral fingerprint" that the anomaly + fuzzy layers operate on.
"""

import numpy as np

# Canonical feature order — must stay consistent across all layers
FEATURE_KEYS = ["after_hours_login", "file_access", "usb_usage", "job_sites"]


# ---------------------------------------------------------------------------
# Feature vector helpers
# ---------------------------------------------------------------------------

def behaviour_to_vector(behaviour: dict) -> np.ndarray:
    """Convert a behaviour dict to a fixed-order numpy array."""
    return np.array([behaviour[k] for k in FEATURE_KEYS], dtype=float)


def vector_to_behaviour(vec: np.ndarray) -> dict:
    """Convert a numpy array back to a behaviour dict."""
    return {k: float(vec[i]) for i, k in enumerate(FEATURE_KEYS)}


# ---------------------------------------------------------------------------
# Behavioral Fingerprint
# ---------------------------------------------------------------------------

class BehavioralFingerprint:
    """
    Represents a single snapshot of an employee's behavioral state.

    Attributes
    ----------
    employee_id   : str
    role          : str
    baseline_vec  : np.ndarray   — historical normal (from role baseline)
    current_vec   : np.ndarray   — observed at this timestep
    deviation_vec : np.ndarray   — |current - baseline|
    deviation_score : float      — scalar magnitude of deviation (0-1 range)
    """

    def __init__(self, employee):
        self.employee_id = employee.id
        self.role = employee.role

        self.baseline_vec = behaviour_to_vector(employee.baseline)
        self.current_vec  = behaviour_to_vector(employee.current)

        # Unsigned deviation per feature
        self.deviation_vec = np.abs(self.current_vec - self.baseline_vec)

        # Scalar deviation score: mean of deviations (0–1 scale)
        self.deviation_score = float(np.mean(self.deviation_vec))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def baseline_dict(self) -> dict:
        return vector_to_behaviour(self.baseline_vec)

    @property
    def current_dict(self) -> dict:
        return vector_to_behaviour(self.current_vec)

    @property
    def deviation_dict(self) -> dict:
        return vector_to_behaviour(self.deviation_vec)

    @property
    def feature_vector(self) -> np.ndarray:
        """
        The combined feature vector fed to the anomaly model.
        Concatenates: [baseline | current | deviation]
        Shape: (12,)
        """
        return np.concatenate([self.baseline_vec,
                                self.current_vec,
                                self.deviation_vec])

    def __repr__(self):
        return (f"Fingerprint({self.employee_id} | {self.role} | "
                f"dev={self.deviation_score:.3f})")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def build_fingerprints(employees) -> list[BehavioralFingerprint]:
    """Return a fingerprint for every employee in the list."""
    return [BehavioralFingerprint(emp) for emp in employees]


def fingerprints_to_matrix(fingerprints: list[BehavioralFingerprint]) -> np.ndarray:
    """
    Stack all feature vectors into a 2-D array for model training/inference.
    Shape: (n_employees, 12)
    """
    return np.vstack([fp.feature_vector for fp in fingerprints])


def current_matrix(fingerprints: list[BehavioralFingerprint]) -> np.ndarray:
    """
    Returns only current-behaviour vectors.
    Shape: (n_employees, 4)
    Useful for training the anomaly model on 'normal' current behaviour.
    """
    return np.vstack([fp.current_vec for fp in fingerprints])