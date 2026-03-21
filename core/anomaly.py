"""
core/anomaly.py
Anomaly Detection Layer — Isolation Forest

Trains on normal (clean) employee feature vectors and scores
new observations. Returns an anomaly score in [0, 1] where
1.0 = maximally anomalous.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from core.features import (
    BehavioralFingerprint,
    build_fingerprints,
    current_matrix,
)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Wraps Isolation Forest with normalised [0,1] anomaly scores.

    Usage
    -----
    detector = AnomalyDetector()
    detector.train(normal_employees)          # or detector.train_matrix(X)
    score = detector.score(fingerprint)       # float in [0, 1]
    scores = detector.score_all(fingerprints) # list[float]
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.scaler = MinMaxScaler()
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, employees) -> None:
        """
        Train on a population of employees whose current behaviour
        is assumed to be normal (no insiders injected yet).
        Uses only current-behaviour vectors (4 features).
        """
        fingerprints = build_fingerprints(employees)
        X = current_matrix(fingerprints)
        self.train_matrix(X)

    def train_matrix(self, X: np.ndarray) -> None:
        """Train directly on a (n_samples, n_features) matrix."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._trained = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _raw_score(self, vec: np.ndarray) -> float:
        """
        Isolation Forest returns decision_function scores:
          - positive  → more normal
          - negative  → more anomalous
        We convert to [0, 1]:
          0 = normal, 1 = anomalous
        """
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))
        raw = self.model.decision_function(vec_scaled)[0]

        # Typical IF scores range roughly [-0.5, 0.5]
        # Map: raw=-0.5 → 1.0, raw=+0.5 → 0.0
        normalized = 1.0 - (raw + 0.5)          # shift to [0, 1]
        return float(np.clip(normalized, 0.0, 1.0))

    def score(self, fingerprint: BehavioralFingerprint) -> float:
        """Return anomaly score in [0, 1] for a single fingerprint."""
        if not self._trained:
            raise RuntimeError("AnomalyDetector has not been trained yet.")
        return self._raw_score(fingerprint.current_vec)

    def score_all(self, fingerprints: list[BehavioralFingerprint]) -> list[float]:
        """Return anomaly scores for a list of fingerprints."""
        return [self.score(fp) for fp in fingerprints]

    def annotate_employees(self, employees) -> None:
        """
        Convenience: compute fingerprints, score them, and write
        anomaly_score back onto each Employee object in-place.
        """
        fingerprints = build_fingerprints(employees)
        for emp, fp in zip(employees, fingerprints):
            emp.anomaly_score = self.score(fp)


# ---------------------------------------------------------------------------
# Convenience factory — quick one-liner for demos
# ---------------------------------------------------------------------------

def build_and_train_detector(normal_employees,
                              contamination: float = 0.1) -> AnomalyDetector:
    """
    Create and immediately train a detector from a clean employee population.

    Example
    -------
    detector = build_and_train_detector(employees)
    detector.annotate_employees(employees)
    """
    detector = AnomalyDetector(contamination=contamination)
    detector.train(normal_employees)
    return detector