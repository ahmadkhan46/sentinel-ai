from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IForestDetector:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.03, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray) -> None:
        self.model.fit(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        # Higher value means more anomalous.
        return -self.model.score_samples(x)
