from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


class OCSVMDetector:
    def __init__(self, kernel: str = "rbf", nu: float = 0.03, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, x: np.ndarray) -> None:
        self.model.fit(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        # Decision function: positive = inlier. Flip sign for anomaly score.
        return -self.model.decision_function(x)
