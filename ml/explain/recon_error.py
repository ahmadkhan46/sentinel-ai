from __future__ import annotations

import numpy as np


def sensor_reconstruction_contrib(x_true: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    """
    Returns per-sensor mean squared reconstruction contribution.
    x arrays shape: [batch, seq_len, n_features]
    """
    err = (x_true - x_recon) ** 2
    return err.mean(axis=(0, 1))


def sequence_sensor_reconstruction_contrib(x_true: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
    """
    Returns per-sequence, per-sensor reconstruction contribution.
    Output shape: [batch, n_features]
    """
    err = (x_true - x_recon) ** 2
    return err.mean(axis=1)
