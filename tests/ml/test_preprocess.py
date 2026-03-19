from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocess import OperatingConditionNormaliser, compute_health_index


SENSOR_COLS = [f"sensor_{i}" for i in range(1, 8)]


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "op_setting_1": rng.uniform(-0.01, 0.01, n),
        "op_setting_2": rng.uniform(0, 0.001, n),
        "op_setting_3": rng.choice([60.0, 80.0, 100.0], n),
        **{f"sensor_{i}": rng.uniform(100, 600, n) for i in range(1, 8)},
    }
    return pd.DataFrame(data)


def test_oc_normaliser_fit_transform():
    df = _make_df(200)
    healthy_mask = np.ones(200, dtype=bool)
    healthy_mask[180:] = False

    norm = OperatingConditionNormaliser(n_regimes=3, seed=0)
    norm.fit(df, SENSOR_COLS, healthy_mask)
    out = norm.transform(df, SENSOR_COLS)

    assert out.shape == df.shape  # transform returns full df with sensor cols updated
    assert not out.isnull().any().any()


def test_oc_normaliser_zero_variation():
    """Columns with zero std in a regime should not produce NaN."""
    df = _make_df(60)
    df["op_setting_3"] = 60.0  # constant op condition
    healthy_mask = np.ones(60, dtype=bool)

    norm = OperatingConditionNormaliser(n_regimes=2, seed=0)
    norm.fit(df, SENSOR_COLS, healthy_mask)
    out = norm.transform(df, SENSOR_COLS)
    assert not out.isnull().any().any()


def test_compute_health_index_range():
    df = _make_df(120)
    healthy_mask = np.ones(120, dtype=bool)
    healthy_mask[100:] = False

    hi = compute_health_index(df, SENSOR_COLS, healthy_mask)
    assert hi.shape == (120,)
    assert np.all((hi >= 0) & (hi <= 1))
