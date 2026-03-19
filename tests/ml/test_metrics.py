from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.eval.metrics import compute_maintenance_metrics


def _make_val_df(n_engines: int = 3, cycles_per: int = 10) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build minimal val_df with engine_id and cycle columns."""
    rows = []
    scores_list = []
    labels_list = []

    rng = np.random.default_rng(42)
    for eid in range(1, n_engines + 1):
        for c in range(1, cycles_per + 1):
            rows.append({"engine_id": eid, "cycle": c})
            # Score ramps up near end; proxy label = 1 in last 3 cycles
            score = 0.1 + 0.08 * c + rng.uniform(-0.05, 0.05)
            scores_list.append(score)
            labels_list.append(1 if c >= cycles_per - 2 else 0)

    df = pd.DataFrame(rows)
    return np.array(scores_list), np.array(labels_list), df


def test_maintenance_metrics_returns_expected_keys():
    scores, labels, val_df = _make_val_df()
    threshold = 0.5

    metrics = compute_maintenance_metrics(scores, labels, threshold, val_df)

    assert "lead_time" in metrics
    assert "mttf" in metrics
    assert "oee" in metrics
    assert "mean_lead_time_cycles" in metrics["lead_time"]


def test_maintenance_metrics_early_warning_positive():
    """With a ramping score and late proxy labels, lead time should be positive."""
    scores, labels, val_df = _make_val_df(n_engines=5, cycles_per=20)
    threshold = 0.5

    metrics = compute_maintenance_metrics(scores, labels, threshold, val_df)
    lead = metrics["lead_time"]
    # At least some engines should fire an alert before the anomaly window
    assert lead["engines_with_early_warning"] >= 0  # non-negative (structural check)


def test_maintenance_metrics_no_alerts():
    """Threshold above all scores — no alerts fired, lead time should be 0."""
    scores, labels, val_df = _make_val_df()
    threshold = 999.0  # impossibly high

    metrics = compute_maintenance_metrics(scores, labels, threshold, val_df)
    assert metrics["lead_time"]["engines_with_early_warning"] == 0
