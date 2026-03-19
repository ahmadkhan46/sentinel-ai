from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)


def anomaly_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric RUL score.
    Overestimation (late maintenance) is penalized more than underestimation.
    """
    delta = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(
        delta < 0.0,
        np.exp((-delta) / 13.0) - 1.0,
        np.exp(delta / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def rul_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nasa_score = nasa_rul_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "nasa_score": nasa_score}


# ── B5: Maintenance metrics ────────────────────────────────────────────────────

def compute_early_warning_lead_time(
    anomaly_scores: np.ndarray,
    proxy_labels: np.ndarray,
    threshold: float,
    val_df,
) -> dict[str, float]:
    """
    How many cycles of warning does the model provide on average?

    For each engine, find the first cycle where the anomaly score exceeds the
    threshold, then compute the distance to the first proxy-anomalous cycle.
    Positive lead time = alert fired before the anomaly window started (good).
    """

    if not hasattr(val_df, "iloc"):
        raise TypeError("val_df must be a pandas DataFrame")

    df = val_df.copy()
    df["_score"] = anomaly_scores
    df["_label"] = proxy_labels

    lead_times: list[float] = []
    for eid, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        first_anomaly_cycle = group.loc[group["_label"] == 1, "cycle"]
        first_alert_cycle = group.loc[group["_score"] >= threshold, "cycle"]

        if first_anomaly_cycle.empty:
            continue
        fa_cycle = int(first_anomaly_cycle.iloc[0])

        if first_alert_cycle.empty:
            # Missed — alert never fired; penalise as 0 lead time
            lead_times.append(0.0)
        else:
            lt = fa_cycle - int(first_alert_cycle.iloc[0])
            lead_times.append(float(lt))

    if not lead_times:
        return {"mean_lead_time_cycles": 0.0, "median_lead_time_cycles": 0.0,
                "engines_with_early_warning": 0, "total_engines": 0}

    arr = np.array(lead_times)
    return {
        "mean_lead_time_cycles": float(np.mean(arr)),
        "median_lead_time_cycles": float(np.median(arr)),
        "engines_with_early_warning": int((arr > 0).sum()),
        "total_engines": len(arr),
    }


def compute_mttf(
    anomaly_scores: np.ndarray,
    proxy_labels: np.ndarray,
    threshold: float,
    val_df,
) -> dict[str, float]:
    """
    Mean Time To Failure: average number of cycles between the first alert and
    the actual failure (last cycle of each engine).

    This is the key operational metric: it tells a plant manager how much time
    they have between the model firing an alert and the engine failing.
    """
    df = val_df.copy()
    df["_score"] = anomaly_scores
    df["_label"] = proxy_labels

    mttf_vals: list[float] = []
    for eid, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        failure_cycle = int(group["cycle"].max())
        first_alert_cycle = group.loc[group["_score"] >= threshold, "cycle"]

        if first_alert_cycle.empty:
            continue
        fa = int(first_alert_cycle.iloc[0])
        mttf_vals.append(float(failure_cycle - fa))

    if not mttf_vals:
        return {"mttf_cycles": 0.0, "engines_with_alert": 0}

    arr = np.array(mttf_vals)
    return {
        "mttf_cycles": float(np.mean(arr)),
        "mttf_median_cycles": float(np.median(arr)),
        "mttf_min_cycles": float(arr.min()),
        "engines_with_alert": len(arr),
    }


def compute_oee_impact(
    mttf_cycles: float,
    mttr_cycles: float,
    total_operational_cycles: float,
    unplanned_failure_cost: float = 100_000.0,
    planned_maintenance_cost: float = 10_000.0,
) -> dict[str, float]:
    """
    Estimate Overall Equipment Effectiveness (OEE) impact of the early-warning system.

    OEE Availability = (total - downtime) / total.
    With predictive maintenance, downtime is planned (MTTR_planned << MTTR_unplanned).

    Args:
        mttf_cycles:              Average cycles of warning before failure.
        mttr_cycles:              Assumed Mean Time To Repair (cycles of downtime).
        total_operational_cycles: Total cycles in the evaluation period.
        unplanned_failure_cost:   Cost of one unplanned failure event.
        planned_maintenance_cost: Cost of one planned maintenance event.
    """
    if total_operational_cycles <= 0:
        return {}

    availability_without = max(0.0, (total_operational_cycles - mttr_cycles * 2) / total_operational_cycles)
    availability_with = max(0.0, (total_operational_cycles - mttr_cycles) / total_operational_cycles)
    availability_gain = availability_with - availability_without

    cost_saving_per_event = unplanned_failure_cost - planned_maintenance_cost
    roi_percent = cost_saving_per_event / max(planned_maintenance_cost, 1.0) * 100.0

    return {
        "availability_without_predictive": round(availability_without, 4),
        "availability_with_predictive": round(availability_with, 4),
        "availability_gain": round(availability_gain, 4),
        "cost_saving_per_avoided_failure": round(cost_saving_per_event, 2),
        "roi_percent": round(roi_percent, 1),
        "mttf_cycles_warning": mttf_cycles,
    }


def compute_maintenance_metrics(
    anomaly_scores: np.ndarray,
    proxy_labels: np.ndarray,
    threshold: float,
    val_df,
    mttr_cycles: float = 10.0,
) -> dict[str, object]:
    """
    Compute all maintenance-relevant metrics in one call.
    Returns a nested dict suitable for JSON serialisation.
    """
    lead = compute_early_warning_lead_time(anomaly_scores, proxy_labels, threshold, val_df)
    mttf = compute_mttf(anomaly_scores, proxy_labels, threshold, val_df)
    total_cycles = float(val_df.groupby("engine_id")["cycle"].max().sum())
    oee = compute_oee_impact(
        mttf_cycles=mttf.get("mttf_cycles", 0.0),
        mttr_cycles=mttr_cycles,
        total_operational_cycles=total_cycles,
    )
    return {
        "lead_time": lead,
        "mttf": mttf,
        "oee": oee,
    }
