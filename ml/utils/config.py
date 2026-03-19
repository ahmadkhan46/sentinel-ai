from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_probability(name: str, value: Any, errors: list[str], *, lower: float = 0.0, upper: float = 1.0) -> None:
    if not _is_number(value):
        errors.append(f"{name} must be numeric.")
        return
    if not (lower <= float(value) <= upper):
        errors.append(f"{name} must be between {lower} and {upper}.")


def validate_config(cfg: dict[str, Any]) -> None:
    errors: list[str] = []

    dataset_cfg = cfg.get("dataset", {})
    if isinstance(dataset_cfg, dict):
        subset = str(dataset_cfg.get("subset", "FD001")).upper()
        if subset not in VALID_SUBSETS:
            errors.append(f"dataset.subset must be one of {sorted(VALID_SUBSETS)} (got: {subset}).")

        # Optional B1/B3 flags — boolean, default false
        for bool_key in ("operating_condition_normalisation", "health_index_feature"):
            val = dataset_cfg.get(bool_key)
            if val is not None and not isinstance(val, bool):
                errors.append(f"dataset.{bool_key} must be a boolean when provided.")

        sequence_length = dataset_cfg.get("sequence_length", 30)
        if not isinstance(sequence_length, int) or sequence_length < 2:
            errors.append("dataset.sequence_length must be an integer >= 2.")

        rul_clip = dataset_cfg.get("target_rul_clip", 125)
        if rul_clip is not None and (not isinstance(rul_clip, int) or rul_clip <= 0):
            errors.append("dataset.target_rul_clip must be a positive integer or null.")
    else:
        errors.append("dataset must be a mapping.")

    phase1_cfg = cfg.get("phase1", {})
    if isinstance(phase1_cfg, dict):
        val_ratio = phase1_cfg.get("val_ratio", 0.2)
        if not _is_number(val_ratio) or not (0.0 < float(val_ratio) < 1.0):
            errors.append("phase1.val_ratio must be numeric in (0, 1).")

        anomaly_horizon = phase1_cfg.get("anomaly_proxy_horizon", 20)
        if not isinstance(anomaly_horizon, int) or anomaly_horizon < 1:
            errors.append("phase1.anomaly_proxy_horizon must be an integer >= 1.")

        healthy_frac = phase1_cfg.get("healthy_cycle_fraction", 0.3)
        _validate_probability("phase1.healthy_cycle_fraction", healthy_frac, errors)
    else:
        errors.append("phase1 must be a mapping.")

    phase2_cfg = cfg.get("phase2", {})
    if isinstance(phase2_cfg, dict):
        threshold_quantile = phase2_cfg.get("threshold_quantile", 0.97)
        _validate_probability("phase2.threshold_quantile", threshold_quantile, errors)

        healthy_frac = phase2_cfg.get("healthy_cycle_fraction", 0.3)
        _validate_probability("phase2.healthy_cycle_fraction", healthy_frac, errors)

        anomaly_horizon = phase2_cfg.get("anomaly_proxy_horizon", 20)
        if not isinstance(anomaly_horizon, int) or anomaly_horizon < 1:
            errors.append("phase2.anomaly_proxy_horizon must be an integer >= 1.")

        hidden_size = phase2_cfg.get("hidden_size", 64)
        if not isinstance(hidden_size, int) or hidden_size < 2:
            errors.append("phase2.hidden_size must be an integer >= 2.")

        epochs = phase2_cfg.get("epochs", 20)
        if not isinstance(epochs, int) or epochs < 1:
            errors.append("phase2.epochs must be an integer >= 1.")

        batch_size = phase2_cfg.get("batch_size", 128)
        if not isinstance(batch_size, int) or batch_size < 1:
            errors.append("phase2.batch_size must be an integer >= 1.")

        learning_rate = phase2_cfg.get("learning_rate", 1e-3)
        if not _is_number(learning_rate) or float(learning_rate) <= 0.0:
            errors.append("phase2.learning_rate must be numeric and > 0.")

        model_type = str(phase2_cfg.get("model_type", "lstm")).lower()
        if model_type not in {"lstm", "gru"}:
            errors.append("phase2.model_type must be 'lstm' or 'gru'.")

        fixed_threshold = phase2_cfg.get("fixed_threshold")
        if fixed_threshold is not None and not _is_number(fixed_threshold):
            errors.append("phase2.fixed_threshold must be numeric when provided.")
    else:
        errors.append("phase2 must be a mapping.")

    anomaly_cfg = cfg.get("anomaly", {})
    if isinstance(anomaly_cfg, dict):
        iforest_cfg = anomaly_cfg.get("iforest", {})
        if isinstance(iforest_cfg, dict):
            contamination = iforest_cfg.get("contamination", 0.03)
            if not _is_number(contamination) or not (0.0 < float(contamination) < 0.5):
                errors.append("anomaly.iforest.contamination must be numeric in (0, 0.5).")
            fixed_threshold = iforest_cfg.get("fixed_threshold")
            if fixed_threshold is not None and not _is_number(fixed_threshold):
                errors.append("anomaly.iforest.fixed_threshold must be numeric when provided.")
        else:
            errors.append("anomaly.iforest must be a mapping.")

        ocsvm_cfg = anomaly_cfg.get("ocsvm", {})
        if isinstance(ocsvm_cfg, dict):
            nu = ocsvm_cfg.get("nu", 0.03)
            if not _is_number(nu) or not (0.0 < float(nu) < 1.0):
                errors.append("anomaly.ocsvm.nu must be numeric in (0, 1).")
            fixed_threshold = ocsvm_cfg.get("fixed_threshold")
            if fixed_threshold is not None and not _is_number(fixed_threshold):
                errors.append("anomaly.ocsvm.fixed_threshold must be numeric when provided.")
        else:
            errors.append("anomaly.ocsvm must be a mapping.")

        alert_cfg = anomaly_cfg.get("alert", {})
        if isinstance(alert_cfg, dict):
            quantile = alert_cfg.get("threshold_quantile", 0.97)
            _validate_probability("anomaly.alert.threshold_quantile", quantile, errors)
        else:
            errors.append("anomaly.alert must be a mapping.")
    else:
        errors.append("anomaly must be a mapping.")

    rul_cfg = cfg.get("rul", {})
    if isinstance(rul_cfg, dict):
        xgb_cfg = rul_cfg.get("xgboost", {})
        if isinstance(xgb_cfg, dict):
            n_estimators = xgb_cfg.get("n_estimators", 500)
            max_depth = xgb_cfg.get("max_depth", 6)
            learning_rate = xgb_cfg.get("learning_rate", 0.03)
            if not isinstance(n_estimators, int) or n_estimators < 1:
                errors.append("rul.xgboost.n_estimators must be an integer >= 1.")
            if not isinstance(max_depth, int) or max_depth < 1:
                errors.append("rul.xgboost.max_depth must be an integer >= 1.")
            if not _is_number(learning_rate) or float(learning_rate) <= 0.0:
                errors.append("rul.xgboost.learning_rate must be numeric and > 0.")
        else:
            errors.append("rul.xgboost must be a mapping.")

        # Optional B2 monotonic flag
        fit_monotonic = rul_cfg.get("fit_monotonic")
        if fit_monotonic is not None and not isinstance(fit_monotonic, bool):
            errors.append("rul.fit_monotonic must be a boolean when provided.")
    else:
        errors.append("rul must be a mapping.")

    if errors:
        joined = "\n".join(f"- {msg}" for msg in errors)
        raise ValueError(f"Invalid configuration:\n{joined}")


def load_config(path: Path, *, validate: bool = True) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
    if validate:
        validate_config(cfg)
    return cfg
