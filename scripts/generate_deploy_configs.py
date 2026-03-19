from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.utils.config import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-subset deployment configs from benchmark deployment recommendations."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/fd001.yaml"),
        help="Base config path. Default: configs/fd001.yaml",
    )
    parser.add_argument(
        "--recommendations",
        type=Path,
        default=Path("reports/benchmark_summary/deployment_recommendations.json"),
        help="Recommendations JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/deploy"),
        help="Output directory for generated deploy configs. Default: configs/deploy",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports/deploy_runs"),
        help="Reports root path to set in generated configs. Default: reports/deploy_runs",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config file: {path}")
    validate_config(cfg)
    return cfg


def load_recommendations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Recommendations file must be a JSON list: {path}")
    return payload


def ensure_nested(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in cfg or not isinstance(cfg[key], dict):
        cfg[key] = {}
    return cfg[key]


def build_config_for_subset(
    base_cfg: dict[str, Any],
    rec: dict[str, Any],
    reports_root: Path,
) -> dict[str, Any]:
    subset = str(rec["subset"]).upper()
    deploy_phase = str(rec["deploy_phase"])

    cfg = deepcopy(base_cfg)
    dataset = ensure_nested(cfg, "dataset")
    dataset["subset"] = subset

    paths_cfg = ensure_nested(cfg, "paths")
    paths_cfg["reports_dir"] = str(reports_root / subset)

    phase2 = ensure_nested(cfg, "phase2")
    phase3 = ensure_nested(cfg, "phase3")
    anomaly = ensure_nested(cfg, "anomaly")
    alert = ensure_nested(anomaly, "alert")
    iforest_cfg = ensure_nested(anomaly, "iforest")
    ocsvm_cfg = ensure_nested(anomaly, "ocsvm")

    # Keep phase3 off in deploy configs unless explicitly needed.
    phase3["enabled"] = False

    if deploy_phase == "phase2":
        phase2["enabled"] = True
        phase2["model_type"] = str(rec.get("phase2_model_type", "lstm"))
        phase2["threshold_quantile"] = float(rec.get("deploy_threshold_quantile", 0.97))
        phase2["anomaly_proxy_horizon"] = int(rec.get("deploy_horizon_cycles", 20))
        phase2["healthy_cycle_fraction"] = float(rec.get("deploy_healthy_cycle_fraction", 0.3))
        phase2["fixed_threshold"] = float(rec.get("deploy_threshold", 0.0))
        dataset["sequence_length"] = int(rec.get("deploy_sequence_length", dataset.get("sequence_length", 30)))
        # Keep classical baseline thresholds untouched for reference-only configs.
        if "fixed_threshold" in iforest_cfg:
            iforest_cfg.pop("fixed_threshold")
        if "fixed_threshold" in ocsvm_cfg:
            ocsvm_cfg.pop("fixed_threshold")
    else:
        phase2["enabled"] = False
        if "fixed_threshold" in phase2:
            phase2.pop("fixed_threshold")
        alert["threshold_quantile"] = float(rec.get("deploy_threshold_quantile", alert.get("threshold_quantile", 0.97)))
        phase1_model = str(rec.get("phase1_model", "iforest")).lower()
        fixed_thr = float(rec.get("deploy_threshold", 0.0))
        if phase1_model == "iforest":
            iforest_cfg["fixed_threshold"] = fixed_thr
            if "fixed_threshold" in ocsvm_cfg:
                ocsvm_cfg.pop("fixed_threshold")
        elif phase1_model == "ocsvm":
            ocsvm_cfg["fixed_threshold"] = fixed_thr
            if "fixed_threshold" in iforest_cfg:
                iforest_cfg.pop("fixed_threshold")

    cfg["deployment"] = {
        "recommended_strategy": rec.get("recommended_strategy", ""),
        "deploy_phase": deploy_phase,
        "deploy_model": rec.get("deploy_model", ""),
        "selection_reason": rec.get("reason", ""),
        "expected_phase1_f1": rec.get("phase1_f1", ""),
        "expected_phase2_f1": rec.get("phase2_f1", ""),
        "selected_profile": rec.get("phase2_tuned_profile", ""),
        "reference_threshold": rec.get("deploy_threshold", ""),
    }
    return cfg


def main() -> int:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)
    recommendations = load_recommendations(args.recommendations)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, str]] = []
    for rec in recommendations:
        subset = str(rec["subset"]).upper()
        cfg = build_config_for_subset(base_cfg, rec, reports_root=args.reports_root)
        validate_config(cfg)
        out_path = args.output_dir / f"{subset.lower()}_deploy.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        generated.append({"subset": subset, "config_path": str(out_path), "deploy_phase": str(rec.get("deploy_phase"))})
        print(f"Generated: {out_path}")

    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(generated, f, indent=2)
    print(f"Generated manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
