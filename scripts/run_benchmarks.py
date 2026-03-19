from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.utils.config import validate_config

PHASE2_TUNE_PROFILES: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("q90", {"phase2": {"threshold_quantile": 0.90}}),
    ("q85", {"phase2": {"threshold_quantile": 0.85}}),
    ("q90_h20", {"phase2": {"threshold_quantile": 0.90, "healthy_cycle_fraction": 0.20}}),
    ("q85_h20", {"phase2": {"threshold_quantile": 0.85, "healthy_cycle_fraction": 0.20}}),
    ("seq20_q90", {"dataset": {"sequence_length": 20}, "phase2": {"threshold_quantile": 0.90}}),
    ("gru_default", {"phase2": {"model_type": "gru"}}),
    ("gru_q90", {"phase2": {"model_type": "gru", "threshold_quantile": 0.90}}),
    (
        "gru_q85_h20",
        {
            "phase2": {
                "model_type": "gru",
                "threshold_quantile": 0.85,
                "healthy_cycle_fraction": 0.20,
            }
        },
    ),
    (
        "lstm_q85_h20_h15",
        {
            "phase2": {
                "model_type": "lstm",
                "threshold_quantile": 0.85,
                "healthy_cycle_fraction": 0.20,
                "anomaly_proxy_horizon": 15,
            }
        },
    ),
    (
        "lstm_q85_h20_h30",
        {
            "phase2": {
                "model_type": "lstm",
                "threshold_quantile": 0.85,
                "healthy_cycle_fraction": 0.20,
                "anomaly_proxy_horizon": 30,
            }
        },
    ),
    (
        "gru_q85_h20_h15",
        {
            "phase2": {
                "model_type": "gru",
                "threshold_quantile": 0.85,
                "healthy_cycle_fraction": 0.20,
                "anomaly_proxy_horizon": 15,
            }
        },
    ),
    (
        "gru_q85_h20_h30",
        {
            "phase2": {
                "model_type": "gru",
                "threshold_quantile": 0.85,
                "healthy_cycle_fraction": 0.20,
                "anomaly_proxy_horizon": 30,
            }
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FD001-FD004 benchmark pipeline and produce combined summary outputs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fd001.yaml"),
        help="Base config file path. Default: configs/fd001.yaml",
    )
    parser.add_argument(
        "--subsets",
        default="FD001,FD002,FD003,FD004",
        help="Comma-separated subsets to benchmark. Default: FD001,FD002,FD003,FD004",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing train/test/RUL files. Default: data",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("reports/benchmark_runs"),
        help="Directory where per-subset run artifacts are written. Default: reports/benchmark_runs",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports/benchmark_summary"),
        help="Directory where combined benchmark summary is written. Default: reports/benchmark_summary",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="If set, automatically download missing subset files using scripts/download_cmapss.py.",
    )
    parser.add_argument(
        "--tune-phase2",
        action="store_true",
        help="Run small profile search for Phase 2 and pick best F1 per target subset.",
    )
    parser.add_argument(
        "--tune-subsets",
        default="FD002,FD004",
        help="Comma-separated subsets to tune when --tune-phase2 is enabled. Default: FD002,FD004",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected mapping): {path}")
    validate_config(cfg)
    return cfg


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_subset_files(subset: str, data_dir: Path, auto_download: bool) -> None:
    expected = [
        data_dir / f"train_{subset}.txt",
        data_dir / f"test_{subset}.txt",
        data_dir / f"RUL_{subset}.txt",
    ]
    missing = [p for p in expected if not p.exists()]
    if not missing:
        return
    if not auto_download:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing files for {subset}: {names}. "
            "Use --auto-download or run scripts/download_cmapss.py."
        )

    cmd = [
        sys.executable,
        "scripts/download_cmapss.py",
        "--subset",
        subset,
        "--data-dir",
        str(data_dir),
    ]
    print(f"Downloading missing files for {subset} ...", flush=True)
    subprocess.run(cmd, check=True)


def run_subset_pipeline(
    base_cfg: dict[str, Any],
    subset: str,
    reports_dir: Path,
    data_dir: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(base_cfg)
    cfg.setdefault("dataset", {})
    cfg["dataset"]["subset"] = subset
    cfg.setdefault("paths", {})
    cfg["paths"]["data_dir"] = str(data_dir)
    cfg["paths"]["reports_dir"] = str(reports_dir)
    if overrides:
        deep_update(cfg, overrides)
    validate_config(cfg)

    run_cfg_path = reports_dir / "run_config.yaml"
    with run_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with tempfile.NamedTemporaryFile("w", suffix=f"_{subset}.yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    try:
        cmd = [sys.executable, "-m", "ml.main", "--config", str(tmp_path)]
        subprocess.run(cmd, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    return reports_dir


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_top_feature(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if not row:
        return ""
    return str(row.get("feature", ""))


def tune_phase2_for_subset(
    base_cfg: dict[str, Any],
    subset: str,
    runs_root: Path,
    data_dir: Path,
) -> tuple[str, dict[str, Any]]:
    tuning_root = runs_root / "_tuning" / subset
    tuning_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best_profile = "default"
    best_overrides: dict[str, Any] = {}
    best_key: tuple[float, float, float] | None = None

    for profile_name, profile_overrides in PHASE2_TUNE_PROFILES:
        tune_overrides: dict[str, Any] = {"phase3": {"enabled": False}}
        if profile_overrides:
            deep_update(tune_overrides, deepcopy(profile_overrides))

        profile_reports_dir = tuning_root / profile_name
        run_subset_pipeline(
            base_cfg=base_cfg,
            subset=subset,
            reports_dir=profile_reports_dir,
            data_dir=data_dir,
            overrides=tune_overrides,
        )

        phase2_metrics = read_json(profile_reports_dir / "phase2" / "phase2_metrics.json")
        f1 = float(phase2_metrics["anomaly"]["f1"])
        precision = float(phase2_metrics["anomaly"]["precision"])
        recall = float(phase2_metrics["anomaly"]["recall"])
        model_type = str(phase2_metrics.get("meta", {}).get("model_type", "lstm")).lower()
        horizon = int(phase2_metrics.get("meta", {}).get("anomaly_proxy_horizon_cycles", 20))
        key = (f1, recall, precision)
        results.append(
            {
                "profile": profile_name,
                "model_type": model_type,
                "horizon": horizon,
                "f1": f1,
                "recall": recall,
                "precision": precision,
            }
        )
        print(
            f"  tune[{subset}] {profile_name} ({model_type}): "
            f"horizon={horizon} f1={f1:.4f} recall={recall:.4f} precision={precision:.4f}",
            flush=True,
        )

        if best_key is None or key > best_key:
            best_key = key
            best_profile = profile_name
            best_overrides = deepcopy(profile_overrides)

    results_csv = tuning_root / "tuning_results.csv"
    write_summary_csv(results, results_csv)
    best_payload = {"best_profile": best_profile, "best_overrides": best_overrides, "results": results}
    with (tuning_root / "best_profile.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    return best_profile, best_overrides


def collect_subset_metrics(subset: str, subset_reports_dir: Path) -> dict[str, Any]:
    phase1 = read_json(subset_reports_dir / "phase1" / "baseline_metrics.json")
    phase2 = read_json(subset_reports_dir / "phase2" / "phase2_metrics.json")
    phase3 = read_json(subset_reports_dir / "phase3" / "phase3_summary.json")
    top_shap_feature = read_top_feature(subset_reports_dir / "phase3" / "shap_feature_importance.csv")
    top_sensor_feature = read_top_feature(subset_reports_dir / "phase3" / "autoencoder_sensor_contributions.csv")

    selected_model = str(phase1.get("meta", {}).get("selected_anomaly_model", phase1["anomaly"].get("model", "")))
    phase1_best_f1 = float(phase1["anomaly"]["f1"])
    phase2_f1 = float(phase2["anomaly"]["f1"])
    phase2_model_type = str(phase2.get("meta", {}).get("model_type", "lstm")).lower()
    phase2_horizon = int(phase2.get("meta", {}).get("anomaly_proxy_horizon_cycles", 20))
    phase2_threshold = float(phase2.get("meta", {}).get("threshold", 0.0))
    phase2_threshold_quantile = float(phase2.get("meta", {}).get("threshold_quantile", 0.97))
    phase2_sequence_length = int(phase2.get("meta", {}).get("sequence_length", 30))
    phase2_healthy_fraction = float(phase2.get("meta", {}).get("healthy_cycle_fraction", 0.3))

    phase1_models = phase1.get("anomaly_models", {})
    phase1_selected_threshold = float(phase1_models.get(selected_model, {}).get("threshold", 0.0))
    phase1_threshold_quantile = float(phase1.get("meta", {}).get("threshold_quantile", 0.97))
    phase1_nasa_score_raw = phase1.get("rul", {}).get("nasa_score")
    phase1_nasa_score = float(phase1_nasa_score_raw) if phase1_nasa_score_raw is not None else float("nan")
    phase2_gain_vs_phase1 = phase2_f1 - phase1_best_f1
    recommended_strategy = (
        f"phase2_{phase2_model_type}_autoencoder" if phase2_f1 >= phase1_best_f1 else f"phase1_{selected_model}"
    )

    return {
        "subset": subset,
        "phase1_rul_mae": float(phase1["rul"]["mae"]),
        "phase1_rul_rmse": float(phase1["rul"]["rmse"]),
        "phase1_rul_nasa_score": phase1_nasa_score,
        "phase1_selected_model": selected_model,
        "phase1_anomaly_f1": phase1_best_f1,
        "phase2_anomaly_f1": phase2_f1,
        "phase2_gain_vs_phase1_f1": phase2_gain_vs_phase1,
        "phase2_model_type": phase2_model_type,
        "phase2_horizon": phase2_horizon,
        "phase2_threshold": phase2_threshold,
        "phase2_threshold_quantile": phase2_threshold_quantile,
        "phase2_sequence_length": phase2_sequence_length,
        "phase2_healthy_cycle_fraction": phase2_healthy_fraction,
        "phase2_precision": float(phase2["anomaly"]["precision"]),
        "phase2_recall": float(phase2["anomaly"]["recall"]),
        "phase1_threshold": phase1_selected_threshold,
        "phase1_threshold_quantile": phase1_threshold_quantile,
        "phase1_precision": float(phase1["anomaly"]["precision"]),
        "phase1_recall": float(phase1["anomaly"]["recall"]),
        "recommended_anomaly_strategy": recommended_strategy,
        "top_shap_feature": top_shap_feature,
        "top_sensor_feature": top_sensor_feature,
        "shap_samples": int(phase3["shap_sample_count"]),
    }


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("# Benchmark Summary\n\nNo rows generated.\n", encoding="utf-8")
        return

    sorted_by_rmse = sorted(rows, key=lambda r: r["phase1_rul_rmse"])
    sorted_by_f1 = sorted(rows, key=lambda r: r["phase2_anomaly_f1"], reverse=True)
    sorted_by_nasa = sorted(
        rows,
        key=lambda r: float(r["phase1_rul_nasa_score"])
        if not math.isnan(float(r["phase1_rul_nasa_score"]))
        else float("inf"),
    )
    avg_rmse = sum(r["phase1_rul_rmse"] for r in rows) / len(rows)
    avg_mae = sum(r["phase1_rul_mae"] for r in rows) / len(rows)
    finite_nasa = [float(r["phase1_rul_nasa_score"]) for r in rows if not math.isnan(float(r["phase1_rul_nasa_score"]))]
    avg_nasa = (sum(finite_nasa) / len(finite_nasa)) if finite_nasa else float("nan")
    avg_phase1_f1 = sum(r["phase1_anomaly_f1"] for r in rows) / len(rows)
    avg_phase2_f1 = sum(r["phase2_anomaly_f1"] for r in rows) / len(rows)
    avg_phase2_gain = sum(r["phase2_gain_vs_phase1_f1"] for r in rows) / len(rows)
    phase2_reco_count = sum(1 for r in rows if str(r["recommended_anomaly_strategy"]).startswith("phase2_"))
    phase1_reco_count = len(rows) - phase2_reco_count

    lines = [
        "# Benchmark Summary",
        "",
        "## Metrics Table",
        "",
        "| Subset | RUL RMSE | RUL MAE | NASA Score | Phase1 Model | Phase1 F1 | Phase2 Model | Phase2 Horizon | Phase2 F1 | F1 Gain | Phase2 Tune Profile | Recommended Anomaly Strategy | Top SHAP | Top AE Sensor |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subset']} | {r['phase1_rul_rmse']:.4f} | {r['phase1_rul_mae']:.4f} | "
            f"{r['phase1_rul_nasa_score']:.4f} | "
            f"{r['phase1_selected_model']} | {r['phase1_anomaly_f1']:.4f} | "
            f"{r.get('phase2_model_type', 'lstm')} | {r.get('phase2_horizon', 20)} | {r['phase2_anomaly_f1']:.4f} | "
            f"{r['phase2_gain_vs_phase1_f1']:.4f} | "
            f"{r.get('phase2_tuned_profile', 'default')} | "
            f"{r['recommended_anomaly_strategy']} | {r['top_shap_feature']} | {r['top_sensor_feature']} |"
        )

    lines.extend(
        [
            "",
            "## Ranking",
            "",
            f"- Best RUL RMSE: {sorted_by_rmse[0]['subset']} ({sorted_by_rmse[0]['phase1_rul_rmse']:.4f})",
            f"- Best NASA score: {sorted_by_nasa[0]['subset']} ({sorted_by_nasa[0]['phase1_rul_nasa_score']:.4f})",
            f"- Best Phase2 anomaly F1: {sorted_by_f1[0]['subset']} ({sorted_by_f1[0]['phase2_anomaly_f1']:.4f})",
            "",
            "## Averages",
            "",
            f"- Avg RUL RMSE: {avg_rmse:.4f}",
            f"- Avg RUL MAE: {avg_mae:.4f}",
            f"- Avg NASA score: {avg_nasa:.4f}",
            f"- Avg Phase1 anomaly F1: {avg_phase1_f1:.4f}",
            f"- Avg Phase2 anomaly F1: {avg_phase2_f1:.4f}",
            f"- Avg Phase2 gain vs Phase1 F1: {avg_phase2_gain:.4f}",
            "",
            "## Strategy Split",
            "",
            f"- Recommended Phase2 strategy count: {phase2_reco_count}",
            f"- Recommended Phase1 strategy count: {phase1_reco_count}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_deployment_recommendations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for r in rows:
        use_phase2 = str(r["recommended_anomaly_strategy"]).startswith("phase2_")
        base = {
            "subset": r["subset"],
            "recommended_strategy": r["recommended_anomaly_strategy"],
            "reason": (
                f"phase2_f1={r['phase2_anomaly_f1']:.4f} >= phase1_f1={r['phase1_anomaly_f1']:.4f}"
                if use_phase2
                else f"phase1_f1={r['phase1_anomaly_f1']:.4f} > phase2_f1={r['phase2_anomaly_f1']:.4f}"
            ),
            "phase1_model": r["phase1_selected_model"],
            "phase1_f1": r["phase1_anomaly_f1"],
            "phase2_model_type": r.get("phase2_model_type", "lstm"),
            "phase2_f1": r["phase2_anomaly_f1"],
            "phase2_tuned_profile": r.get("phase2_tuned_profile", "default"),
        }
        if use_phase2:
            recommendation = {
                **base,
                "deploy_phase": "phase2",
                "deploy_model": f"{r.get('phase2_model_type', 'lstm')}_autoencoder",
                "deploy_threshold": r.get("phase2_threshold", 0.0),
                "deploy_threshold_quantile": r.get("phase2_threshold_quantile", 0.97),
                "deploy_sequence_length": r.get("phase2_sequence_length", 30),
                "deploy_horizon_cycles": r.get("phase2_horizon", 20),
                "deploy_healthy_cycle_fraction": r.get("phase2_healthy_cycle_fraction", 0.3),
            }
        else:
            recommendation = {
                **base,
                "deploy_phase": "phase1",
                "deploy_model": r["phase1_selected_model"],
                "deploy_threshold": r.get("phase1_threshold", 0.0),
                "deploy_threshold_quantile": r.get("phase1_threshold_quantile", 0.97),
                "deploy_sequence_length": "",
                "deploy_horizon_cycles": "",
                "deploy_healthy_cycle_fraction": "",
            }
        recommendations.append(recommendation)
    return recommendations


def write_recommendations_json(recommendations: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2)


def write_recommendations_md(recommendations: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not recommendations:
        output_path.write_text("# Deployment Recommendations\n\nNo recommendations generated.\n", encoding="utf-8")
        return

    lines = [
        "# Deployment Recommendations",
        "",
        "| Subset | Deploy Phase | Deploy Model | Threshold | Quantile | Horizon | Reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in recommendations:
        horizon = r["deploy_horizon_cycles"] if r["deploy_horizon_cycles"] != "" else "-"
        lines.append(
            f"| {r['subset']} | {r['deploy_phase']} | {r['deploy_model']} | "
            f"{float(r['deploy_threshold']):.6f} | {float(r['deploy_threshold_quantile']):.2f} | "
            f"{horizon} | {r['reason']} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    subsets = [r["subset"] for r in rows]
    phase1_f1 = [float(r["phase1_anomaly_f1"]) for r in rows]
    phase2_f1 = [float(r["phase2_anomaly_f1"]) for r in rows]
    rmse_vals = [float(r["phase1_rul_rmse"]) for r in rows]
    nasa_vals = [float(r["phase1_rul_nasa_score"]) for r in rows]

    x = list(range(len(subsets)))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], phase1_f1, width=width, label="Phase1 F1")
    ax.bar([i + width / 2 for i in x], phase2_f1, width=width, label="Phase2 F1")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Anomaly Detection F1 by Subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_f1_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subsets, rmse_vals, color="#4C78A8")
    ax.set_ylabel("RMSE")
    ax.set_title("RUL RMSE by Subset")
    for i, val in enumerate(rmse_vals):
        ax.text(i, val + 0.2, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "rul_rmse_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(subsets, nasa_vals, color="#72B7B2")
    ax.set_ylabel("NASA score (lower better)")
    ax.set_title("RUL NASA Score by Subset")
    finite_nasa_vals = [v for v in nasa_vals if not math.isnan(v)]
    y_offset = (max(finite_nasa_vals) * 0.01) if finite_nasa_vals else 0.0
    for i, val in enumerate(nasa_vals):
        if math.isnan(val):
            continue
        ax.text(i, val + y_offset, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "rul_nasa_score_comparison.png", dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    subsets = [s.strip().upper() for s in args.subsets.split(",") if s.strip()]
    tune_subsets = {s.strip().upper() for s in args.tune_subsets.split(",") if s.strip()}
    valid = {"FD001", "FD002", "FD003", "FD004"}
    invalid = [s for s in subsets if s not in valid]
    if invalid:
        print(f"Invalid subsets: {invalid}. Allowed: FD001, FD002, FD003, FD004", file=sys.stderr)
        return 2
    invalid_tune = [s for s in tune_subsets if s not in valid]
    if invalid_tune:
        print(f"Invalid tune subsets: {invalid_tune}. Allowed: FD001, FD002, FD003, FD004", file=sys.stderr)
        return 2

    base_cfg = load_yaml(args.config)
    rows: list[dict[str, Any]] = []

    for subset in subsets:
        print(f"\n=== Running benchmark for {subset} ===", flush=True)
        ensure_subset_files(subset=subset, data_dir=args.data_dir, auto_download=args.auto_download)
        selected_profile = "default"
        selected_overrides: dict[str, Any] = {}
        if args.tune_phase2 and subset in tune_subsets:
            print(f"Tuning Phase 2 profiles for {subset} ...", flush=True)
            selected_profile, selected_overrides = tune_phase2_for_subset(
                base_cfg=base_cfg,
                subset=subset,
                runs_root=args.runs_root,
                data_dir=args.data_dir,
            )
            print(f"Selected profile for {subset}: {selected_profile}", flush=True)

        subset_reports_dir = run_subset_pipeline(
            base_cfg=base_cfg,
            subset=subset,
            reports_dir=args.runs_root / subset,
            data_dir=args.data_dir,
            overrides=selected_overrides if selected_overrides else None,
        )
        row = collect_subset_metrics(subset=subset, subset_reports_dir=subset_reports_dir)
        row["phase2_tuned_profile"] = selected_profile
        rows.append(row)
        print(
            f"{subset}: RMSE={row['phase1_rul_rmse']:.4f}, "
            f"Phase2_F1={row['phase2_anomaly_f1']:.4f}, "
            f"Gain={row['phase2_gain_vs_phase1_f1']:.4f}, "
            f"Model={row.get('phase2_model_type', 'lstm')}, "
            f"Horizon={row.get('phase2_horizon', 20)}, "
            f"Profile={selected_profile}, Top_SHAP={row['top_shap_feature']}"
        )

    csv_path = args.summary_dir / "benchmark_results.csv"
    md_path = args.summary_dir / "benchmark_summary.md"
    plots_dir = args.summary_dir / "plots"
    reco_json_path = args.summary_dir / "deployment_recommendations.json"
    reco_csv_path = args.summary_dir / "deployment_recommendations.csv"
    reco_md_path = args.summary_dir / "deployment_recommendations.md"
    write_summary_csv(rows, csv_path)
    write_summary_md(rows, md_path)
    write_summary_plots(rows, plots_dir)
    recommendations = build_deployment_recommendations(rows)
    write_recommendations_json(recommendations, reco_json_path)
    write_summary_csv(recommendations, reco_csv_path)
    write_recommendations_md(recommendations, reco_md_path)
    print(f"\nSaved combined CSV: {csv_path}")
    print(f"Saved combined report: {md_path}")
    print(f"Saved benchmark plots to: {plots_dir}")
    print(f"Saved deployment recommendations JSON: {reco_json_path}")
    print(f"Saved deployment recommendations CSV: {reco_csv_path}")
    print(f"Saved deployment recommendations report: {reco_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
