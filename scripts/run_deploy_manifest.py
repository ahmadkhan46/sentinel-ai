from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.utils.config import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all deployment configs listed in a manifest and produce consolidated run summaries."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/deploy/manifest.json"),
        help="Path to deployment manifest JSON.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("reports/deploy_runs/summary"),
        help="Output directory for deploy run summaries.",
    )
    parser.add_argument(
        "--subsets",
        default="",
        help="Optional comma-separated subset filter (e.g., FD001,FD003).",
    )
    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run configs even when outputs already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failed run.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run pipeline. Default: current interpreter.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    validate_config(payload)
    return payload


def has_complete_outputs(reports_dir: Path, deploy_phase_expected: str) -> bool:
    phase1_ok = (reports_dir / "phase1" / "baseline_metrics.json").exists()
    phase2_ok = (reports_dir / "phase2" / "phase2_metrics.json").exists()
    if deploy_phase_expected == "phase2":
        return phase1_ok and phase2_ok
    return phase1_ok


def parse_subset_filter(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def derive_strategy(phase1_model: str, phase1_f1: float | None, phase2_model: str, phase2_f1: float | None) -> str:
    if phase1_f1 is None and phase2_f1 is None:
        return ""
    if phase2_f1 is not None and phase1_f1 is not None and phase2_f1 >= phase1_f1:
        model = phase2_model if phase2_model else "lstm"
        return f"phase2_{model}_autoencoder"
    if phase1_model:
        return f"phase1_{phase1_model}"
    return ""


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("# Deploy Manifest Run Summary\n\nNo rows generated.\n", encoding="utf-8")
        return

    total = len(rows)
    success = sum(1 for r in rows if r["status"] in {"success", "skipped_existing"})
    failed = sum(1 for r in rows if r["status"] == "failed")
    matches = sum(1 for r in rows if str(r["recommendation_match"]).lower() == "true")
    attempted = sum(1 for r in rows if r["status"] != "skipped_existing")

    lines = [
        "# Deploy Manifest Run Summary",
        "",
        "## Runs",
        "",
        f"- Total rows: {total}",
        f"- Successful/skipped: {success}",
        f"- Failed: {failed}",
        f"- Recommendation matches: {matches}",
        f"- Attempted executions: {attempted}",
        "",
        "## Results Table",
        "",
        "| Subset | Status | Expected Strategy | Derived Strategy | Match | Phase1 F1 | Phase2 F1 | Duration (s) |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        p1 = "" if r["phase1_f1"] is None else f"{float(r['phase1_f1']):.4f}"
        p2 = "" if r["phase2_f1"] is None else f"{float(r['phase2_f1']):.4f}"
        lines.append(
            f"| {r['subset']} | {r['status']} | {r['recommended_strategy_expected']} | {r['derived_strategy']} | "
            f"{r['recommendation_match']} | {p1} | {p2} | {float(r['duration_sec']):.2f} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_entry(entry: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(str(entry.get("config_path", "")))
    subset_hint = str(entry.get("subset", ""))
    if not config_path.exists():
        return {
            "subset": subset_hint,
            "config_path": str(config_path),
            "status": "failed",
            "duration_sec": 0.0,
            "return_code": -1,
            "deploy_phase_expected": str(entry.get("deploy_phase", "")),
            "deploy_model_expected": "",
            "recommended_strategy_expected": "",
            "derived_strategy": "",
            "recommendation_match": False,
            "phase1_model": "",
            "phase1_f1": None,
            "phase2_model": "",
            "phase2_f1": None,
            "reports_dir": "",
            "error": f"Missing config file: {config_path}",
        }

    try:
        cfg = load_yaml(config_path)
    except Exception as exc:
        return {
            "subset": subset_hint,
            "config_path": str(config_path),
            "status": "failed",
            "duration_sec": 0.0,
            "return_code": -1,
            "deploy_phase_expected": str(entry.get("deploy_phase", "")),
            "deploy_model_expected": "",
            "recommended_strategy_expected": "",
            "derived_strategy": "",
            "recommendation_match": False,
            "phase1_model": "",
            "phase1_f1": None,
            "phase2_model": "",
            "phase2_f1": None,
            "reports_dir": "",
            "error": f"Invalid deploy config {config_path}: {exc}",
        }
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    deployment_cfg = cfg.get("deployment", {}) if isinstance(cfg.get("deployment"), dict) else {}

    subset = str(entry.get("subset") or dataset_cfg.get("subset", "")).upper()
    reports_dir = Path(str(paths_cfg.get("reports_dir", "")))
    deploy_phase_expected = str(deployment_cfg.get("deploy_phase", entry.get("deploy_phase", "")))
    deploy_model_expected = str(deployment_cfg.get("deploy_model", ""))
    recommended_strategy_expected = str(deployment_cfg.get("recommended_strategy", ""))

    start = time.perf_counter()
    status = "success"
    return_code = 0
    err_msg = ""

    if not args.force_run and reports_dir and has_complete_outputs(reports_dir, deploy_phase_expected):
        status = "skipped_existing"
    else:
        cmd = [args.python_exe, "-m", "ml.main", "--config", str(config_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        return_code = int(completed.returncode)
        if return_code != 0:
            status = "failed"
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            err_msg = (stderr or stdout)[-1000:]

    duration_sec = time.perf_counter() - start

    phase1_path = reports_dir / "phase1" / "baseline_metrics.json"
    phase2_path = reports_dir / "phase2" / "phase2_metrics.json"

    phase1_model = ""
    phase1_f1: float | None = None
    phase2_model = ""
    phase2_f1: float | None = None

    if phase1_path.exists():
        phase1 = load_json(phase1_path)
        phase1_model = str(phase1.get("anomaly", {}).get("model", ""))
        val = phase1.get("anomaly", {}).get("f1")
        phase1_f1 = float(val) if val is not None else None

    if phase2_path.exists():
        phase2 = load_json(phase2_path)
        phase2_model = str(phase2.get("meta", {}).get("model_type", ""))
        val = phase2.get("anomaly", {}).get("f1")
        phase2_f1 = float(val) if val is not None else None

    derived_strategy = derive_strategy(phase1_model, phase1_f1, phase2_model, phase2_f1)
    recommendation_match = bool(recommended_strategy_expected and derived_strategy == recommended_strategy_expected)

    return {
        "subset": subset,
        "config_path": str(config_path),
        "status": status,
        "duration_sec": round(duration_sec, 3),
        "return_code": return_code,
        "deploy_phase_expected": deploy_phase_expected,
        "deploy_model_expected": deploy_model_expected,
        "recommended_strategy_expected": recommended_strategy_expected,
        "derived_strategy": derived_strategy,
        "recommendation_match": recommendation_match,
        "phase1_model": phase1_model,
        "phase1_f1": phase1_f1,
        "phase2_model": phase2_model,
        "phase2_f1": phase2_f1,
        "reports_dir": str(reports_dir),
        "error": err_msg,
    }


def main() -> int:
    args = parse_args()
    manifest_payload = load_json(args.manifest)
    if not isinstance(manifest_payload, list):
        raise ValueError(f"Manifest must be a list: {args.manifest}")

    subset_filter = parse_subset_filter(args.subsets)
    rows: list[dict[str, Any]] = []

    for entry in manifest_payload:
        if not isinstance(entry, dict):
            continue
        subset = str(entry.get("subset", "")).upper()
        if subset_filter and subset not in subset_filter:
            continue

        print(f"Running deploy config for {subset} ...", flush=True)
        row = run_entry(entry, args)
        rows.append(row)
        print(
            f"  {subset}: status={row['status']} phase1_f1={row['phase1_f1']} "
            f"phase2_f1={row['phase2_f1']} match={row['recommendation_match']}",
            flush=True,
        )

        if row["status"] == "failed" and args.fail_fast:
            print("Stopping early due to --fail-fast.", flush=True)
            break

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.summary_dir / "deploy_manifest_results.csv"
    json_path = args.summary_dir / "deploy_manifest_results.json"
    md_path = args.summary_dir / "deploy_manifest_results.md"

    write_csv(rows, csv_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    write_md(rows, md_path)

    print(f"\nSaved deploy summary CSV: {csv_path}")
    print(f"Saved deploy summary JSON: {json_path}")
    print(f"Saved deploy summary report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
