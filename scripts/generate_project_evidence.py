from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one-page project evidence brief (Markdown + PDF) for PhD applications."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("reports/benchmark_summary/benchmark_results.csv"),
        help="Benchmark results CSV path.",
    )
    parser.add_argument(
        "--recommendations-json",
        type=Path,
        default=Path("reports/benchmark_summary/deployment_recommendations.json"),
        help="Deployment recommendations JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/application"),
        help="Output directory for generated brief files.",
    )
    parser.add_argument(
        "--project-title",
        default="AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance (NASA C-MAPSS)",
        help="Title used in the generated brief.",
    )
    return parser.parse_args()


def load_benchmark_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in benchmark CSV: {path}")
    return rows


def load_recommendations(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing recommendations JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Recommendations must be a list: {path}")
    return payload


def to_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def build_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    sorted_rmse = sorted(rows, key=lambda r: to_float(r, "phase1_rul_rmse"))
    sorted_nasa = sorted(rows, key=lambda r: to_float(r, "phase1_rul_nasa_score"))
    sorted_phase2_f1 = sorted(rows, key=lambda r: to_float(r, "phase2_anomaly_f1"), reverse=True)

    avg_rmse = sum(to_float(r, "phase1_rul_rmse") for r in rows) / len(rows)
    avg_mae = sum(to_float(r, "phase1_rul_mae") for r in rows) / len(rows)
    avg_nasa = sum(to_float(r, "phase1_rul_nasa_score") for r in rows) / len(rows)
    avg_phase1_f1 = sum(to_float(r, "phase1_anomaly_f1") for r in rows) / len(rows)
    avg_phase2_f1 = sum(to_float(r, "phase2_anomaly_f1") for r in rows) / len(rows)

    phase2_reco = sum(
        1 for r in rows if str(r.get("recommended_anomaly_strategy", "")).startswith("phase2_")
    )
    phase1_reco = len(rows) - phase2_reco

    return {
        "best_rmse_subset": sorted_rmse[0]["subset"],
        "best_rmse": to_float(sorted_rmse[0], "phase1_rul_rmse"),
        "best_nasa_subset": sorted_nasa[0]["subset"],
        "best_nasa": to_float(sorted_nasa[0], "phase1_rul_nasa_score"),
        "best_phase2_subset": sorted_phase2_f1[0]["subset"],
        "best_phase2_f1": to_float(sorted_phase2_f1[0], "phase2_anomaly_f1"),
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_nasa": avg_nasa,
        "avg_phase1_f1": avg_phase1_f1,
        "avg_phase2_f1": avg_phase2_f1,
        "phase1_reco_count": phase1_reco,
        "phase2_reco_count": phase2_reco,
    }


def build_md(
    project_title: str,
    rows: list[dict[str, str]],
    recommendations: list[dict[str, object]],
    summary: dict[str, object],
) -> str:
    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Project Evidence Brief",
        "",
        f"**Project:** {project_title}",
        "**Target Position:** UCC PhD Studentship in AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance",
        f"**Generated:** {generated_utc}",
        "",
        "## Project Scope",
        "- Developed a reproducible end-to-end industrial AI pipeline on NASA C-MAPSS turbofan data (FD001-FD004).",
        "- Implemented baseline anomaly detection (Isolation Forest, One-Class SVM), sequence models (LSTM/GRU autoencoders), RUL prediction (XGBoost), and diagnostics (SHAP plus reconstruction-error sensor attribution).",
        "- Added deployment-oriented strategy selection per subset, fixed-threshold deployment configs, and manifest-based execution validation.",
        "",
        "## Quantitative Results",
        "",
        "| Subset | Phase 1 Model | Phase 1 F1 | Phase 2 F1 | Recommended Strategy | RUL RMSE | NASA Score | Top SHAP Feature |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['subset']} | {r['phase1_selected_model']} | {float(r['phase1_anomaly_f1']):.4f} | "
            f"{float(r['phase2_anomaly_f1']):.4f} | {r['recommended_anomaly_strategy']} | "
            f"{float(r['phase1_rul_rmse']):.4f} | {float(r['phase1_rul_nasa_score']):.2f} | {r['top_shap_feature']} |"
        )

    lines.extend(
        [
            "",
            "## Highlights",
            f"- Best RUL RMSE: {summary['best_rmse_subset']} ({summary['best_rmse']:.4f})",
            f"- Best NASA RUL score (lower is better): {summary['best_nasa_subset']} ({summary['best_nasa']:.2f})",
            f"- Best sequence anomaly F1: {summary['best_phase2_subset']} ({summary['best_phase2_f1']:.4f})",
            f"- Average RUL RMSE across all subsets: {summary['avg_rmse']:.4f}",
            f"- Average RUL MAE across all subsets: {summary['avg_mae']:.4f}",
            f"- Average NASA RUL score across all subsets: {summary['avg_nasa']:.2f}",
            f"- Average anomaly F1: phase1={summary['avg_phase1_f1']:.4f}, phase2={summary['avg_phase2_f1']:.4f}",
            f"- Deployment split from benchmark recommendations: phase2={summary['phase2_reco_count']}, phase1={summary['phase1_reco_count']}",
            "",
            "## Research Alignment With UCC Topic",
            "- Fault detection: tabular and sequence anomaly detection with subset-specific thresholding.",
            "- Diagnostics: SHAP feature attribution and sensor-level reconstruction-error ranking.",
            "- Predictive maintenance: RUL forecasting with RMSE/MAE and NASA asymmetric scoring.",
            "- Deployment readiness: generated deploy configs plus manifest execution with recommendation-match checks.",
            "",
            "## Deployment Recommendations (Extract)",
            "",
            "| Subset | Deploy Phase | Deploy Model | Threshold | Reason |",
            "|---|---|---|---:|---|",
        ]
    )

    for rec in recommendations:
        lines.append(
            f"| {rec.get('subset', '')} | {rec.get('deploy_phase', '')} | {rec.get('deploy_model', '')} | "
            f"{float(rec.get('deploy_threshold', 0.0)):.6f} | {rec.get('reason', '')} |"
        )

    lines.extend(
        [
            "",
            "## Evidence Files",
            "- Benchmark summary: `reports/benchmark_summary/benchmark_summary.md`",
            "- Benchmark metrics table: `reports/benchmark_summary/benchmark_results.csv`",
            "- Deployment recommendations: `reports/benchmark_summary/deployment_recommendations.json`",
            "- Deploy validation summary: `reports/deploy_runs/summary/deploy_manifest_results.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pdf(
    output_path: Path,
    project_title: str,
    rows: list[dict[str, str]],
    summary: dict[str, object],
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    y = 0.97
    lh = 0.025

    def write(line: str, *, size: int = 10, weight: str = "normal", gap: float = 1.0) -> None:
        nonlocal y
        ax.text(0.05, y, line, fontsize=size, fontweight=weight, va="top", ha="left", family="DejaVu Sans")
        y -= lh * gap

    write("Project Evidence Brief", size=16, weight="bold", gap=1.3)
    write(project_title, size=11, weight="bold")
    write("Target: UCC PhD in AI-Enabled Fault Detection, Diagnostics and Predictive Maintenance", size=10)
    write("", gap=0.5)

    write("Scope", size=12, weight="bold")
    for b in [
        "Built an end-to-end industrial AI pipeline on NASA C-MAPSS (FD001-FD004).",
        "Implemented anomaly detection (Isolation Forest, One-Class SVM, LSTM autoencoder) plus RUL prediction (XGBoost).",
        "Added diagnostics using SHAP and sequence reconstruction-error sensor attribution.",
        "Created deploy strategy recommendations, generated deploy configs, and validated deploy runs via manifest automation.",
    ]:
        write(fill(f"- {b}", width=100), size=10)
    write("", gap=0.5)

    write("Key Results", size=12, weight="bold")
    write("Subset  Phase1_F1  Phase2_F1  Strategy                    RMSE     NASA", size=10, weight="bold")
    for r in rows:
        line = (
            f"{r['subset']:<7} "
            f"{float(r['phase1_anomaly_f1']):>8.4f}  "
            f"{float(r['phase2_anomaly_f1']):>8.4f}  "
            f"{r['recommended_anomaly_strategy']:<26} "
            f"{float(r['phase1_rul_rmse']):>7.4f}  "
            f"{float(r['phase1_rul_nasa_score']):>8.0f}"
        )
        write(line, size=9)
    write("", gap=0.5)

    write("Highlights", size=12, weight="bold")
    highlights = [
        f"Best RMSE: {summary['best_rmse_subset']} ({summary['best_rmse']:.4f})",
        f"Best NASA score: {summary['best_nasa_subset']} ({summary['best_nasa']:.2f})",
        f"Best Phase2 F1: {summary['best_phase2_subset']} ({summary['best_phase2_f1']:.4f})",
        f"Average RMSE={summary['avg_rmse']:.4f}, MAE={summary['avg_mae']:.4f}, NASA={summary['avg_nasa']:.2f}",
        f"Average anomaly F1: phase1={summary['avg_phase1_f1']:.4f}, phase2={summary['avg_phase2_f1']:.4f}",
        f"Deployment split: phase2={summary['phase2_reco_count']} subsets, phase1={summary['phase1_reco_count']} subsets",
    ]
    for h in highlights:
        write(fill(f"- {h}", width=100), size=10)
    write("", gap=0.5)

    write("Relevance to UCC PhD", size=12, weight="bold")
    for b in [
        "Fault detection: robust tabular and sequence anomaly modeling.",
        "Diagnostics: interpretable root-cause indicators via SHAP and sensor-level error deltas.",
        "Predictive maintenance: run-to-failure RUL forecasting with domain-relevant NASA scoring.",
        "Industrial readiness: reproducible, config-driven pipeline with deploy-run validation.",
    ]:
        write(fill(f"- {b}", width=100), size=10)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ax.text(
        0.05,
        0.03,
        f"Generated automatically from benchmark artifacts on {generated}",
        fontsize=8,
        va="bottom",
        ha="left",
        family="DejaVu Sans",
        color="#555555",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_benchmark_rows(args.benchmark_csv)
    recommendations = load_recommendations(args.recommendations_json)
    summary = build_summary(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "project_evidence_ucc.md"
    pdf_path = args.output_dir / "project_evidence_ucc.pdf"

    md = build_md(
        project_title=args.project_title,
        rows=rows,
        recommendations=recommendations,
        summary=summary,
    )
    md_path.write_text(md, encoding="utf-8")
    render_pdf(
        output_path=pdf_path,
        project_title=args.project_title,
        rows=rows,
        summary=summary,
    )

    print(f"Generated Markdown brief: {md_path}")
    print(f"Generated PDF brief: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
