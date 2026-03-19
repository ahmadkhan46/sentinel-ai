from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_anomaly_scores(scores: np.ndarray, title: str = "Anomaly Score Over Time") -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.tight_layout()


def plot_rul_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "RUL: True vs Predicted") -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.title(title)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.tight_layout()


# ── B7: Enhanced visualisations ───────────────────────────────────────────────

def plot_fault_timeline(
    engine_df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    rul_preds: np.ndarray,
    threshold: float,
    proxy_labels: np.ndarray,
    top_sensors: list[str],
    engine_id: int,
    output_path: Path,
    rul_true: np.ndarray | None = None,
) -> None:
    """
    3-panel fault timeline for a single engine:
      Panel 1 — top sensors over time with anomalous region shaded
      Panel 2 — anomaly score with threshold line and alert markers
      Panel 3 — RUL prediction (+ ground truth if available)

    This is the most visually compelling output for the application.
    """
    cycles = engine_df["cycle"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── Panel 1: Sensor trajectories ─────────────────────────────────────────
    ax1 = axes[0]
    colors = ["#4C78A8", "#72B7B2", "#F58518"]
    for i, sensor in enumerate(top_sensors[:3]):
        if sensor in engine_df.columns:
            ax1.plot(cycles, engine_df[sensor].to_numpy(),
                     label=sensor, color=colors[i % len(colors)])
    # Shade anomalous region
    anomalous_cycles = cycles[proxy_labels == 1]
    if len(anomalous_cycles):
        ax1.axvspan(anomalous_cycles[0], anomalous_cycles[-1],
                    alpha=0.15, color="red", label="anomaly window")
    ax1.set_ylabel("Sensor value (scaled)")
    ax1.set_title(f"Fault Timeline — Engine {engine_id}")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # ── Panel 2: Anomaly score ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(cycles, anomaly_scores, color="#E45756", linewidth=1.5, label="anomaly score")
    ax2.axhline(threshold, color="orange", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.3f}")
    alert_mask = anomaly_scores >= threshold
    if alert_mask.any():
        first_alert = cycles[np.argmax(alert_mask)]
        ax2.axvline(first_alert, color="red", linestyle=":", linewidth=1.5,
                    label=f"first alert (cycle {first_alert})")
    ax2.set_ylabel("Anomaly score (raw)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    # ── Panel 3: RUL prediction ───────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(cycles, rul_preds, color="#54A24B", linewidth=1.5, label="predicted RUL")
    if rul_true is not None:
        ax3.plot(cycles, rul_true, color="steelblue", linestyle="--",
                 linewidth=1.2, alpha=0.8, label="true RUL")
    ax3.set_ylabel("RUL (cycles)")
    ax3.set_xlabel("Cycle")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sensor_correlation_heatmap(
    healthy_df: pd.DataFrame,
    anomalous_df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """
    Side-by-side sensor correlation matrices for healthy vs anomalous phases.
    Reveals which sensor relationships break down during degradation.
    """
    import seaborn as sns

    sensor_cols = [c for c in feature_cols if c.startswith("sensor_")][:12]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df, label in zip(axes, [healthy_df, anomalous_df], ["Healthy", "Anomalous"]):
        if df.empty or not all(c in df.columns for c in sensor_cols):
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        corr = df[sensor_cols].corr()
        sns.heatmap(
            corr, ax=ax, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.3,
            xticklabels=[c.replace("sensor_", "s") for c in sensor_cols],
            yticklabels=[c.replace("sensor_", "s") for c in sensor_cols],
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{label} Phase — Sensor Correlations", fontsize=11)

    fig.suptitle("Sensor Correlation: Healthy vs Anomalous", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_health_index_trajectory(
    engine_df: pd.DataFrame,
    health_index: np.ndarray,
    engine_id: int,
    output_path: Path,
) -> None:
    """HI over time with degradation zones coloured."""
    cycles = engine_df["cycle"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_between(cycles, health_index, alpha=0.3, color="#54A24B")
    ax.plot(cycles, health_index, color="#54A24B", linewidth=1.5)
    ax.axhline(0.7, color="orange", linestyle="--", linewidth=1, label="caution (0.7)")
    ax.axhline(0.4, color="red", linestyle="--", linewidth=1, label="critical (0.4)")
    ax.fill_between(cycles, 0, 0.4, alpha=0.08, color="red")
    ax.fill_between(cycles, 0.4, 0.7, alpha=0.08, color="orange")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health Index (1=healthy, 0=failed)")
    ax.set_title(f"Health Index Trajectory — Engine {engine_id}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_digital_twin_comparison(
    results: dict,
    output_path: Path,
) -> None:
    """
    Compare RUL and anomaly score trajectories across what-if scenarios.
    Each scenario gets its own line; alert timings are marked.
    """
    colors = {
        "continue": "#E45756",
        "reduce_load": "#F58518",
        "maintenance_now": "#54A24B",
        "baseline": "#4C78A8",
    }
    labels = {
        "continue": "Continue (current load)",
        "reduce_load": "Reduce load",
        "maintenance_now": "Maintenance now",
        "baseline": "Baseline",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    for scenario, result in results.items():
        color = colors.get(scenario, "#888888")
        label = labels.get(scenario, scenario)
        axes[0].plot(result.cycles, result.rul_predictions, color=color, label=label, linewidth=1.8)
        axes[1].plot(result.cycles, result.anomaly_scores, color=color, label=label, linewidth=1.8)
        if result.alert_cycle is not None:
            axes[1].axvline(result.alert_cycle, color=color, linestyle=":", linewidth=1.2)

    axes[0].set_ylabel("Predicted RUL (cycles)")
    axes[0].set_title("Digital Twin — What-If Scenario Comparison")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Anomaly score (normalised)")
    axes[1].set_xlabel("Simulated cycle")
    axes[1].axhline(0.5, color="orange", linestyle="--", linewidth=1, label="alert threshold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
