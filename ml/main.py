from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data.load_cmapss import load_cmapss_bundle
from ml.data.preprocess import (
    OperatingConditionNormaliser,
    add_train_rul,
    apply_scaler,
    default_feature_columns,
    drop_near_constant,
    fit_scaler,
    proxy_anomaly_labels,
    train_val_engine_split,
)
from ml.data.windowing import build_sequences_with_metadata
from ml.eval.metrics import (
    anomaly_prf,
    compute_maintenance_metrics,
    rul_regression_metrics,
)
from ml.eval.plots import (
    plot_fault_timeline,
    plot_rul_true_vs_pred,
    plot_sensor_correlation_heatmap,
)
from ml.models.anomaly.iforest import IForestDetector
from ml.models.anomaly.ocsvm import OCSVMDetector
from ml.models.rul.xgb_regressor import RULXGB, monotonic_rul_predictions
from ml.utils.config import load_config
from ml.utils.seed import set_seed


def _has_explicit_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_phase1_report(report_path: Path, metrics: dict[str, Any]) -> None:
    selected_name = metrics["meta"]["selected_anomaly_model"]
    selected_metrics = metrics["anomaly_models"][selected_name]
    selected_threshold = selected_metrics.get("threshold", 0.0)
    threshold_source = selected_metrics.get("threshold_source", "quantile")
    lines = [
        "# Phase 1 Results",
        "",
        "## Baseline Metrics",
        "",
        f"- RUL MAE: {metrics['rul']['mae']:.4f}",
        f"- RUL RMSE: {metrics['rul']['rmse']:.4f}",
        f"- RUL NASA score (lower is better): {metrics['rul']['nasa_score']:.4f}",
        f"- Selected anomaly model: {selected_name}",
        f"- Anomaly Precision (proxy): {selected_metrics['precision']:.4f}",
        f"- Anomaly Recall (proxy): {selected_metrics['recall']:.4f}",
        f"- Anomaly F1 (proxy): {selected_metrics['f1']:.4f}",
        f"- Selected threshold: {selected_threshold:.6f} ({threshold_source})",
        "",
        "## Anomaly Model Comparison",
        "",
        f"- IsolationForest F1: {metrics['anomaly_models']['iforest']['f1']:.4f}",
        f"- OneClassSVM F1: {metrics['anomaly_models']['ocsvm']['f1']:.4f}",
        "",
        "## Notes",
        "",
        "- Anomaly labels are proxy labels based on the final degradation horizon per engine.",
        "- The highest-F1 anomaly model is selected automatically for reporting.",
        "- This phase is intended as a baseline before sequence models and explainability modules.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_phase2_report(report_path: Path, metrics: dict[str, Any]) -> None:
    model_type = str(metrics["meta"].get("model_type", "lstm")).upper()
    threshold = float(metrics["meta"].get("threshold", 0.0))
    threshold_source = str(metrics["meta"].get("threshold_source", "quantile"))
    lines = [
        "# Phase 2 Results",
        "",
        f"## {model_type} Autoencoder (Sequence Anomaly Detection)",
        "",
        f"- Sequence anomaly precision (proxy): {metrics['anomaly']['precision']:.4f}",
        f"- Sequence anomaly recall (proxy): {metrics['anomaly']['recall']:.4f}",
        f"- Sequence anomaly F1 (proxy): {metrics['anomaly']['f1']:.4f}",
        f"- Threshold: {threshold:.6f} ({threshold_source})",
        "",
        "## Training",
        "",
        f"- Epochs: {metrics['meta']['epochs']}",
        f"- Hidden size: {metrics['meta']['hidden_size']}",
        f"- Device: {metrics['meta']['device']}",
        "",
        "## Notes",
        "",
        "- Labels are proxy labels based on the final degradation horizon.",
        "- This phase introduces sequence-aware anomaly modeling beyond tabular baselines.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_phase3_report(
    report_path: Path,
    shap_df: pd.DataFrame,
    sensor_df: pd.DataFrame | None,
    diagnostics: dict[str, Any],
) -> None:
    top_shap = shap_df.head(5)
    lines = [
        "# Phase 3 Results",
        "",
        "## SHAP Diagnostics (RUL Model)",
        "",
    ]
    for _, row in top_shap.iterrows():
        lines.append(f"- {row['feature']}: {row['mean_abs_shap']:.6f}")

    if sensor_df is not None and not sensor_df.empty:
        top_sensor = sensor_df.head(5)
        lines.extend(
            [
                "",
                "## Autoencoder Sensor Diagnostics",
                "",
            ]
        )
        for _, row in top_sensor.iterrows():
            lines.append(f"- {row['feature']}: anomalous_error={row['anomalous_mean_recon_error']:.6f}")
    else:
        lines.extend(["", "## Autoencoder Sensor Diagnostics", "", "- Skipped (Phase 2 artifacts unavailable)."])

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- SHAP explains feature impact on predicted RUL for the XGBoost model.",
            "- Autoencoder diagnostics rank sensors that contribute most to anomalous sequence reconstruction error.",
            f"- SHAP samples: {diagnostics['shap_sample_count']}, background: {diagnostics['shap_background_count']}.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_sensor_trajectories(train_df: pd.DataFrame, reports_dir: Path) -> None:
    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")][:3]
    engine_ids = train_df["engine_id"].drop_duplicates().sort_values().head(3).tolist()
    if not sensor_cols or not engine_ids:
        return

    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(10, 9), sharex=True)
    if len(sensor_cols) == 1:
        axes = [axes]

    for idx, sensor in enumerate(sensor_cols):
        ax = axes[idx]
        for engine_id in engine_ids:
            engine_df = train_df[train_df["engine_id"] == engine_id].sort_values("cycle")
            ax.plot(engine_df["cycle"], engine_df[sensor], label=f"engine_{engine_id}")
        ax.set_ylabel(sensor)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Cycle")
    fig.suptitle("FD001 Sensor Trajectories (sample engines)")
    fig.tight_layout()
    fig.savefig(reports_dir / "sensor_trajectories.png", dpi=150)
    plt.close(fig)


def _plot_anomaly_engine_timeline(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    proxy_labels: np.ndarray,
    reports_dir: Path,
) -> None:
    scored_df = val_df.copy()
    scored_df["anomaly_score"] = val_scores
    scored_df["proxy_label"] = proxy_labels
    if scored_df.empty:
        return

    engine_id = int(scored_df["engine_id"].iloc[0])
    plot_df = scored_df[scored_df["engine_id"] == engine_id].sort_values("cycle")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["cycle"], plot_df["anomaly_score"], label="anomaly_score")
    anomalous = plot_df[plot_df["proxy_label"] == 1]
    ax.scatter(anomalous["cycle"], anomalous["anomaly_score"], c="red", s=10, label="proxy anomaly")
    ax.set_title(f"Anomaly Score Timeline (engine {engine_id})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports_dir / "anomaly_timeline_engine_sample.png", dpi=150)
    plt.close(fig)


def _plot_phase1_anomaly_model_f1(model_metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    labels = []
    values = []
    for model_name, payload in model_metrics.items():
        labels.append(model_name)
        values.append(float(payload["f1"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 (proxy)")
    ax.set_title("Phase 1 Anomaly Model Comparison")
    for idx, val in enumerate(values):
        ax.text(idx, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_phase2_training_loss(loss_history: list[float], reports_dir: Path, model_type: str = "lstm") -> None:
    if not loss_history:
        return
    model_name = model_type.lower()
    title = f"{model_name.upper()} Autoencoder Training Loss"
    output_name = f"phase2_{model_name}_training_loss.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(reports_dir / output_name, dpi=150)
    plt.close(fig)


def _plot_phase2_engine_timeline(
    meta_df: pd.DataFrame,
    seq_scores: np.ndarray,
    proxy_labels: np.ndarray,
    reports_dir: Path,
) -> None:
    if meta_df.empty:
        return

    scored = meta_df.copy()
    scored["anomaly_score"] = seq_scores
    scored["proxy_label"] = proxy_labels
    engine_id = int(scored["engine_id"].iloc[0])
    plot_df = scored[scored["engine_id"] == engine_id].sort_values("end_cycle")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df["end_cycle"], plot_df["anomaly_score"], label="sequence anomaly score")
    anomalous = plot_df[plot_df["proxy_label"] == 1]
    ax.scatter(anomalous["end_cycle"], anomalous["anomaly_score"], c="red", s=10, label="proxy anomaly")
    ax.set_title(f"Phase 2 Sequence Timeline (engine {engine_id})")
    ax.set_xlabel("Sequence End Cycle")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "phase2_sequence_timeline_engine_sample.png", dpi=150)
    plt.close(fig)


def _plot_top_feature_bar(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    top_k: int = 10,
) -> None:
    if df.empty:
        return
    top_df = df.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["feature"], top_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _run_phase2(
    cfg: dict[str, Any],
    feature_cols: list[str],
    train_scaled: pd.DataFrame,
    val_scaled: pd.DataFrame,
    seed: int,
    base_reports_dir: Path,
) -> dict[str, Any] | None:
    phase2_cfg = cfg.get("phase2", {})
    if not bool(phase2_cfg.get("enabled", True)):
        return None

    sequence_length = int(cfg.get("dataset", {}).get("sequence_length", 30))
    horizon_cycles = int(phase2_cfg.get("anomaly_proxy_horizon", 20))
    healthy_cycle_fraction = float(phase2_cfg.get("healthy_cycle_fraction", 0.3))
    threshold_quantile = float(phase2_cfg.get("threshold_quantile", 0.97))
    hidden_size = int(phase2_cfg.get("hidden_size", 64))
    epochs = int(phase2_cfg.get("epochs", 20))
    batch_size = int(phase2_cfg.get("batch_size", 128))
    learning_rate = float(phase2_cfg.get("learning_rate", 1e-3))
    device = str(phase2_cfg.get("device", "auto"))
    fixed_threshold_raw = phase2_cfg.get("fixed_threshold")
    model_type = str(phase2_cfg.get("model_type", "lstm")).lower()
    if model_type not in {"lstm", "gru"}:
        print(f"Phase 2 invalid model_type '{model_type}', falling back to 'lstm'.")
        model_type = "lstm"

    phase2_reports_dir = base_reports_dir / "phase2"
    phase2_reports_dir.mkdir(parents=True, exist_ok=True)

    train_x_seq, train_meta = build_sequences_with_metadata(
        df=train_scaled,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
        target_col="rul",
    )
    val_x_seq, val_meta = build_sequences_with_metadata(
        df=val_scaled,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
        target_col="rul",
    )

    if train_x_seq.size == 0 or val_x_seq.size == 0:
        print("Phase 2 skipped: not enough rows to build sequences.")
        return None

    healthy_train_mask = train_meta["end_cycle"] <= (healthy_cycle_fraction * train_meta["max_cycle"])
    healthy_train_x = train_x_seq[healthy_train_mask.to_numpy()]
    if healthy_train_x.size == 0:
        healthy_train_x = train_x_seq

    try:
        from ml.models.anomaly.lstm_autoencoder import (
            GRUAutoencoder,
            LSTMAutoencoder,
            reconstruction_error,
            resolve_device,
            train_autoencoder,
        )
    except ImportError as exc:
        print(f"Phase 2 skipped: missing dependency for LSTM autoencoder ({exc}).")
        return None

    model = (
        LSTMAutoencoder(n_features=len(feature_cols), hidden_size=hidden_size)
        if model_type == "lstm"
        else GRUAutoencoder(n_features=len(feature_cols), hidden_size=hidden_size)
    )
    loss_history = train_autoencoder(
        model=model,
        x_train=healthy_train_x,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    train_healthy_scores = reconstruction_error(model, healthy_train_x, device=device)
    if _has_explicit_value(fixed_threshold_raw):
        threshold = float(fixed_threshold_raw)
        threshold_source = "fixed_config"
    else:
        threshold = float(np.quantile(train_healthy_scores, threshold_quantile))
        threshold_source = "quantile_train_healthy"

    val_scores = reconstruction_error(model, val_x_seq, device=device)
    val_proxy_labels = (val_meta["end_cycle"] >= (val_meta["max_cycle"] - horizon_cycles + 1)).astype(int).to_numpy()
    val_preds = (val_scores >= threshold).astype(int)
    anomaly_metrics = anomaly_prf(val_proxy_labels, val_preds)

    _plot_phase2_training_loss(loss_history, phase2_reports_dir, model_type=model_type)
    _plot_phase2_engine_timeline(val_meta, val_scores, val_proxy_labels, phase2_reports_dir)

    metrics_payload: dict[str, Any] = {
        "anomaly": anomaly_metrics,
        "meta": {
            "sequence_length": sequence_length,
            "train_sequence_count": int(train_x_seq.shape[0]),
            "val_sequence_count": int(val_x_seq.shape[0]),
            "healthy_train_sequence_count": int(healthy_train_x.shape[0]),
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "threshold_source": threshold_source,
            "anomaly_proxy_horizon_cycles": horizon_cycles,
            "healthy_cycle_fraction": healthy_cycle_fraction,
            "hidden_size": hidden_size,
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": resolve_device(device),
            "seed": seed,
        },
        "training": {
            "loss_history": [float(x) for x in loss_history],
        },
    }
    _save_json(phase2_reports_dir / "phase2_metrics.json", metrics_payload)
    _write_phase2_report(phase2_reports_dir / "results.md", metrics_payload)
    val_meta.assign(
        anomaly_score=val_scores,
        proxy_label=val_proxy_labels,
        predicted_label=val_preds,
    ).to_csv(phase2_reports_dir / "sequence_scores.csv", index=False)

    print(f"Phase 2 complete. Outputs saved to: {phase2_reports_dir}")
    return {
        "model": model,
        "device": device,
        "val_x_seq": val_x_seq,
        "val_meta": val_meta,
        "val_scores": val_scores,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "metrics": metrics_payload,
        "model_type": model_type,
    }


def _run_phase3(
    cfg: dict[str, Any],
    feature_cols: list[str],
    x_train_rul: np.ndarray,
    x_val_rul: np.ndarray,
    rul_model: RULXGB,
    phase2_artifacts: dict[str, Any] | None,
    base_reports_dir: Path,
    seed: int,
    # Phase 1 artifacts for enhanced plots
    val_scaled: pd.DataFrame | None = None,
    val_rul_pred: np.ndarray | None = None,
    y_val_rul: np.ndarray | None = None,
    selected_scores: np.ndarray | None = None,
    proxy_labels: np.ndarray | None = None,
    selected_threshold: float | None = None,
) -> None:
    phase3_cfg = cfg.get("phase3", {})
    if not bool(phase3_cfg.get("enabled", True)):
        return

    phase3_reports_dir = base_reports_dir / "phase3"
    phase3_reports_dir.mkdir(parents=True, exist_ok=True)

    top_k_features = int(phase3_cfg.get("top_k_features", 10))
    shap_background_size = int(phase3_cfg.get("shap_background_size", 1000))
    shap_sample_size = int(phase3_cfg.get("shap_sample_size", 2000))
    seq_diag_quantile = float(phase3_cfg.get("sequence_diagnostics_quantile", 0.95))
    rng = np.random.default_rng(seed)

    try:
        from ml.explain.shap_explain import compute_shap_values, mean_abs_shap
    except ImportError as exc:
        print(f"Phase 3 skipped: SHAP utilities unavailable ({exc}).")
        return

    bg_n = min(shap_background_size, x_train_rul.shape[0])
    target_n = min(shap_sample_size, x_val_rul.shape[0])
    bg_idx = rng.choice(x_train_rul.shape[0], size=bg_n, replace=False)
    target_idx = rng.choice(x_val_rul.shape[0], size=target_n, replace=False)
    x_background = x_train_rul[bg_idx]
    x_target = x_val_rul[target_idx]

    shap_explanation = compute_shap_values(
        rul_model.model,
        x_background=x_background,
        x_target=x_target,
        silent=True,
    )
    mean_abs = mean_abs_shap(shap_explanation)
    shap_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    shap_df.to_csv(phase3_reports_dir / "shap_feature_importance.csv", index=False)
    _plot_top_feature_bar(
        df=shap_df,
        value_col="mean_abs_shap",
        title="Top SHAP Features for RUL Model",
        output_path=phase3_reports_dir / "shap_top_features.png",
        top_k=top_k_features,
    )

    sensor_df: pd.DataFrame | None = None
    if phase2_artifacts is not None:
        try:
            from ml.explain.recon_error import sequence_sensor_reconstruction_contrib
            from ml.models.anomaly.lstm_autoencoder import reconstruct
        except ImportError as exc:
            print(f"Phase 3 sensor diagnostics skipped: missing dependency ({exc}).")
        else:
            val_x_seq = phase2_artifacts["val_x_seq"]
            val_scores = phase2_artifacts["val_scores"]
            val_meta = phase2_artifacts["val_meta"]
            ae_model = phase2_artifacts["model"]
            ae_device = phase2_artifacts["device"]

            val_recon = reconstruct(ae_model, val_x_seq, device=ae_device)
            per_seq_sensor = sequence_sensor_reconstruction_contrib(val_x_seq, val_recon)

            anomaly_cutoff = float(np.quantile(val_scores, seq_diag_quantile))
            anomalous_mask = val_scores >= anomaly_cutoff
            normal_mask = ~anomalous_mask

            anomalous_mean = per_seq_sensor[anomalous_mask].mean(axis=0)
            normal_mean = per_seq_sensor[normal_mask].mean(axis=0) if normal_mask.any() else np.zeros_like(
                anomalous_mean
            )
            diff_mean = anomalous_mean - normal_mean

            sensor_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "anomalous_mean_recon_error": anomalous_mean,
                    "normal_mean_recon_error": normal_mean,
                    "delta_recon_error": diff_mean,
                }
            ).sort_values("anomalous_mean_recon_error", ascending=False)
            sensor_df.to_csv(phase3_reports_dir / "autoencoder_sensor_contributions.csv", index=False)
            _plot_top_feature_bar(
                df=sensor_df.sort_values("delta_recon_error", ascending=False),
                value_col="delta_recon_error",
                title="Top Sensor Reconstruction Error Delta (Anomalous - Normal)",
                output_path=phase3_reports_dir / "autoencoder_sensor_deltas.png",
                top_k=top_k_features,
            )

            worst_idx = int(np.argmax(val_scores))
            worst_row = val_meta.iloc[worst_idx]
            worst_sensor_df = pd.DataFrame(
                {
                    "feature": feature_cols,
                    "recon_error": per_seq_sensor[worst_idx],
                }
            ).sort_values("recon_error", ascending=False)
            worst_sensor_df.to_csv(phase3_reports_dir / "worst_sequence_sensor_rank.csv", index=False)
            _save_json(
                phase3_reports_dir / "worst_sequence_metadata.json",
                {
                    "engine_id": int(worst_row["engine_id"]),
                    "start_cycle": int(worst_row["start_cycle"]),
                    "end_cycle": int(worst_row["end_cycle"]),
                    "anomaly_score": float(val_scores[worst_idx]),
                    "quantile_cutoff": anomaly_cutoff,
                },
            )

    diagnostics_payload = {
        "seed": seed,
        "top_k_features": top_k_features,
        "shap_background_count": bg_n,
        "shap_sample_count": target_n,
        "sequence_diagnostics_quantile": seq_diag_quantile,
        "autoencoder_diagnostics_available": sensor_df is not None,
    }
    _save_json(phase3_reports_dir / "phase3_summary.json", diagnostics_payload)
    _write_phase3_report(
        report_path=phase3_reports_dir / "results.md",
        shap_df=shap_df,
        sensor_df=sensor_df,
        diagnostics=diagnostics_payload,
    )

    # B7: Fault timeline plots (per engine) — requires Phase 1 artifacts
    top_sensors = shap_df["feature"].head(3).tolist() if not shap_df.empty else feature_cols[:3]
    if val_scaled is not None and val_rul_pred is not None and selected_scores is not None:
        try:
            sample_engines = val_scaled["engine_id"].unique()[:3]
            for eng_id in sample_engines:
                eng_mask = (val_scaled["engine_id"] == eng_id).to_numpy()
                eng_df = val_scaled[val_scaled["engine_id"] == eng_id].sort_values("cycle")

                eng_scores = selected_scores[eng_mask]
                eng_rul_pred = val_rul_pred[eng_mask]
                eng_rul_true = y_val_rul[eng_mask] if y_val_rul is not None else None
                eng_labels = proxy_labels[eng_mask] if proxy_labels is not None else np.zeros(eng_mask.sum(), int)
                thr = selected_threshold if selected_threshold is not None else float(np.quantile(selected_scores, 0.97))
                plot_fault_timeline(
                    engine_df=eng_df,
                    anomaly_scores=eng_scores,
                    rul_preds=eng_rul_pred,
                    threshold=thr,
                    proxy_labels=eng_labels,
                    top_sensors=top_sensors,
                    engine_id=int(eng_id),
                    output_path=phase3_reports_dir / f"fault_timeline_engine_{int(eng_id)}.png",
                    rul_true=eng_rul_true,
                )
        except Exception as exc:
            print(f"  Fault timeline plots skipped: {exc}")

    # B4: LIME explainability alongside SHAP
    phase3_lime_enabled = bool(phase3_cfg.get("lime_enabled", True))
    if phase3_lime_enabled:
        try:
            from ml.explain.lime_explain import LIMEExplainer

            lime_explainer = LIMEExplainer(feature_cols=feature_cols, seed=seed)
            bg_n_lime = min(500, x_train_rul.shape[0])
            rng_lime = np.random.default_rng(seed)
            lime_bg_idx = rng_lime.choice(x_train_rul.shape[0], size=bg_n_lime, replace=False)
            lime_explainer.fit(x_train_rul[lime_bg_idx])

            # Explain a representative instance (median-anomaly-score validation row)
            if x_val_rul.shape[0] > 0:
                mid_idx = int(np.argsort(np.abs(shap_df["mean_abs_shap"].to_numpy() - shap_df["mean_abs_shap"].mean()))[0])
                sample_idx = min(mid_idx, x_val_rul.shape[0] - 1)
                lime_attrs = lime_explainer.explain(
                    x_val_rul[sample_idx],
                    predict_fn=rul_model.predict,
                    num_features=top_k_features,
                )
                shap_dict_for_lime = dict(zip(shap_df["feature"].tolist(), shap_df["mean_abs_shap"].tolist()))
                comparison = lime_explainer.compare_with_shap(lime_attrs, shap_dict_for_lime)
                summary = lime_explainer.agreement_summary(comparison)

                _save_json(phase3_reports_dir / "lime_shap_comparison.json", {
                    "lime_attrs": lime_attrs,
                    "comparison": comparison,
                    "agreement_summary": summary,
                })
                print(f"  LIME vs SHAP: {summary['agreement_rate']:.0%} agreement "
                      f"({len(summary['agreed_features'])} of {summary['n_features']} sensors agree).")
        except Exception as exc:
            print(f"  LIME explainability skipped: {exc}")

    print(f"Phase 3 complete. Outputs saved to: {phase3_reports_dir}")


def run_pipeline(config_path: Path) -> None:
    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Config load/validation failed: {exc}")
        return
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    paths_cfg = cfg.get("paths", {})
    data_dir = Path(paths_cfg.get("data_dir", "data"))
    reports_base_dir = Path(paths_cfg.get("reports_dir", "reports"))
    phase1_reports_dir = reports_base_dir / "phase1"
    phase1_reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("dataset", {})
    subset = str(dataset_cfg.get("subset", "FD001"))
    rul_clip = int(dataset_cfg.get("target_rul_clip", 125))
    phase1_cfg = cfg.get("phase1", {})
    val_ratio = float(phase1_cfg.get("val_ratio", 0.2))
    anomaly_horizon = int(phase1_cfg.get("anomaly_proxy_horizon", 20))
    healthy_cycle_fraction = float(phase1_cfg.get("healthy_cycle_fraction", 0.3))

    try:
        train_df, test_df, test_rul_df = load_cmapss_bundle(data_dir=data_dir, subset=subset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset load failed: {exc}")
        print("Expected files under data/:")
        print(f"- train_{subset}.txt")
        print(f"- test_{subset}.txt")
        print(f"- RUL_{subset}.txt")
        return

    feature_cols = default_feature_columns(train_df)
    feature_cols = drop_near_constant(train_df, feature_cols)

    train_df = add_train_rul(train_df, clip_upper=rul_clip)
    train_split, val_split = train_val_engine_split(train_df, val_ratio=val_ratio, seed=seed)

    # B1: Physics-informed operating condition normalisation (optional)
    use_oc_norm = bool(dataset_cfg.get("operating_condition_normalisation", False))
    oc_normaliser = None
    sensor_cols_for_oc = [c for c in feature_cols if c.startswith("sensor_")]
    if use_oc_norm and sensor_cols_for_oc:
        max_cycle_train = train_split.groupby("engine_id")["cycle"].transform("max")
        healthy_mask_train = train_split["cycle"] <= (healthy_cycle_fraction * max_cycle_train)
        oc_normaliser = OperatingConditionNormaliser(n_regimes=6, seed=seed)
        oc_normaliser.fit(train_split, sensor_cols_for_oc, healthy_mask_train)
        train_split = oc_normaliser.transform(train_split, sensor_cols_for_oc)
        val_split = oc_normaliser.transform(val_split, sensor_cols_for_oc)
        print(f"  OC normalisation: 6 regimes, {len(sensor_cols_for_oc)} sensors normalised.")

    scaler = fit_scaler(train_split, feature_cols)
    train_scaled = apply_scaler(train_split, feature_cols, scaler)
    val_scaled = apply_scaler(val_split, feature_cols, scaler)

    # B3: Scalar Health Index (optional — computed for reporting, not added to features by default)
    use_health_index = bool(dataset_cfg.get("health_index_feature", False))
    health_index_val: np.ndarray | None = None
    if use_health_index:
        max_c = train_scaled.groupby("engine_id")["cycle"].transform("max")
        hi_healthy_mask = train_scaled["cycle"] <= (healthy_cycle_fraction * max_c)
        # Fit on training, transform validation
        from sklearn.decomposition import PCA
        hi_train_healthy = train_scaled.loc[hi_healthy_mask, feature_cols].to_numpy(dtype=float)
        pca_hi = PCA(n_components=1)
        pca_hi.fit(hi_train_healthy)
        hi_val_raw = pca_hi.transform(val_scaled[feature_cols].to_numpy(dtype=float)).ravel()
        hi_min, hi_max = hi_val_raw.min(), hi_val_raw.max()
        health_index_val = (hi_val_raw - hi_min) / max(hi_max - hi_min, 1e-9)
        # Flip so 1 = healthy
        val_hi_healthy_mask = val_scaled["cycle"] <= (healthy_cycle_fraction * val_scaled.groupby("engine_id")["cycle"].transform("max"))
        if health_index_val[val_hi_healthy_mask.to_numpy()].mean() < 0.5:
            health_index_val = 1.0 - health_index_val
        print(f"  Health index computed. Val range: [{health_index_val.min():.3f}, {health_index_val.max():.3f}]")

    profile = {
        "subset": subset,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_rul_rows": int(len(test_rul_df)),
        "train_engines": int(train_df["engine_id"].nunique()),
        "test_engines": int(test_df["engine_id"].nunique()),
        "val_rows": int(len(val_scaled)),
        "val_engines": int(val_scaled["engine_id"].nunique()),
        "feature_count": len(feature_cols),
    }
    _save_json(phase1_reports_dir / "dataset_profile.json", profile)

    train_df[feature_cols].describe().T.to_csv(phase1_reports_dir / "sensor_descriptive_stats.csv")
    train_df.groupby("engine_id")["cycle"].max().describe().to_csv(phase1_reports_dir / "engine_cycle_summary.csv")
    _plot_sensor_trajectories(train_df, phase1_reports_dir)

    anomaly_cfg = cfg.get("anomaly", {})
    iforest_cfg = anomaly_cfg.get("iforest", {})
    ocsvm_cfg = anomaly_cfg.get("ocsvm", {})
    iforest_fixed_threshold_raw = iforest_cfg.get("fixed_threshold")
    ocsvm_fixed_threshold_raw = ocsvm_cfg.get("fixed_threshold")
    threshold_quantile = float(anomaly_cfg.get("alert", {}).get("threshold_quantile", 0.97))

    max_cycle = train_scaled.groupby("engine_id")["cycle"].transform("max")
    healthy_mask = train_scaled["cycle"] <= (healthy_cycle_fraction * max_cycle)
    healthy_df = train_scaled.loc[healthy_mask]
    if healthy_df.empty:
        healthy_df = train_scaled

    x_healthy = healthy_df[feature_cols].to_numpy()
    x_val = val_scaled[feature_cols].to_numpy()
    proxy_labels = proxy_anomaly_labels(val_scaled, horizon_cycles=anomaly_horizon)

    iforest = IForestDetector(
        n_estimators=int(iforest_cfg.get("n_estimators", 200)),
        contamination=float(iforest_cfg.get("contamination", 0.03)),
        random_state=seed,
    )
    iforest.fit(x_healthy)
    iforest_healthy_scores = iforest.score(x_healthy)
    if _has_explicit_value(iforest_fixed_threshold_raw):
        iforest_threshold = float(iforest_fixed_threshold_raw)
        iforest_threshold_source = "fixed_config"
    else:
        iforest_threshold = float(np.quantile(iforest_healthy_scores, threshold_quantile))
        iforest_threshold_source = "quantile_train_healthy"
    iforest_val_scores = iforest.score(x_val)
    iforest_pred = (iforest_val_scores >= iforest_threshold).astype(int)
    iforest_metrics = anomaly_prf(proxy_labels, iforest_pred)

    ocsvm = OCSVMDetector(
        kernel=str(ocsvm_cfg.get("kernel", "rbf")),
        nu=float(ocsvm_cfg.get("nu", 0.03)),
        gamma=str(ocsvm_cfg.get("gamma", "scale")),
    )
    ocsvm.fit(x_healthy)
    ocsvm_healthy_scores = ocsvm.score(x_healthy)
    if _has_explicit_value(ocsvm_fixed_threshold_raw):
        ocsvm_threshold = float(ocsvm_fixed_threshold_raw)
        ocsvm_threshold_source = "fixed_config"
    else:
        ocsvm_threshold = float(np.quantile(ocsvm_healthy_scores, threshold_quantile))
        ocsvm_threshold_source = "quantile_train_healthy"
    ocsvm_val_scores = ocsvm.score(x_val)
    ocsvm_pred = (ocsvm_val_scores >= ocsvm_threshold).astype(int)
    ocsvm_metrics = anomaly_prf(proxy_labels, ocsvm_pred)

    anomaly_models = {
        "iforest": {
            **iforest_metrics,
            "threshold": iforest_threshold,
            "threshold_source": iforest_threshold_source,
        },
        "ocsvm": {
            **ocsvm_metrics,
            "threshold": ocsvm_threshold,
            "threshold_source": ocsvm_threshold_source,
        },
    }
    selected_anomaly_model = max(anomaly_models.keys(), key=lambda k: anomaly_models[k]["f1"])
    selected_scores = iforest_val_scores if selected_anomaly_model == "iforest" else ocsvm_val_scores

    _plot_anomaly_engine_timeline(val_scaled, selected_scores, proxy_labels, phase1_reports_dir)
    _plot_phase1_anomaly_model_f1(anomaly_models, phase1_reports_dir / "anomaly_model_f1_comparison.png")

    rul_cfg = cfg.get("rul", {}).get("xgboost", {})
    rul_model = RULXGB(
        n_estimators=int(rul_cfg.get("n_estimators", 500)),
        max_depth=int(rul_cfg.get("max_depth", 6)),
        learning_rate=float(rul_cfg.get("learning_rate", 0.03)),
        subsample=float(rul_cfg.get("subsample", 0.8)),
        colsample_bytree=float(rul_cfg.get("colsample_bytree", 0.8)),
        random_state=seed,
    )

    x_train_rul = train_scaled[feature_cols].to_numpy()
    y_train_rul = train_scaled["rul"].to_numpy()
    x_val_rul = val_scaled[feature_cols].to_numpy()
    y_val_rul = val_scaled["rul"].to_numpy()

    rul_model.fit(x_train_rul, y_train_rul)
    val_rul_pred = rul_model.predict(x_val_rul)

    # B2: Monotonic RUL post-processing (optional, physics-informed)
    fit_monotonic = bool(cfg.get("rul", {}).get("fit_monotonic", False))
    val_rul_pred_raw = val_rul_pred.copy()
    if fit_monotonic:
        val_rul_pred = monotonic_rul_predictions(val_rul_pred, val_scaled)
        rul_metrics_raw = rul_regression_metrics(y_val_rul, val_rul_pred_raw)
        rul_metrics = rul_regression_metrics(y_val_rul, val_rul_pred)
        print(
            f"  Monotonic RUL: RMSE {rul_metrics_raw['rmse']:.2f}->{rul_metrics['rmse']:.2f}, "
            f"NASA {rul_metrics_raw['nasa_score']:.0f}->{rul_metrics['nasa_score']:.0f}"
        )
    else:
        rul_metrics = rul_regression_metrics(y_val_rul, val_rul_pred)
        rul_metrics_raw = None

    # B5: Maintenance metrics
    selected_threshold = iforest_threshold if selected_anomaly_model == "iforest" else ocsvm_threshold
    maintenance_metrics = compute_maintenance_metrics(
        anomaly_scores=selected_scores,
        proxy_labels=proxy_labels,
        threshold=selected_threshold,
        val_df=val_scaled,
    )

    plot_rul_true_vs_pred(y_val_rul, val_rul_pred)
    plt.savefig(phase1_reports_dir / "rul_true_vs_pred.png", dpi=150)
    plt.close()

    # B7: Sensor correlation heatmap (healthy vs anomalous phases)
    try:
        anomalous_mask_bool = proxy_labels == 1
        healthy_mask_bool = proxy_labels == 0
        plot_sensor_correlation_heatmap(
            healthy_df=val_scaled.iloc[healthy_mask_bool],
            anomalous_df=val_scaled.iloc[anomalous_mask_bool],
            feature_cols=feature_cols,
            output_path=phase1_reports_dir / "sensor_correlation_heatmap.png",
        )
    except Exception as exc:
        print(f"  Sensor correlation heatmap skipped: {exc}")

    metrics_payload: dict[str, Any] = {
        "rul": rul_metrics,
        "rul_raw_before_monotonic": rul_metrics_raw,
        "anomaly": {
            "model": selected_anomaly_model,
            "precision": anomaly_models[selected_anomaly_model]["precision"],
            "recall": anomaly_models[selected_anomaly_model]["recall"],
            "f1": anomaly_models[selected_anomaly_model]["f1"],
        },
        "anomaly_models": anomaly_models,
        "maintenance": maintenance_metrics,
        "meta": {
            "selected_anomaly_model": selected_anomaly_model,
            "proxy_anomaly_horizon_cycles": anomaly_horizon,
            "threshold_quantile": threshold_quantile,
            "operating_condition_normalisation": use_oc_norm,
            "monotonic_rul": fit_monotonic,
            "health_index_enabled": use_health_index,
        },
    }
    _save_json(phase1_reports_dir / "baseline_metrics.json", metrics_payload)
    _write_phase1_report(phase1_reports_dir / "results.md", metrics_payload)

    print(f"Project: {cfg.get('project_name')}")
    print(f"Phase 1 complete. Outputs saved to: {phase1_reports_dir}")
    print("Generated: dataset profile, EDA tables/plots, and baseline metrics.")

    phase2_artifacts = _run_phase2(
        cfg=cfg,
        feature_cols=feature_cols,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        seed=seed,
        base_reports_dir=reports_base_dir,
    )
    _run_phase3(
        cfg=cfg,
        feature_cols=feature_cols,
        x_train_rul=x_train_rul,
        x_val_rul=x_val_rul,
        rul_model=rul_model,
        phase2_artifacts=phase2_artifacts,
        base_reports_dir=reports_base_dir,
        seed=seed,
        # Phase 1 artifacts for enhanced plots (B4, B7)
        val_scaled=val_scaled,
        val_rul_pred=val_rul_pred,
        y_val_rul=y_val_rul,
        selected_scores=selected_scores,
        proxy_labels=proxy_labels,
        selected_threshold=selected_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NASA C-MAPSS fault detection and predictive maintenance pipeline"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config)
