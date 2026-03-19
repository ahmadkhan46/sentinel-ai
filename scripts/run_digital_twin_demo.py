"""
scripts/run_digital_twin_demo.py

Demonstrates the SENTINEL Digital Twin on FD001 data.

Runs three what-if scenarios for a representative engine:
  1. continue       — operate at current load until failure
  2. reduce_load    — switch to lower-stress operating condition
  3. maintenance_now — reset to fresh engine (simulate post-maintenance state)

Outputs:
  reports/digital_twin/twin_demo.png        — side-by-side RUL + anomaly score
  reports/digital_twin/twin_comparison.json — numerical scenario comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from ml.data.load_cmapss import load_cmapss_bundle
from ml.data.preprocess import (
    add_train_rul,
    apply_scaler,
    default_feature_columns,
    drop_near_constant,
    fit_scaler,
    train_val_engine_split,
)
from ml.engine import ModelBundle
from ml.eval.plots import plot_digital_twin_comparison
from ml.models.anomaly.iforest import IForestDetector
from ml.models.rul.xgb_regressor import RULXGB
from ml.simulation.digital_twin import DigitalTwin
from ml.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    data_dir = ROOT_DIR / "data"
    out_dir = ROOT_DIR / "reports" / "digital_twin"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FD001 training data...")
    train_df, _, _ = load_cmapss_bundle(data_dir, "FD001")
    feature_cols = drop_near_constant(train_df, default_feature_columns(train_df))
    train_df = add_train_rul(train_df, clip_upper=125)
    train_split, val_split = train_val_engine_split(train_df, val_ratio=0.2, seed=42)
    scaler = fit_scaler(train_split, feature_cols)
    train_scaled = apply_scaler(train_split, feature_cols, scaler)
    val_scaled = apply_scaler(val_split, feature_cols, scaler)

    print("Training IForest anomaly model...")
    max_cycle = train_scaled.groupby("engine_id")["cycle"].transform("max")
    healthy = train_scaled[train_scaled["cycle"] <= 0.3 * max_cycle]
    iforest = IForestDetector(n_estimators=200, contamination=0.03, random_state=42)
    iforest.fit(healthy[feature_cols].to_numpy())
    healthy_scores = iforest.score(healthy[feature_cols].to_numpy())
    threshold = float(np.quantile(healthy_scores, 0.97))

    print("Training XGBoost RUL model...")
    rul_model = RULXGB(n_estimators=300, random_state=42)
    rul_model.fit(train_scaled[feature_cols].to_numpy(), train_scaled["rul"].to_numpy())
    val_pred = rul_model.predict(val_scaled[feature_cols].to_numpy())
    mae = float(np.mean(np.abs(val_pred - val_scaled["rul"].to_numpy())))

    bundle = ModelBundle(
        anomaly_model=iforest,
        rul_model=rul_model,
        scaler=scaler,
        feature_cols=feature_cols,
        threshold=threshold,
        model_type="iforest",
        healthy_score_mean=float(healthy_scores.mean()),
        healthy_score_std=float(healthy_scores.std()),
        training_mae=mae,
    )

    print("Building digital twin...")
    # DigitalTwin works on raw (unscaled) sensor readings — the engine handles scaling internally
    twin = DigitalTwin(bundle=bundle, train_df=train_split, feature_cols=feature_cols)
    twin.fit()

    # Use the last-seen state of the first validation engine as "current state" (unscaled)
    sample_engine = int(val_split["engine_id"].iloc[0])
    current_state = val_split[val_split["engine_id"] == sample_engine].sort_values("cycle").tail(35)

    # Scenario set 1: different initial health states (most meaningful for single-condition FD001)
    print("Running health-state scenarios...")
    health_scenarios = {}
    for health, label in [(1.0, "new_engine"), (0.6, "mid_life"), (0.25, "near_end_of_life")]:
        result = twin.simulate(
            initial_health=health, n_cycles=120, profile="medium_load", scenario_name=label
        )
        health_scenarios[label] = result
        alert_str = f"cycle {result.alert_cycle}" if result.alert_cycle else "no alert"
        print(f"  {label:22s}: final_RUL={result.final_rul:.1f}, alert={alert_str}")

    plot_digital_twin_comparison(health_scenarios, output_path=out_dir / "twin_demo.png")
    print(f"Plot saved: {out_dir / 'twin_demo.png'}")

    # Scenario set 2: operational what-if from current engine state
    print("\nRunning what-if scenarios from current engine state...")
    scenarios = twin.what_if(current_state=current_state, n_forward_cycles=100)

    for name, result in scenarios.items():
        alert_str = f"cycle {result.alert_cycle}" if result.alert_cycle else "no alert"
        print(f"  {name:20s}: final_RUL={result.final_rul:.1f}, alert={alert_str}")

    plot_digital_twin_comparison(scenarios, output_path=out_dir / "twin_whatif_scenarios.png")

    # Merge both sets for the comparison JSON
    all_scenarios = {**health_scenarios, **scenarios}

    # Save JSON comparison
    comparison = {}
    for name, result in all_scenarios.items():
        comparison[name] = {
            "final_rul_cycles": round(float(result.final_rul), 1),
            "alert_cycle": int(result.alert_cycle) if result.alert_cycle else None,
            "time_to_alert_cycles": int(result.time_to_alert) if result.time_to_alert else None,
            "n_simulated_cycles": int(len(result.cycles)),
            "summary": result.summary,
        }

    (out_dir / "twin_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(f"Comparison saved: {out_dir / 'twin_comparison.json'}")
    print("\nDigital twin demo complete.")


if __name__ == "__main__":
    main()
