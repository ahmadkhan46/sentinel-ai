"""
ml/models/registry.py — ModelRegistry

Manages trained model artefacts for multiple assets.
Storage layout: {root}/{asset_id}/{version}/
Each version directory contains:
  anomaly_model.pkl | (anomaly_model.pt + anomaly_arch.json)
  rul_model.ubj
  scaler.pkl
  config.yaml
  metadata.json
A production.json file at {root}/{asset_id}/production.json tracks the promoted version.
"""
from __future__ import annotations

import json
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ml.engine import ModelBundle
from ml.models.rul.xgb_regressor import RULXGB


# ── Data types ──────────────────────────────────────────────────────────────────

@dataclass
class ModelVersion:
    version: str
    asset_id: str
    model_type: str
    training_date: str
    metrics: dict[str, float]
    is_production: bool
    git_commit: str | None


@dataclass
class ComparisonReport:
    asset_id: str
    v1: ModelVersion
    v2: ModelVersion
    metric_deltas: dict[str, float]   # v2_value - v1_value for each metric
    recommendation: str               # "v1" | "v2" | "equivalent"


# ── ModelRegistry ───────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Save, load, version, and compare trained model bundles per asset.

    Example:
        registry = ModelRegistry()
        version = registry.save("asset_fd001", bundle, metrics)
        engine = SentinelEngine.from_registry("asset_fd001")
    """

    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _asset_dir(self, asset_id: str) -> Path:
        return self.root / asset_id

    def _version_dir(self, asset_id: str, version: str) -> Path:
        return self._asset_dir(asset_id) / version

    def _index_path(self, asset_id: str) -> Path:
        return self._asset_dir(asset_id) / "versions.json"

    def _load_index(self, asset_id: str) -> list[dict[str, Any]]:
        p = self._index_path(asset_id)
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_index(self, asset_id: str, index: list[dict[str, Any]]) -> None:
        self._index_path(asset_id).write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    def _resolve_version(self, asset_id: str, version: str) -> str:
        if version != "latest":
            return version
        index = self._load_index(asset_id)
        if not index:
            raise FileNotFoundError(f"No saved versions found for asset '{asset_id}'.")
        return index[-1]["version"]

    def _production_version(self, asset_id: str) -> str | None:
        prod_file = self._asset_dir(asset_id) / "production.json"
        if not prod_file.exists():
            return None
        return json.loads(prod_file.read_text(encoding="utf-8")).get("version")

    @staticmethod
    def _git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    def save(
        self,
        asset_id: str,
        bundle: ModelBundle,
        metrics: dict[str, float],
    ) -> str:
        """
        Persist a trained ModelBundle.
        Returns the version string (timestamp format: YYYYMMDD_HHMMSS).
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        vdir = self._version_dir(asset_id, version)
        vdir.mkdir(parents=True, exist_ok=True)

        model_type = bundle.model_type

        # Anomaly model
        if model_type in ("lstm", "gru"):
            import torch
            torch.save(bundle.anomaly_model.state_dict(), vdir / "anomaly_model.pt")
            # Store architecture params so we can reconstruct the nn.Module on load
            (vdir / "anomaly_arch.json").write_text(
                json.dumps({
                    "model_type": model_type,
                    "n_features": len(bundle.feature_cols),
                    "hidden_size": bundle.anomaly_model.encoder.hidden_size,
                }),
                encoding="utf-8",
            )
        else:
            with (vdir / "anomaly_model.pkl").open("wb") as fh:
                pickle.dump(bundle.anomaly_model, fh)

        # RUL model — XGBoost native binary format
        bundle.rul_model.model.save_model(str(vdir / "rul_model.ubj"))

        # Scaler
        with (vdir / "scaler.pkl").open("wb") as fh:
            pickle.dump(bundle.scaler, fh)

        # OC normaliser (optional, B1)
        if bundle.oc_normaliser is not None:
            with (vdir / "oc_normaliser.pkl").open("wb") as fh:
                pickle.dump(bundle.oc_normaliser, fh)

        # Config
        (vdir / "config.yaml").write_text(
            yaml.dump(bundle.config, default_flow_style=False),
            encoding="utf-8",
        )

        # Metadata
        metadata: dict[str, Any] = {
            **bundle.metadata,
            "version": version,
            "asset_id": asset_id,
            "model_type": model_type,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "feature_cols": bundle.feature_cols,
            "threshold": bundle.threshold,
            "sequence_length": bundle.sequence_length,
            "healthy_score_mean": bundle.healthy_score_mean,
            "healthy_score_std": bundle.healthy_score_std,
            "cycle_duration_hours": bundle.cycle_duration_hours,
            "training_mae": bundle.training_mae,
            "metrics": metrics,
            "git_commit": ModelRegistry._git_commit(),
        }
        (vdir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        # Append to index
        index = self._load_index(asset_id)
        index.append({
            "version": version,
            "model_type": model_type,
            "training_date": metadata["training_date"],
        })
        self._save_index(asset_id, index)

        return version

    def load(self, asset_id: str, version: str = "latest") -> ModelBundle:
        """Load and reconstruct a ModelBundle from disk."""
        version = self._resolve_version(asset_id, version)
        vdir = self._version_dir(asset_id, version)
        if not vdir.exists():
            raise FileNotFoundError(
                f"Model version '{version}' not found for asset '{asset_id}' "
                f"(expected directory: {vdir})."
            )

        meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
        model_type: str = meta["model_type"]
        feature_cols: list[str] = meta["feature_cols"]

        # Anomaly model
        if model_type in ("lstm", "gru"):
            import torch
            arch = json.loads((vdir / "anomaly_arch.json").read_text(encoding="utf-8"))
            if model_type == "lstm":
                from ml.models.anomaly.lstm_autoencoder import LSTMAutoencoder
                ae: Any = LSTMAutoencoder(
                    n_features=arch["n_features"], hidden_size=arch["hidden_size"]
                )
            else:
                from ml.models.anomaly.lstm_autoencoder import GRUAutoencoder
                ae = GRUAutoencoder(
                    n_features=arch["n_features"], hidden_size=arch["hidden_size"]
                )
            ae.load_state_dict(
                torch.load(str(vdir / "anomaly_model.pt"), map_location="cpu", weights_only=True)
            )
            ae.eval()
            anomaly_model = ae
        else:
            with (vdir / "anomaly_model.pkl").open("rb") as fh:
                anomaly_model = pickle.load(fh)

        # RUL model
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(vdir / "rul_model.ubj"))
        rul_model = RULXGB.__new__(RULXGB)
        rul_model.model = xgb_model

        # Scaler
        with (vdir / "scaler.pkl").open("rb") as fh:
            scaler = pickle.load(fh)

        config = (
            yaml.safe_load((vdir / "config.yaml").read_text(encoding="utf-8")) or {}
        )

        # OC normaliser (optional, B1)
        oc_normaliser = None
        oc_path = vdir / "oc_normaliser.pkl"
        if oc_path.exists():
            with oc_path.open("rb") as fh:
                oc_normaliser = pickle.load(fh)

        return ModelBundle(
            anomaly_model=anomaly_model,
            rul_model=rul_model,
            scaler=scaler,
            feature_cols=feature_cols,
            threshold=meta["threshold"],
            model_type=model_type,
            sequence_length=meta.get("sequence_length", 30),
            healthy_score_mean=meta.get("healthy_score_mean", 0.0),
            healthy_score_std=meta.get("healthy_score_std", 1.0),
            config=config,
            metadata=meta,
            cycle_duration_hours=meta.get("cycle_duration_hours"),
            training_mae=meta.get("training_mae"),
            oc_normaliser=oc_normaliser,
        )

    def list_versions(self, asset_id: str) -> list[ModelVersion]:
        """List all saved versions for an asset, newest last."""
        index = self._load_index(asset_id)
        prod = self._production_version(asset_id)
        versions: list[ModelVersion] = []
        for entry in index:
            v = entry["version"]
            try:
                meta = json.loads(
                    (self._version_dir(asset_id, v) / "metadata.json").read_text(
                        encoding="utf-8"
                    )
                )
                versions.append(
                    ModelVersion(
                        version=v,
                        asset_id=asset_id,
                        model_type=meta.get("model_type", "unknown"),
                        training_date=meta.get("training_date", ""),
                        metrics=meta.get("metrics", {}),
                        is_production=(v == prod),
                        git_commit=meta.get("git_commit"),
                    )
                )
            except Exception:
                # Skip corrupt or partially-written versions
                continue
        return versions

    def compare(self, asset_id: str, v1: str, v2: str) -> ComparisonReport:
        """
        Side-by-side metric comparison between two versions.
        Metrics where higher = better: f1, precision, recall.
        Metrics where lower = better: rmse, mae, nasa_score.
        """
        by_version = {mv.version: mv for mv in self.list_versions(asset_id)}
        for v in (v1, v2):
            if v not in by_version:
                raise ValueError(f"Version '{v}' not found for asset '{asset_id}'.")

        mv1, mv2 = by_version[v1], by_version[v2]
        all_keys = set(mv1.metrics) | set(mv2.metrics)
        deltas = {
            k: mv2.metrics.get(k, 0.0) - mv1.metrics.get(k, 0.0) for k in all_keys
        }

        # Compute signed improvement score (positive = v2 is better)
        score = 0.0
        for k, delta in deltas.items():
            if k in ("f1", "precision", "recall"):
                score += delta
            elif k in ("rmse", "mae", "nasa_score"):
                score -= delta

        recommendation = "v2" if score > 0 else ("v1" if score < 0 else "equivalent")
        return ComparisonReport(
            asset_id=asset_id,
            v1=mv1,
            v2=mv2,
            metric_deltas=deltas,
            recommendation=recommendation,
        )

    def promote(self, asset_id: str, version: str) -> None:
        """Tag a version as production. Used by the API model management endpoint."""
        known = [mv.version for mv in self.list_versions(asset_id)]
        if version not in known:
            raise ValueError(
                f"Cannot promote: version '{version}' not found for asset '{asset_id}'."
            )
        prod_file = self._asset_dir(asset_id) / "production.json"
        prod_file.write_text(json.dumps({"version": version}), encoding="utf-8")
