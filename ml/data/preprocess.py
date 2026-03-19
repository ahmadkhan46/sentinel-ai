from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("op_setting_") or c.startswith("sensor_")]
    if not cols:
        raise ValueError("No operating setting or sensor columns found in dataframe.")
    return cols


def drop_near_constant(df: pd.DataFrame, feature_cols: list[str], threshold: float = 1e-8) -> list[str]:
    kept = []
    for col in feature_cols:
        if df[col].var() > threshold:
            kept.append(col)
    return kept


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols])
    return out


def add_train_rul(df: pd.DataFrame, clip_upper: int | None = None) -> pd.DataFrame:
    """
    For train split, target RUL is max_cycle_per_engine - current_cycle.
    """
    out = df.copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    if clip_upper is not None:
        out["rul"] = out["rul"].clip(upper=clip_upper)
    return out


def train_val_engine_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # NumPy 2.4 can return read-only views from pandas arrays; copy to allow in-place shuffle.
    engine_ids = df["engine_id"].drop_duplicates().to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(engine_ids)

    n_val = max(1, int(len(engine_ids) * val_ratio))
    val_ids = set(engine_ids[:n_val].tolist())

    val_df = df[df["engine_id"].isin(val_ids)].copy()
    train_df = df[~df["engine_id"].isin(val_ids)].copy()
    return train_df, val_df


def proxy_anomaly_labels(df: pd.DataFrame, horizon_cycles: int = 20) -> np.ndarray:
    """
    Proxy labels for benchmarking anomaly detectors:
    last `horizon_cycles` cycles in each engine trajectory are labeled anomalous (1).
    """
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    labels = (df["cycle"] >= (max_cycle - horizon_cycles + 1)).astype(int)
    return labels.to_numpy()


# ── B1: Physics-informed operating condition normalisation ─────────────────────

class OperatingConditionNormaliser:
    """
    Physics-informed preprocessing: normalise each sensor reading relative to
    the healthy baseline for its operating regime.

    Encodes the physical knowledge that "normal" degradation readings are
    regime-dependent — a reading that looks anomalous under one regime may be
    perfectly healthy under another.  Addresses the LSTM AE degradation on
    multi-condition subsets FD002 and FD004.

    Usage:
        oc = OperatingConditionNormaliser(n_regimes=6)
        oc.fit(train_df, sensor_cols, healthy_mask)
        train_oc = oc.transform(train_df, sensor_cols)
        val_oc   = oc.transform(val_df,   sensor_cols)
    """

    def __init__(self, n_regimes: int = 6, seed: int = 42) -> None:
        self.n_regimes = n_regimes
        self.seed = seed
        self._kmeans: KMeans | None = None
        self._op_cols: list[str] = []
        # regime_id → {sensor: (mean, std)}
        self._stats: dict[int, dict[str, tuple[float, float]]] = {}

    def fit(
        self,
        df: pd.DataFrame,
        sensor_cols: list[str],
        healthy_mask: pd.Series,
    ) -> "OperatingConditionNormaliser":
        self._op_cols = [c for c in df.columns if c.startswith("op_setting_")]
        if not self._op_cols:
            raise ValueError("No op_setting_* columns found; cannot determine regimes.")

        self._kmeans = KMeans(
            n_clusters=self.n_regimes, random_state=self.seed, n_init=10
        )
        self._kmeans.fit(df[self._op_cols])
        regimes = self._kmeans.predict(df[self._op_cols])

        healthy_bool = np.asarray(healthy_mask, dtype=bool)
        data_arr = df[sensor_cols].to_numpy(dtype=float)

        for r in range(self.n_regimes):
            regime_mask = regimes == r
            combined = regime_mask & healthy_bool
            # Fall back to all healthy data if this regime has < 5 healthy samples
            rows = data_arr[combined] if combined.sum() >= 5 else data_arr[healthy_bool]
            self._stats[r] = {
                col: (float(rows[:, j].mean()), float(rows[:, j].std() + 1e-8))
                for j, col in enumerate(sensor_cols)
            }
        return self

    def transform(self, df: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
        if self._kmeans is None:
            raise RuntimeError("Call fit() before transform().")
        regimes = self._kmeans.predict(df[self._op_cols])
        arr = df[sensor_cols].to_numpy(dtype=float).copy()

        for r in range(self.n_regimes):
            mask = regimes == r
            if not mask.any():
                continue
            for j, col in enumerate(sensor_cols):
                mean, std = self._stats[r][col]
                arr[mask, j] = (arr[mask, j] - mean) / std

        out = df.copy()
        out[sensor_cols] = arr
        return out


# ── B3: Scalar Health Index ────────────────────────────────────────────────────

def compute_health_index(
    df: pd.DataFrame,
    feature_cols: list[str],
    healthy_mask: pd.Series,
) -> np.ndarray:
    """
    Derive a scalar Health Index (HI) via PCA on healthy-phase sensor readings.

    The HI is normalised to [0, 1] where 1.0 = healthy and 0.0 = near-failure.
    This is a standard PHM (Prognostics and Health Management) technique used
    in industrial condition monitoring.

    Returns an array of shape [len(df)] with HI per row.
    """
    healthy_bool = np.asarray(healthy_mask, dtype=bool)
    healthy_data = df[feature_cols].to_numpy(dtype=float)[healthy_bool]
    pca = PCA(n_components=1)
    pca.fit(healthy_data)

    hi_raw = pca.transform(df[feature_cols].to_numpy(dtype=float)).ravel()
    hi_min, hi_max = hi_raw.min(), hi_raw.max()
    hi = (hi_raw - hi_min) / max(hi_max - hi_min, 1e-9)

    # Flip so that 1.0 = healthy (healthy rows should have above-median HI)
    healthy_mean_hi = hi[healthy_bool].mean()
    if healthy_mean_hi < 0.5:
        hi = 1.0 - hi

    return hi
