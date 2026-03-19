from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences_with_metadata(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    target_col: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build fixed-length sequences per engine with sequence metadata.
    Returns:
    - X: [num_sequences, sequence_length, num_features]
    - meta_df: engine_id/start_cycle/end_cycle/max_cycle (+ target if provided)
    """
    sequences = []
    metadata: list[dict[str, float | int]] = []

    for engine_id, group in df.groupby("engine_id"):
        ordered = group.sort_values("cycle")
        arr = ordered[feature_cols].to_numpy()
        cycles = ordered["cycle"].to_numpy()
        max_cycle = int(cycles.max())
        target_vals = ordered[target_col].to_numpy() if target_col and target_col in ordered.columns else None

        if len(arr) < sequence_length:
            continue

        for i in range(len(arr) - sequence_length + 1):
            end_idx = i + sequence_length - 1
            sequences.append(arr[i : i + sequence_length])

            row: dict[str, float | int] = {
                "engine_id": int(engine_id),
                "start_cycle": int(cycles[i]),
                "end_cycle": int(cycles[end_idx]),
                "max_cycle": max_cycle,
            }
            if target_vals is not None:
                row["target"] = float(target_vals[end_idx])
            metadata.append(row)

    if not sequences:
        empty_seq = np.empty((0, sequence_length, len(feature_cols)))
        return empty_seq, pd.DataFrame(columns=["engine_id", "start_cycle", "end_cycle", "max_cycle"])

    return np.stack(sequences), pd.DataFrame(metadata)


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build fixed-length sequences per engine.
    Returns:
    - X: [num_sequences, sequence_length, num_features]
    - meta_engine_ids: [num_sequences]
    """
    x, meta = build_sequences_with_metadata(
        df=df,
        feature_cols=feature_cols,
        sequence_length=sequence_length,
    )
    return x, meta["engine_id"].to_numpy() if not meta.empty else np.empty((0,))
