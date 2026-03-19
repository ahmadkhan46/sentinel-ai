from __future__ import annotations

from pathlib import Path

import pandas as pd


CMAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def read_cmapss_split(data_dir: Path, subset: str, split: str) -> pd.DataFrame:
    """
    Read C-MAPSS train/test split text files.
    Expected filenames: train_FD001.txt, test_FD001.txt, etc.
    """
    filename = f"{split}_{subset}.txt"
    file_path = data_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    # Raw files can contain trailing empty columns depending on source mirror.
    df = df.dropna(axis=1, how="all")
    df.columns = CMAPSS_COLUMNS[: len(df.columns)]
    return df


def read_cmapss_rul_truth(data_dir: Path, subset: str) -> pd.DataFrame:
    """
    Read RUL truth file for C-MAPSS test set.
    Expected filename: RUL_FD001.txt, etc.
    """
    file_path = data_dir / f"RUL_{subset}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing RUL truth file: {file_path}")

    rul = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    rul = rul.dropna(axis=1, how="all")
    if rul.shape[1] != 1:
        raise ValueError(f"Unexpected RUL truth shape for {file_path}: {rul.shape}")

    rul.columns = ["rul"]
    rul["engine_id"] = rul.index + 1
    return rul[["engine_id", "rul"]]


def load_cmapss_bundle(data_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = read_cmapss_split(data_dir=data_dir, subset=subset, split="train")
    test_df = read_cmapss_split(data_dir=data_dir, subset=subset, split="test")
    rul_df = read_cmapss_rul_truth(data_dir=data_dir, subset=subset)
    return train_df, test_df, rul_df
