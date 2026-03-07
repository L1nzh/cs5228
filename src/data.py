from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    raw: pd.DataFrame


def load_csv(path: str | Path, target_col: str = "Churn") -> Dataset:
    p = Path(path)
    df = pd.read_csv(p)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {p}")

    y_raw = df[target_col].astype(str).str.upper().str.strip()
    y = y_raw.map({"TRUE": 1, "FALSE": 0})
    if y.isna().any():
        bad = sorted(set(y_raw[y.isna()].unique().tolist()))
        raise ValueError(f"Unexpected target values in {p}: {bad}")

    X = df.drop(columns=[target_col])
    return Dataset(X=X, y=y.astype(int), raw=df)


def assert_same_schema(train_X: pd.DataFrame, test_X: pd.DataFrame) -> None:
    train_cols = list(train_X.columns)
    test_cols = list(test_X.columns)
    if train_cols != test_cols:
        missing_in_test = [c for c in train_cols if c not in test_cols]
        extra_in_test = [c for c in test_cols if c not in train_cols]
        raise ValueError(
            "Train/test schema mismatch. "
            f"missing_in_test={missing_in_test}, extra_in_test={extra_in_test}"
        )

