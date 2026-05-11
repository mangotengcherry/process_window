"""Column type registry and helpers (split, impute, time-based split)."""
from __future__ import annotations

import pandas as pd

NUMERIC_PREFIXES = ("metrology_", "vm_", "sensor_")
CATEGORICAL_COLS = ["product", "process_step", "tool_id", "chamber_id", "recipe_id"]
ID_COLS = ["wafer_id", "lot_id", "event_time"]
L3_TARGETS = [
    "eds_item_001_fail_rate",
    "eds_item_002_fail_rate",
    "eds_item_003_fail_rate",
    "total_fail_rate",
    "yield",
]


def split_columns(df: pd.DataFrame) -> dict:
    """Return dict with numeric_features, categorical_features, id_cols, targets."""
    numeric = [c for c in df.columns
               if any(c.startswith(p) for p in NUMERIC_PREFIXES)]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    ids = [c for c in ID_COLS if c in df.columns]
    targets = [c for c in L3_TARGETS if c in df.columns]
    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "id_cols": ids,
        "targets": targets,
    }


def basic_impute(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    """Median-impute numeric columns on a copy (does not mutate input)."""
    out = df.copy()
    for c in numeric_features:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    return out


def time_split(df: pd.DataFrame, time_col: str = "event_time",
               test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Most recent test_frac of rows go to test set (out-of-time split)."""
    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_frac))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()
