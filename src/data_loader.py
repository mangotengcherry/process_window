"""Data loading and L1+L3 mart join.

Leakage note: L3 columns (eds_item_xxx_fail_rate, total_fail_rate, yield) are
PREDICTION TARGETS. They must never be used as model features. This module only
joins on wafer_id; feature_engineering.split_columns separates targets from
features so downstream training/PDP only sees L1 columns.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample_l1.csv and sample_l3.csv, parse event_time."""
    data_dir = Path(data_dir)
    l1 = pd.read_csv(data_dir / "sample_l1.csv", parse_dates=["event_time"])
    l3 = pd.read_csv(data_dir / "sample_l3.csv")
    return l1, l3


def load_specs(data_dir: Path) -> pd.DataFrame:
    """Load current_specs.csv."""
    return pd.read_csv(Path(data_dir) / "current_specs.csv")


def build_mart(l1: pd.DataFrame, l3: pd.DataFrame) -> pd.DataFrame:
    """Inner-join L1 with L3 on wafer_id. Warn if rows are dropped."""
    before = len(l1)
    mart = l1.merge(l3, on="wafer_id", how="inner")
    after = len(mart)
    if after < before:
        warnings.warn(
            f"build_mart: {before - after} L1 wafers dropped (no matching L3). "
            f"{before} -> {after}"
        )
    return mart
