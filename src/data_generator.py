"""Synthetic semiconductor L1/L3 data generator with known ground-truth relationships.

The goal is to produce data where window optimization can find meaningful recommendations
that match planted truth, so the prototype's behaviour can be validated end-to-end.

Planted relationships (used to verify recommendations later):
- metrology_x1: quadratic, true optimum ~= 10.0, mean 10, sigma 1
- metrology_x2: monotonic, fail rises above ~= 5.5, mean 5, sigma 0.5
- metrology_x3: noise-only (no real effect)
- vm_x1: step function, fail jumps when value < 95, mean 100, sigma 3
- vm_x2: mild monotonic (linear), small effect
- sensor_mean_1: interacts with TOOL_A only (high value -> fail)
- sensor_std_1/max_1/slope_1: weak effects, mostly noise

In current_specs.csv, current_target for metrology_x1 is deliberately offset
from the true optimum (10.5 vs 10) so the optimizer should recommend a shift.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate(out_dir: Path, seed: int = 42, n_wafers: int = 3000) -> dict[str, Path]:
    """Generate sample_l1.csv, sample_l3.csv, current_specs.csv into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    products = [f"P{i}" for i in range(1, 6)]
    tools = ["TOOL_A", "TOOL_B", "TOOL_C"]
    chambers = ["CH1", "CH2", "CH3"]
    steps = ["STEP_1", "STEP_2", "STEP_3", "STEP_4"]
    recipes = ["R1", "R2", "R3"]

    start = datetime(2026, 2, 10)
    event_times = sorted(
        start + timedelta(minutes=int(m))
        for m in rng.uniform(0, 90 * 24 * 60, size=n_wafers)
    )

    df = pd.DataFrame({
        "wafer_id": [f"W{i:06d}" for i in range(n_wafers)],
        "lot_id": [f"LOT{(i // 25):05d}" for i in range(n_wafers)],
        "product": rng.choice(products, n_wafers),
        "process_step": rng.choice(steps, n_wafers),
        "tool_id": rng.choice(tools, n_wafers),
        "chamber_id": rng.choice(chambers, n_wafers),
        "recipe_id": rng.choice(recipes, n_wafers),
        "event_time": event_times,
    })

    df["metrology_x1"] = rng.normal(10.0, 1.0, n_wafers)
    df["metrology_x2"] = rng.normal(5.0, 0.5, n_wafers)
    df["metrology_x3"] = rng.normal(20.0, 2.0, n_wafers)
    df["vm_x1"] = rng.normal(100.0, 3.0, n_wafers)
    df["vm_x2"] = rng.normal(50.0, 2.0, n_wafers)
    df["sensor_mean_1"] = rng.normal(1.0, 0.1, n_wafers)
    df["sensor_std_1"] = rng.normal(0.2, 0.05, n_wafers).clip(0.01, None)
    df["sensor_max_1"] = df["sensor_mean_1"] + rng.normal(0.5, 0.1, n_wafers)
    df["sensor_slope_1"] = rng.normal(0.0, 0.05, n_wafers)

    # small product-level offset on metrology_x1 (realistic systematic bias)
    product_offset = {"P1": 0.0, "P2": 0.1, "P3": -0.1, "P4": 0.05, "P5": -0.05}
    df["metrology_x1"] += df["product"].map(product_offset)

    # 2% NaN on sensor_slope_1 to exercise imputation
    miss_idx = rng.choice(n_wafers, size=int(0.02 * n_wafers), replace=False)
    df.loc[miss_idx, "sensor_slope_1"] = np.nan

    l1_cols = [
        "wafer_id", "lot_id", "product", "process_step", "tool_id",
        "chamber_id", "recipe_id", "event_time",
        "metrology_x1", "metrology_x2", "metrology_x3",
        "vm_x1", "vm_x2",
        "sensor_mean_1", "sensor_std_1", "sensor_max_1", "sensor_slope_1",
    ]
    l1 = df[l1_cols].copy()

    # --- L3: planted ground-truth relationships ---
    x1 = df["metrology_x1"].fillna(10.0).to_numpy()
    item1 = 0.02 + 0.04 * ((x1 - 10.0) ** 2) / 4.0  # quadratic, min at 10

    vm1 = df["vm_x1"].to_numpy()
    item2 = np.where(vm1 < 95.0, 0.10, 0.02) + 0.005 * np.maximum(0, 95 - vm1)

    x2 = df["metrology_x2"].to_numpy()
    item3 = 0.02 + 0.08 * np.maximum(0, x2 - 5.5)

    tool_a_mask = (df["tool_id"] == "TOOL_A").to_numpy()
    sensor_effect = np.where(
        tool_a_mask,
        0.05 * np.maximum(0, df["sensor_mean_1"].to_numpy() - 1.0),
        0.0,
    )

    noise = rng.normal(0.0, 0.01, n_wafers)

    item1 = np.clip(item1 + 0.5 * noise, 0.0, 1.0)
    item2 = np.clip(item2 + 0.5 * noise, 0.0, 1.0)
    item3 = np.clip(item3 + 0.5 * noise, 0.0, 1.0)

    total_fail = np.clip(item1 + item2 + item3 + sensor_effect + noise, 0.0, 1.0)
    yield_ = 1.0 - total_fail

    l3 = pd.DataFrame({
        "wafer_id": df["wafer_id"],
        "eds_item_001_fail_rate": item1,
        "eds_item_002_fail_rate": item2,
        "eds_item_003_fail_rate": item3,
        "total_fail_rate": total_fail,
        "yield": yield_,
    })

    # current SPECs: deliberately suboptimal to exercise the optimizer
    specs = pd.DataFrame([
        {"feature_name": "metrology_x1",   "current_lsl":  8.0, "current_target": 10.5, "current_usl": 12.0, "adjustable_flag": 1},
        {"feature_name": "metrology_x2",   "current_lsl":  4.0, "current_target":  5.0, "current_usl":  6.5, "adjustable_flag": 1},
        {"feature_name": "metrology_x3",   "current_lsl": 15.0, "current_target": 20.0, "current_usl": 25.0, "adjustable_flag": 1},
        {"feature_name": "vm_x1",          "current_lsl": 90.0, "current_target":100.0, "current_usl":110.0, "adjustable_flag": 1},
        {"feature_name": "vm_x2",          "current_lsl": 45.0, "current_target": 50.0, "current_usl": 55.0, "adjustable_flag": 1},
        {"feature_name": "sensor_mean_1",  "current_lsl":  0.8, "current_target":  1.0, "current_usl":  1.2, "adjustable_flag": 1},
        {"feature_name": "sensor_std_1",   "current_lsl": 0.05, "current_target":  0.2, "current_usl": 0.35, "adjustable_flag": 1},
        {"feature_name": "sensor_max_1",   "current_lsl":  1.0, "current_target":  1.5, "current_usl":  2.0, "adjustable_flag": 1},
        {"feature_name": "sensor_slope_1", "current_lsl":-0.15, "current_target":  0.0, "current_usl": 0.15, "adjustable_flag": 1},
    ])

    p_l1 = out_dir / "sample_l1.csv"
    p_l3 = out_dir / "sample_l3.csv"
    p_sp = out_dir / "current_specs.csv"
    l1.to_csv(p_l1, index=False)
    l3.to_csv(p_l3, index=False)
    specs.to_csv(p_sp, index=False)
    return {"l1": p_l1, "l3": p_l3, "specs": p_sp}


if __name__ == "__main__":
    paths = generate(Path("data"))
    for k, v in paths.items():
        print(f"{k}: {v}")
