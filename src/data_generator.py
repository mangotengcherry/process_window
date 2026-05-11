"""Synthetic semiconductor L1/L3 data generator with known ground-truth relationships.

Designed so each of the four typical fab "window shapes" (in fail-rate space) is
represented and the recommender's output can be validated by visual inspection:

  Shape           | Features                          | Expected recommendation
  ----------------|-----------------------------------|--------------------------
  양쪽 들림 (U)   | metrology_x1, vm_x1 (asymmetric)  | target 유지, window 좁힘
  오른쪽 들림     | metrology_x2, sensor_mean_1(*),    | USL 좁힘
                  | sensor_std_1                      |
  왼쪽 들림       | metrology_x3                       | LSL 끌어올림
  flat (noise)    | vm_x2, sensor_max_1, sensor_slope_1| 추천 없음 / Confidence C

  (*) sensor_mean_1 has additional TOOL_A interaction so segment bias penalty fires.

In current_specs.csv each adjustable SPEC is deliberately suboptimal so the
optimizer has something to recommend.
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

    # numeric L1 (units intentionally varied to look fab-like)
    df["metrology_x1"] = rng.normal(10.0, 1.0, n_wafers)   # 양쪽 들림 (U)
    df["metrology_x2"] = rng.normal(5.0, 0.5, n_wafers)    # 오른쪽 들림
    df["metrology_x3"] = rng.normal(20.0, 2.0, n_wafers)   # 왼쪽 들림
    df["vm_x1"] = rng.normal(100.0, 3.0, n_wafers)         # 양쪽 들림 (asymmetric)
    df["vm_x2"] = rng.normal(50.0, 2.0, n_wafers)          # flat (noise)
    df["sensor_mean_1"] = rng.normal(1.0, 0.1, n_wafers)   # 오른쪽 들림 + TOOL_A 교호작용
    df["sensor_std_1"] = rng.normal(0.2, 0.05, n_wafers).clip(0.01, None)  # 오른쪽 들림 mild
    df["sensor_max_1"] = df["sensor_mean_1"] + rng.normal(0.5, 0.1, n_wafers)  # flat
    df["sensor_slope_1"] = rng.normal(0.0, 0.05, n_wafers) # flat

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

    # --- L3: planted ground-truth fail-rate relationships ---
    BASE = 0.015  # baseline per-item fail rate when feature is in safe zone

    x1 = df["metrology_x1"].fillna(10.0).to_numpy()
    x2 = df["metrology_x2"].to_numpy()
    x3 = df["metrology_x3"].to_numpy()
    vm1 = df["vm_x1"].to_numpy()
    sm1 = df["sensor_mean_1"].to_numpy()
    sst = df["sensor_std_1"].to_numpy()

    # 양쪽 들림 (U): metrology_x1 (optimum 10) + vm_x1 (sweet zone 95~107, asymmetric)
    x1_pen = 0.04 * ((x1 - 10.0) / 1.0) ** 2                       # symmetric U
    vm1_left = 0.006 * np.maximum(0, 95 - vm1) ** 1.5              # left arm steeper
    vm1_right = 0.003 * np.maximum(0, vm1 - 107) ** 1.2            # right arm milder
    item1 = BASE + x1_pen + vm1_left + vm1_right

    # 오른쪽 들림: metrology_x2 (USL 직전부터 fail↑) + sensor_std_1 (산포 클수록 fail↑)
    x2_right = 0.10 * np.maximum(0, x2 - 5.5)
    sst_right = 0.20 * np.maximum(0, sst - 0.22)                   # mild
    item2 = BASE + x2_right + sst_right

    # 왼쪽 들림: metrology_x3 (LSL 부근에서 fail↑)
    x3_left = 0.05 * np.maximum(0, 18.0 - x3)
    item3 = BASE + x3_left

    # tool 교호작용: sensor_mean_1 > 1.0 이면서 TOOL_A 일 때만 fail↑ (segment bias 데모용)
    tool_a_mask = (df["tool_id"] == "TOOL_A").to_numpy()
    sensor_interaction = np.where(
        tool_a_mask,
        0.06 * np.maximum(0, sm1 - 1.0),
        0.0,
    )

    noise = rng.normal(0.0, 0.01, n_wafers)

    item1 = np.clip(item1 + 0.5 * noise, 0.0, 1.0)
    item2 = np.clip(item2 + 0.5 * noise, 0.0, 1.0)
    item3 = np.clip(item3 + 0.5 * noise, 0.0, 1.0)

    total_fail = np.clip(item1 + item2 + item3 + sensor_interaction + noise, 0.0, 1.0)
    yield_ = 1.0 - total_fail

    l3 = pd.DataFrame({
        "wafer_id": df["wafer_id"],
        "eds_item_001_fail_rate": item1,
        "eds_item_002_fail_rate": item2,
        "eds_item_003_fail_rate": item3,
        "total_fail_rate": total_fail,
        "yield": yield_,
    })

    # current SPECs: each shape gets a deliberately suboptimal setting.
    #   - 양쪽 들림 → window 가 너무 넓음 (양쪽 좁히는 것이 정답)
    #   - 오른쪽 들림 → USL 이 fail 영역까지 허용 (USL 좁히기)
    #   - 왼쪽 들림 → LSL 이 fail 영역까지 허용 (LSL 끌어올리기)
    #   - flat       → SPEC 무엇이든 yield 변화 없음 (Confidence C 기대)
    specs = pd.DataFrame([
        # 양쪽 들림
        {"feature_name": "metrology_x1",   "current_lsl":  8.0, "current_target": 10.0, "current_usl": 12.0, "adjustable_flag": 1},
        {"feature_name": "vm_x1",          "current_lsl": 90.0, "current_target":100.0, "current_usl":110.0, "adjustable_flag": 1},
        # 오른쪽 들림
        {"feature_name": "metrology_x2",   "current_lsl":  4.0, "current_target":  5.0, "current_usl":  6.5, "adjustable_flag": 1},
        {"feature_name": "sensor_std_1",   "current_lsl": 0.05, "current_target":  0.2, "current_usl": 0.35, "adjustable_flag": 1},
        {"feature_name": "sensor_mean_1",  "current_lsl":  0.8, "current_target":  1.0, "current_usl":  1.2, "adjustable_flag": 1},
        # 왼쪽 들림
        {"feature_name": "metrology_x3",   "current_lsl": 15.0, "current_target": 20.0, "current_usl": 25.0, "adjustable_flag": 1},
        # flat (noise) — 추천이 거의 무의미 해야 함 (Confidence C 기대)
        {"feature_name": "vm_x2",          "current_lsl": 45.0, "current_target": 50.0, "current_usl": 55.0, "adjustable_flag": 1},
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
