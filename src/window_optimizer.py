"""Per-feature window/target recommendation based on model + actual data.

Algorithm per adjustable feature `f`:
  1. Current window stats: yield mean inside [LSL, USL], coverage fraction.
  2. Build response curve via model sweep (other features fixed at median/mode).
     Also compute actual binned yield curve from data.
  3. Candidate target = argmax of model response curve.
  4. Candidate [LSL, USL] = widest contiguous range around target where model
     yield >= curve_peak - margin, subject to coverage >= min_coverage.
  5. Score = yield uplift - penalties + item-improvement bonus.
  6. Confidence grade A/B/C from model R^2 and model-vs-bin agreement.
  7. Human-readable reason summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# ---------- helpers ----------

def _response_curve(model: Pipeline, df: pd.DataFrame, feature: str,
                    numeric_features: list[str],
                    categorical_features: list[str],
                    grid_size: int = 50) -> pd.DataFrame:
    """Sweep `feature` across percentile 1..99, holding others at median/mode."""
    s = df[feature].dropna()
    lo, hi = float(np.percentile(s, 1)), float(np.percentile(s, 99))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    grid = np.linspace(lo, hi, grid_size)

    template = {}
    for c in numeric_features:
        template[c] = float(df[c].median())
    for c in categorical_features:
        template[c] = df[c].mode().iloc[0]

    sweep = pd.DataFrame([{**template, feature: v} for v in grid])
    pred = model.predict(sweep[numeric_features + categorical_features])
    return pd.DataFrame({"value": grid, "pred_yield": pred})


def _binned_yield(df: pd.DataFrame, feature: str, target_col: str,
                  n_bins: int = 15) -> pd.DataFrame:
    """Actual mean yield per equal-frequency bin of feature."""
    s = df[[feature, target_col]].dropna()
    if s.empty:
        return pd.DataFrame(columns=["bin_center", "mean_yield", "count"])
    s = s.copy()
    try:
        s["bin"] = pd.qcut(s[feature], q=n_bins, duplicates="drop")
    except ValueError:
        s["bin"] = pd.cut(s[feature], bins=n_bins)
    agg = s.groupby("bin", observed=True).agg(
        mean_yield=(target_col, "mean"),
        count=(target_col, "size"),
        bin_center=(feature, "mean"),
    ).reset_index(drop=True)
    return agg


def _find_window(curve: pd.DataFrame, target_value: float,
                 min_coverage_frac: float,
                 feature_values: pd.Series,
                 margin: float = 0.02) -> tuple[float, float, float]:
    """Widest contiguous range around target where curve.pred_yield >= peak - margin.

    Then widen symmetrically if coverage falls below min_coverage_frac.
    """
    curve = curve.sort_values("value").reset_index(drop=True)
    if curve.empty:
        v_min, v_max = float(feature_values.min()), float(feature_values.max())
        return v_min, v_max, 1.0

    threshold = float(curve["pred_yield"].max()) - margin
    ok = curve["pred_yield"] >= threshold

    if not ok.any():
        sigma = float(feature_values.std()) or 1.0
        lsl, usl = target_value - sigma, target_value + sigma
    else:
        idx_target = int((curve["value"] - target_value).abs().idxmin())
        if not bool(ok.iloc[idx_target]):
            lsl = float(curve.loc[ok, "value"].min())
            usl = float(curve.loc[ok, "value"].max())
        else:
            i = idx_target
            while i > 0 and bool(ok.iloc[i - 1]):
                i -= 1
            lsl = float(curve.iloc[i]["value"])
            j = idx_target
            while j < len(curve) - 1 and bool(ok.iloc[j + 1]):
                j += 1
            usl = float(curve.iloc[j]["value"])

    coverage = float(((feature_values >= lsl) & (feature_values <= usl)).mean())

    if coverage < min_coverage_frac:
        sigma = float(feature_values.std()) or 1.0
        step = sigma * 0.1
        attempts = 0
        while coverage < min_coverage_frac and attempts < 100:
            lsl -= step
            usl += step
            coverage = float(((feature_values >= lsl) & (feature_values <= usl)).mean())
            attempts += 1

    return lsl, usl, coverage


def _segment_bias(df: pd.DataFrame, feature: str, target_col: str,
                  recommended_target: float, current_target: float,
                  groupby: str) -> bool:
    """True if any segment's optimum disagrees with overall direction (>=2 segments)."""
    if groupby not in df.columns:
        return False
    overall_dir = np.sign(recommended_target - current_target)
    if overall_dir == 0:
        return False
    flipped = 0
    for _, sub in df.groupby(groupby):
        if len(sub) < 30:
            continue
        bins = _binned_yield(sub, feature, target_col, n_bins=10)
        if bins.empty:
            continue
        best_v = float(bins.loc[bins["mean_yield"].idxmax(), "bin_center"])
        sub_dir = np.sign(best_v - current_target)
        if sub_dir != 0 and sub_dir != overall_dir:
            flipped += 1
    return flipped >= 2


def _confidence(model_r2: float, model_target: float, bin_target: float,
                feature_range: float) -> str:
    diff_frac = abs(model_target - bin_target) / max(feature_range, 1e-9)
    if model_r2 > 0.5 and diff_frac < 0.05:
        return "A"
    if model_r2 > 0.3 and diff_frac < 0.15:
        return "B"
    return "C"


def _reason(current_target: float, recommended_target: float,
            current_window_yield: float, recommended_window_yield: float,
            coverage_current: float, coverage_recommended: float,
            confidence: str, has_bias: bool,
            item_contributions: dict[str, float]) -> str:
    target_shift = recommended_target - current_target
    pp_cov = (coverage_current - coverage_recommended) * 100
    if abs(target_shift) < 1e-6:
        direction = "유지"
    elif target_shift > 0:
        direction = f"+{target_shift:.3f} 이동 권장"
    else:
        direction = f"{target_shift:.3f} 이동 권장"

    yield_delta = recommended_window_yield - current_window_yield
    yield_msg = (f"추천 window 내 평균 yield {recommended_window_yield:.3f} "
                 f"(현재 {current_window_yield:.3f}, "
                 f"{'+' if yield_delta >= 0 else ''}{yield_delta:.3f})")

    cov_sign = "-" if pp_cov >= 0 else "+"
    cov_msg = (f"coverage {coverage_current:.2f}→{coverage_recommended:.2f} "
               f"({cov_sign}{abs(pp_cov):.1f}pp)")

    bias_msg = "tool/product 추천 방향 편향 있음 (검토 필요)" if has_bias else "tool/product 편향 없음"

    if item_contributions:
        top_item = max(item_contributions, key=item_contributions.get)
        delta = item_contributions[top_item]
        item_msg = f"{top_item} fail rate {delta * 100:.1f}pp 개선 기여 추정"
    else:
        item_msg = "특정 EDS item 개선 기여 미확인"

    return (f"현재 target={current_target:.3f}, 모델 기반 optimum={recommended_target:.3f}, "
            f"{direction}. {yield_msg}. {cov_msg}. {bias_msg}. {item_msg}. "
            f"Confidence {confidence}.")


# ---------- main entry ----------

EDS_ITEM_COLS = (
    "eds_item_001_fail_rate",
    "eds_item_002_fail_rate",
    "eds_item_003_fail_rate",
)


def recommend(model: Pipeline,
              df: pd.DataFrame,
              specs: pd.DataFrame,
              numeric_features: list[str],
              categorical_features: list[str],
              target_col: str = "yield",
              model_r2: float = 0.0,
              min_coverage: float = 0.6,
              top_n: int | None = None) -> pd.DataFrame:
    """One recommendation row per adjustable spec feature; sorted by score."""
    rows = []

    for _, spec in specs.iterrows():
        f = spec["feature_name"]
        if not int(spec.get("adjustable_flag", 0)):
            continue
        if f not in df.columns:
            continue

        c_lsl = float(spec["current_lsl"])
        c_target = float(spec["current_target"])
        c_usl = float(spec["current_usl"])

        mask_in = (df[f] >= c_lsl) & (df[f] <= c_usl)
        coverage_current = float(mask_in.mean())
        current_window_yield = (
            float(df.loc[mask_in, target_col].mean()) if mask_in.any() else float("nan")
        )

        curve = _response_curve(model, df, f, numeric_features, categorical_features)
        bins = _binned_yield(df, f, target_col)

        model_target = float(curve.loc[curve["pred_yield"].idxmax(), "value"])
        bin_target = (
            float(bins.loc[bins["mean_yield"].idxmax(), "bin_center"])
            if not bins.empty else model_target
        )

        feature_range = float(df[f].max() - df[f].min())
        recommended_target = model_target

        rec_lsl, rec_usl, coverage_rec = _find_window(
            curve, recommended_target, min_coverage, df[f].dropna()
        )

        rec_mask = (df[f] >= rec_lsl) & (df[f] <= rec_usl)
        recommended_window_yield = (
            float(df.loc[rec_mask, target_col].mean()) if rec_mask.any() else float("nan")
        )

        # per-item contribution from actual data (no extra model needed)
        item_contrib: dict[str, float] = {}
        for item_col in EDS_ITEM_COLS:
            if item_col not in df.columns:
                continue
            cur_fail = (
                float(df.loc[mask_in, item_col].mean()) if mask_in.any() else float("nan")
            )
            rec_fail = (
                float(df.loc[rec_mask, item_col].mean()) if rec_mask.any() else float("nan")
            )
            if np.isnan(cur_fail) or np.isnan(rec_fail):
                continue
            delta = cur_fail - rec_fail
            if delta > 0.005:  # >=0.5pp fail rate reduction
                item_contrib[item_col] = delta

        expected_yield_uplift = (
            recommended_window_yield - current_window_yield
            if pd.notna(current_window_yield) and pd.notna(recommended_window_yield)
            else 0.0
        )
        coverage_loss = max(0.0, coverage_current - coverage_rec)

        current_width = c_usl - c_lsl
        rec_width = rec_usl - rec_lsl
        instability_penalty = (
            0.5 if abs(model_target - bin_target) > 0.15 * feature_range else 0.0
        )
        if current_width > 0 and rec_width < 0.5 * current_width:
            instability_penalty += 0.2

        has_bias = (
            _segment_bias(df, f, target_col, recommended_target, c_target, "tool_id")
            or _segment_bias(df, f, target_col, recommended_target, c_target, "product")
        )
        bias_penalty = 0.3 if has_bias else 0.0
        major_item_bonus = 0.2 if item_contrib else 0.0

        score = (expected_yield_uplift
                 - 0.5 * coverage_loss
                 - instability_penalty
                 - bias_penalty
                 + major_item_bonus)

        confidence = _confidence(model_r2, model_target, bin_target, feature_range)
        reason = _reason(c_target, recommended_target,
                         current_window_yield, recommended_window_yield,
                         coverage_current, coverage_rec,
                         confidence, has_bias, item_contrib)

        rows.append({
            "feature_name": f,
            "current_lsl": c_lsl,
            "current_target": c_target,
            "current_usl": c_usl,
            "recommended_lsl": rec_lsl,
            "recommended_target": recommended_target,
            "recommended_usl": rec_usl,
            "target_shift": recommended_target - c_target,
            "expected_yield_uplift": expected_yield_uplift,
            "current_window_yield": current_window_yield,
            "recommended_window_yield": recommended_window_yield,
            "coverage_current": coverage_current,
            "coverage_recommended": coverage_rec,
            "coverage_loss": coverage_loss,
            "recommendation_score": score,
            "confidence_grade": confidence,
            "reason_summary": reason,
        })

    out = (pd.DataFrame(rows)
           .sort_values("recommendation_score", ascending=False)
           .reset_index(drop=True))
    if top_n:
        out = out.head(top_n)
    return out
