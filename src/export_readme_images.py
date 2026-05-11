"""Generate static PNG images of the key charts for embedding into README.md.

Run: `python -m src.export_readme_images`
Writes to: docs/images/*.png

We use matplotlib (already a transitive dep of sklearn) instead of plotly+kaleido
to avoid the optional kaleido install on Windows.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def _use_korean_font():
    """Pick first available Korean-capable font on Windows/macOS/Linux."""
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            plt.rcParams["axes.unicode_minus"] = False
            return c
    return None


_use_korean_font()

from .data_generator import generate
from .data_loader import build_mart, load_data, load_specs
from .feature_engineering import basic_impute, split_columns
from .modeling import train_and_score
from .window_optimizer import recommend

DATA_DIR = Path("data")
IMG_DIR = Path("docs/images")


def _ensure_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "sample_l1.csv").exists():
        generate(DATA_DIR)


def _binned(df: pd.DataFrame, feature: str, target: str, n_bins: int = 20) -> pd.DataFrame:
    s = df[[feature, target]].dropna().copy()
    try:
        s["bin"] = pd.qcut(s[feature], q=n_bins, duplicates="drop")
    except ValueError:
        s["bin"] = pd.cut(s[feature], bins=n_bins)
    return (s.groupby("bin", observed=True)
              .agg(mean_y=(target, "mean"), bin_center=(feature, "mean"))
              .reset_index(drop=True))


def fig_ground_truth(mart: pd.DataFrame, out: Path) -> Path:
    feats = ["metrology_x1", "vm_x1", "metrology_x2", "metrology_x3"]
    titles = [
        "metrology_x1 (역U, true optimum ≈ 10)",
        "vm_x1 (step, 95 미만 fail jump)",
        "metrology_x2 (monotonic, 5.5 초과 fail↑)",
        "metrology_x3 (noise, 관계 없음)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, f, t in zip(axes.ravel(), feats, titles):
        ax.scatter(mart[f], mart["yield"], s=4, alpha=0.15, color="#1f77b4")
        agg = _binned(mart, f, "yield")
        ax.plot(agg["bin_center"], agg["mean_y"], color="red", lw=2, marker="o", ms=4)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel(f)
        ax.set_ylabel("yield")
        ax.grid(alpha=0.3)
    fig.suptitle("핵심 L1 feature ↔ yield 관계 (planted ground truth)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_predicted_vs_actual(y_true, y_pred, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.25, color="#1f77b4")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="red", ls="--", label="y = x")
    ax.set_xlabel("Actual yield")
    ax.set_ylabel("Predicted yield")
    ax.set_title("Predicted vs Actual (model accuracy)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_recommendation_impact(recs: pd.DataFrame, out: Path) -> Path:
    color_map = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for grade in ["A", "B", "C"]:
        sub = recs[recs["confidence_grade"] == grade]
        if sub.empty:
            continue
        sizes = 100 + 600 * sub["recommendation_score"].abs().clip(lower=0.05)
        ax.scatter(sub["coverage_loss"], sub["expected_yield_uplift"],
                   s=sizes, alpha=0.55, color=color_map[grade], label=f"Confidence {grade}",
                   edgecolors="black", linewidths=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r["feature_name"],
                        (r["coverage_loss"], r["expected_yield_uplift"]),
                        fontsize=8, ha="center", va="center")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("coverage_loss  (낮을수록 좋음)")
    ax.set_ylabel("expected_yield_uplift  (높을수록 좋음)")
    ax.set_title("Recommendation Impact — 좌상단일수록 가치 있는 추천")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_window_comparison(recs: pd.DataFrame, out: Path) -> Path:
    feats = recs["feature_name"].tolist()
    n = len(feats)
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.55)))
    yticks = []
    ylabs = []
    for i, (_, r) in enumerate(recs.iterrows()):
        # normalise to feature's own range for visual comparison
        rng = max(r["current_usl"] - r["current_lsl"], 1e-9)
        # plot current (gray) above, recommended (green) below the same feature label
        y_cur = i * 2 + 0.4
        y_rec = i * 2 - 0.4
        ax.hlines(y_cur, r["current_lsl"], r["current_usl"], colors="gray", lw=6)
        ax.scatter([r["current_target"]], [y_cur], marker="x", color="black", s=80, zorder=5)
        ax.hlines(y_rec, r["recommended_lsl"], r["recommended_usl"],
                  colors="#2ca02c", lw=6)
        ax.scatter([r["recommended_target"]], [y_rec], marker="x", color="black", s=80, zorder=5)
        yticks.extend([y_rec, y_cur])
        ylabs.extend([f"{r['feature_name']} (rec)", f"{r['feature_name']} (cur)"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs, fontsize=9)
    ax.set_xlabel("feature value")
    ax.set_title("Current (gray) vs Recommended (green) Window — × = target")
    ax.grid(axis="x", alpha=0.3)
    # decoration legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="gray", lw=6, label="Current LSL..USL"),
        Line2D([0], [0], color="#2ca02c", lw=6, label="Recommended LSL..USL"),
        Line2D([0], [0], marker="x", color="black", lw=0, markersize=8, label="Target"),
    ], loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_score_breakdown(recs: pd.DataFrame, out: Path) -> Path:
    df = recs.copy()
    uplift = df["expected_yield_uplift"].to_numpy()
    cov_pen = -0.5 * df["coverage_loss"].to_numpy()
    bias_pen = np.array([-0.3 if "편향 있음" in str(s) else 0.0
                         for s in df["reason_summary"]])
    bonus = np.array([0.2 if "개선 기여 추정" in str(s) else 0.0
                      for s in df["reason_summary"]])
    inst_pen = df["recommendation_score"].to_numpy() - uplift - cov_pen - bias_pen - bonus

    x = np.arange(len(df))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    # stack positives above 0, negatives below 0
    ax.bar(x, uplift, width, label="Yield uplift", color="#2ca02c")
    ax.bar(x, bonus, width, bottom=uplift, label="Item-improvement bonus", color="#17becf")
    ax.bar(x, cov_pen, width, label="Coverage penalty", color="#ff7f0e")
    ax.bar(x, inst_pen, width, bottom=cov_pen, label="Instability/width penalty", color="#d62728")
    ax.bar(x, bias_pen, width, bottom=cov_pen + inst_pen, label="Bias penalty", color="#9467bd")
    ax.plot(x, df["recommendation_score"], marker="D", color="black",
            ls="--", lw=1, ms=8, label="Final score")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_name"], rotation=30, ha="right")
    ax.set_ylabel("score component")
    ax.set_title("Recommendation Score Breakdown")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_l1_distributions(mart: pd.DataFrame, num_features, out: Path) -> Path:
    feats = [c for c in num_features if c in mart.columns]
    n = len(feats)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(mart[f].dropna(), bins=40, color="#1f77b4", alpha=0.85)
        ax.set_title(f, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)
    fig.suptitle("L1 numeric feature 분포 (3,000 wafer)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_data()

    l1, l3 = load_data(DATA_DIR)
    mart = build_mart(l1, l3)
    cols_info = split_columns(mart)
    mart = basic_impute(mart, cols_info["numeric_features"])

    model, metrics, _, test_df = train_and_score(
        mart, "yield",
        cols_info["numeric_features"], cols_info["categorical_features"],
        split="random",
    )
    specs = load_specs(DATA_DIR)
    recs = recommend(
        model, mart, specs,
        numeric_features=cols_info["numeric_features"],
        categorical_features=cols_info["categorical_features"],
        target_col="yield", model_r2=metrics["R2"],
    )

    y_true = test_df["yield"].to_numpy()
    y_pred = model.predict(test_df[cols_info["numeric_features"]
                                   + cols_info["categorical_features"]])

    paths = {
        "ground_truth": fig_ground_truth(mart, IMG_DIR / "ground_truth.png"),
        "l1_dist": fig_l1_distributions(mart, cols_info["numeric_features"],
                                        IMG_DIR / "l1_distributions.png"),
        "pred_vs_actual": fig_predicted_vs_actual(y_true, y_pred,
                                                  IMG_DIR / "predicted_vs_actual.png"),
        "impact": fig_recommendation_impact(recs, IMG_DIR / "recommendation_impact.png"),
        "score_breakdown": fig_score_breakdown(recs, IMG_DIR / "score_breakdown.png"),
        "window_comp": fig_window_comparison(recs, IMG_DIR / "window_comparison.png"),
    }
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
