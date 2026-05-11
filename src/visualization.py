"""Plotly chart factories for the Streamlit UI and evaluation reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _spec_lines(fig: go.Figure, spec: dict | None, color: str, label_prefix: str) -> None:
    if spec is None:
        return
    for key, dash, name in [("lsl", "dash", "LSL"),
                            ("target", "solid", "Target"),
                            ("usl", "dash", "USL")]:
        v = spec.get(key)
        if v is None or pd.isna(v):
            continue
        fig.add_vline(x=v, line_dash=dash, line_color=color,
                      annotation_text=f"{label_prefix} {name}",
                      annotation_position="top")


def plot_feature_vs_yield(df: pd.DataFrame, feature: str, target_col: str = "yield",
                          current_spec: dict | None = None,
                          recommended_spec: dict | None = None,
                          n_bins: int = 20) -> go.Figure:
    s = df[[feature, target_col]].dropna()
    fig = px.scatter(s, x=feature, y=target_col, opacity=0.25,
                     title=f"{feature} vs {target_col}")
    try:
        s = s.copy()
        s["bin"] = pd.qcut(s[feature], q=n_bins, duplicates="drop")
        agg = s.groupby("bin", observed=True).agg(
            mean_yield=(target_col, "mean"),
            bin_center=(feature, "mean"),
        ).reset_index(drop=True)
        fig.add_trace(go.Scatter(x=agg["bin_center"], y=agg["mean_yield"],
                                 mode="lines+markers", name="bin mean",
                                 line=dict(color="red")))
    except ValueError:
        pass
    _spec_lines(fig, current_spec, "gray", "Current")
    _spec_lines(fig, recommended_spec, "green", "Rec")
    return fig


def plot_response_curve(curve_df: pd.DataFrame,
                        current_spec: dict | None = None,
                        recommended_spec: dict | None = None,
                        feature_name: str = "") -> go.Figure:
    fig = px.line(curve_df, x="value", y="pred_yield",
                  title=f"Response Curve: {feature_name}")
    _spec_lines(fig, current_spec, "gray", "Current")
    _spec_lines(fig, recommended_spec, "green", "Rec")
    return fig


def plot_segment_yield(df: pd.DataFrame, feature: str, target_col: str = "yield",
                       by: str = "tool_id", n_bins: int = 12) -> go.Figure:
    if by not in df.columns:
        return go.Figure().update_layout(title=f"{by} 컬럼 없음")
    s = df[[feature, target_col, by]].dropna()
    parts = []
    for grp, sub in s.groupby(by):
        if len(sub) < 30:
            continue
        try:
            sub = sub.copy()
            sub["bin"] = pd.qcut(sub[feature], q=n_bins, duplicates="drop")
            agg = sub.groupby("bin", observed=True).agg(
                mean_yield=(target_col, "mean"),
                bin_center=(feature, "mean"),
            ).reset_index(drop=True)
            agg["group"] = grp
            parts.append(agg)
        except ValueError:
            continue
    if not parts:
        return go.Figure().update_layout(title=f"{feature} segment plot: data 부족")
    plot_df = pd.concat(parts, ignore_index=True)
    return px.line(plot_df, x="bin_center", y="mean_yield", color="group",
                   markers=True, title=f"{feature} vs {target_col} by {by}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = importance_df.head(top_n).iloc[::-1]
    return px.bar(top, x="importance", y="feature", orientation="h",
                  title=f"Feature Importance (Top {top_n})")


# ===================== Evaluation visualisations =====================
# These charts are used by the evaluation report (HTML output) so a reviewer can
# judge BOTH model accuracy AND recommendation quality without reading code.


def plot_predicted_vs_actual(y_true, y_pred, title: str = "Predicted vs Actual (yield)") -> go.Figure:
    """45-degree reference line + scatter. Tight diagonal cluster = good fit."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig = px.scatter(x=y_true, y=y_pred, opacity=0.3,
                     labels={"x": "Actual", "y": "Predicted"}, title=title)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             name="y = x", line=dict(color="red", dash="dash")))
    return fig


def plot_residuals(y_true, y_pred, title: str = "Residual Plot") -> go.Figure:
    """Residual = pred - actual. Random scatter around 0 = unbiased model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_pred - y_true
    fig = px.scatter(x=y_pred, y=resid, opacity=0.3,
                     labels={"x": "Predicted", "y": "Residual (pred - actual)"},
                     title=title)
    fig.add_hline(y=0.0, line_color="red", line_dash="dash")
    return fig


def plot_recommendation_impact(recs: pd.DataFrame,
                               title: str = "Recommendation Impact (uplift vs coverage loss)") -> go.Figure:
    """Bubble: x=coverage_loss, y=expected_yield_uplift, size=|score|, color=confidence.

    Top-left (low coverage loss, high uplift) is the most valuable region.
    """
    if recs.empty:
        return go.Figure().update_layout(title=f"{title} — no data")
    df = recs.copy()
    df["abs_score"] = df["recommendation_score"].abs().clip(lower=0.05)
    fig = px.scatter(
        df, x="coverage_loss", y="expected_yield_uplift",
        size="abs_score", color="confidence_grade",
        hover_name="feature_name",
        category_orders={"confidence_grade": ["A", "B", "C"]},
        color_discrete_map={"A": "#2ca02c", "B": "#ff7f0e", "C": "#7f7f7f"},
        title=title,
    )
    fig.add_hline(y=0.0, line_color="gray", line_dash="dot")
    return fig


def plot_recommendation_score_breakdown(recs: pd.DataFrame,
                                        title: str = "Recommendation Score Breakdown") -> go.Figure:
    """Stacked bar per feature: uplift + bonus minus penalties = final score.

    Reconstructs components from rec rows (matches the formula in window_optimizer.recommend).
    """
    if recs.empty:
        return go.Figure().update_layout(title=f"{title} — no data")

    df = recs.copy()
    df["coverage_penalty_value"] = -0.5 * df["coverage_loss"]

    inst_pen = []
    bias_pen = []
    bonus = []
    for _, r in df.iterrows():
        residual = (r["recommendation_score"]
                    - r["expected_yield_uplift"]
                    - (-0.5 * r["coverage_loss"]))
        ipen = 0.0
        bpen = 0.0
        bn = 0.0
        if "tool/product 추천 방향 편향 있음" in str(r.get("reason_summary", "")):
            bpen = -0.3
        if "개선 기여 추정" in str(r.get("reason_summary", "")):
            bn = 0.2
        ipen = residual - bpen - bn
        if abs(ipen) < 1e-9:
            ipen = 0.0
        inst_pen.append(ipen)
        bias_pen.append(bpen)
        bonus.append(bn)
    df["instability_penalty_value"] = inst_pen
    df["bias_penalty_value"] = bias_pen
    df["item_bonus_value"] = bonus

    fig = go.Figure()
    fig.add_bar(name="Expected yield uplift", x=df["feature_name"],
                y=df["expected_yield_uplift"], marker_color="#2ca02c")
    fig.add_bar(name="Coverage penalty (-0.5*loss)", x=df["feature_name"],
                y=df["coverage_penalty_value"], marker_color="#ff7f0e")
    fig.add_bar(name="Instability/width penalty", x=df["feature_name"],
                y=df["instability_penalty_value"], marker_color="#d62728")
    fig.add_bar(name="Segment-bias penalty", x=df["feature_name"],
                y=df["bias_penalty_value"], marker_color="#9467bd")
    fig.add_bar(name="Item improvement bonus", x=df["feature_name"],
                y=df["item_bonus_value"], marker_color="#17becf")
    fig.add_trace(go.Scatter(name="Final score", x=df["feature_name"],
                             y=df["recommendation_score"],
                             mode="markers+lines",
                             marker=dict(color="black", size=10, symbol="diamond")))
    fig.update_layout(barmode="relative", title=title,
                      xaxis_title="feature", yaxis_title="score component")
    return fig


def plot_window_comparison(recs: pd.DataFrame,
                           title: str = "Current vs Recommended Window") -> go.Figure:
    """For each feature, draw current and recommended [LSL..USL] as horizontal bars side-by-side.

    Lets the reviewer see at a glance which features shift, narrow, or widen.
    """
    if recs.empty:
        return go.Figure().update_layout(title=f"{title} — no data")
    rows = []
    for _, r in recs.iterrows():
        f = r["feature_name"]
        rows.append({"feature": f, "type": "Current",
                     "lsl": r["current_lsl"], "usl": r["current_usl"],
                     "target": r["current_target"]})
        rows.append({"feature": f, "type": "Recommended",
                     "lsl": r["recommended_lsl"], "usl": r["recommended_usl"],
                     "target": r["recommended_target"]})
    plot_df = pd.DataFrame(rows)

    fig = go.Figure()
    for typ, color in [("Current", "gray"), ("Recommended", "#2ca02c")]:
        sub = plot_df[plot_df["type"] == typ]
        for _, r in sub.iterrows():
            y_label = f"{r['feature']} ({typ})"
            fig.add_trace(go.Scatter(
                x=[r["lsl"], r["usl"]], y=[y_label, y_label],
                mode="lines+markers", line=dict(color=color, width=6),
                marker=dict(size=10), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[r["target"]], y=[y_label],
                mode="markers", marker=dict(color=color, size=14, symbol="x"),
                showlegend=False,
            ))
    fig.update_layout(title=title + " (× = target, line = LSL..USL)",
                      xaxis_title="feature value")
    return fig


def plot_confidence_distribution(recs: pd.DataFrame,
                                 title: str = "Confidence Grade Distribution") -> go.Figure:
    if recs.empty:
        return go.Figure().update_layout(title=f"{title} — no data")
    counts = recs["confidence_grade"].value_counts().reindex(["A", "B", "C"]).fillna(0)
    fig = px.bar(x=counts.index, y=counts.values,
                 color=counts.index,
                 color_discrete_map={"A": "#2ca02c", "B": "#ff7f0e", "C": "#7f7f7f"},
                 labels={"x": "Confidence", "y": "# features"},
                 title=title)
    fig.update_layout(showlegend=False)
    return fig


# ===================== Data overview visualisations =====================
# Shown at the very top of the evaluation report so a reviewer first sees
# WHAT the data looks like before they see how the model/recommender behaves.


def plot_l1_distributions(df: pd.DataFrame, numeric_features: list[str]) -> go.Figure:
    """Small-multiple histograms of all L1 numeric features (3 columns)."""
    feats = [c for c in numeric_features if c in df.columns]
    if not feats:
        return go.Figure().update_layout(title="L1 분포: 컬럼 없음")
    long = df[feats].melt(var_name="feature", value_name="value").dropna()
    fig = px.histogram(long, x="value", facet_col="feature", facet_col_wrap=3,
                       nbins=40, title="L1 numeric feature 분포")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=600, showlegend=False)
    return fig


def plot_l3_distributions(df: pd.DataFrame) -> go.Figure:
    """Histogram of yield + total_fail_rate side by side."""
    cols = [c for c in ["yield", "total_fail_rate"] if c in df.columns]
    if not cols:
        return go.Figure().update_layout(title="L3 분포: 컬럼 없음")
    long = df[cols].melt(var_name="metric", value_name="value").dropna()
    fig = px.histogram(long, x="value", facet_col="metric", nbins=40,
                       title="L3 yield / total_fail_rate 분포",
                       color="metric",
                       color_discrete_map={"yield": "#2ca02c", "total_fail_rate": "#d62728"})
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    return fig


SHAPE_LABELS = {
    "metrology_x1": "양쪽 들림 (U)",
    "vm_x1": "양쪽 들림 (U, asymmetric)",
    "metrology_x2": "오른쪽 들림",
    "sensor_std_1": "오른쪽 들림 (mild)",
    "sensor_mean_1": "오른쪽 들림 + TOOL_A 교호작용",
    "metrology_x3": "왼쪽 들림",
    "vm_x2": "flat (noise)",
    "sensor_max_1": "flat (noise)",
    "sensor_slope_1": "flat (noise)",
}


def plot_ground_truth_relations(df: pd.DataFrame,
                                features: tuple[str, ...] = (
                                    "metrology_x1", "metrology_x2",
                                    "metrology_x3", "vm_x2",
                                ),
                                target_col: str = "total_fail_rate",
                                specs: pd.DataFrame | None = None,
                                n_bins: int = 20) -> go.Figure:
    """For representative features (one per shape), scatter + binned-mean fail rate.

    Optionally overlays current LSL/Target/USL vertical lines from specs so the
    reviewer can immediately see whether the current SPEC sits inside or outside
    the high-fail-rate region.
    """
    feats = [f for f in features if f in df.columns]
    if not feats or target_col not in df.columns:
        return go.Figure().update_layout(title="Ground-truth 관계: data 부족")

    spec_map = specs.set_index("feature_name").to_dict("index") if specs is not None else {}

    rows = (len(feats) + 1) // 2
    from plotly.subplots import make_subplots
    titles = [f"{f} — {SHAPE_LABELS.get(f, '')}" for f in feats]
    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles,
                        horizontal_spacing=0.10, vertical_spacing=0.15)
    for i, f in enumerate(feats):
        r = i // 2 + 1
        c = i % 2 + 1
        s = df[[f, target_col]].dropna()
        fig.add_trace(go.Scattergl(x=s[f], y=s[target_col], mode="markers",
                                   marker=dict(opacity=0.2, size=4, color="#1f77b4"),
                                   name=f, showlegend=False), row=r, col=c)
        try:
            s = s.copy()
            s["bin"] = pd.qcut(s[f], q=n_bins, duplicates="drop")
            agg = s.groupby("bin", observed=True).agg(
                mean_y=(target_col, "mean"),
                bin_center=(f, "mean"),
            ).reset_index(drop=True)
            fig.add_trace(go.Scatter(x=agg["bin_center"], y=agg["mean_y"],
                                     mode="lines+markers",
                                     line=dict(color="red", width=2),
                                     name="bin mean", showlegend=(i == 0)),
                          row=r, col=c)
        except ValueError:
            pass
        if f in spec_map:
            sp = spec_map[f]
            for key, dash, color in [("current_lsl", "dash", "gray"),
                                     ("current_target", "solid", "black"),
                                     ("current_usl", "dash", "gray")]:
                fig.add_vline(x=sp[key], line_dash=dash, line_color=color,
                              line_width=1.2, row=r, col=c)
        fig.update_xaxes(title_text=f, row=r, col=c)
        fig.update_yaxes(title_text=target_col, row=r, col=c)
    fig.update_layout(height=300 * rows,
                      title=(f"핵심 feature ↔ {target_col} 관계 "
                             "(네 가지 window 모양 예시; 회색 점선 = 현재 LSL/USL)"))
    return fig


def plot_yield_by_segment(df: pd.DataFrame, by: str = "tool_id",
                          target_col: str = "yield") -> go.Figure:
    """Box plot of yield per segment to surface tool/product systematic bias."""
    if by not in df.columns or target_col not in df.columns:
        return go.Figure().update_layout(title=f"{by} 별 {target_col}: 컬럼 없음")
    fig = px.box(df, x=by, y=target_col, color=by,
                 title=f"{by} 별 {target_col} 분포")
    fig.update_layout(showlegend=False)
    return fig
