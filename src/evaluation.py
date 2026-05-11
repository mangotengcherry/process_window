"""Evaluation utilities: model metrics, window quality, top-N summary, HTML report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .visualization import (
    plot_confidence_distribution,
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_recommendation_impact,
    plot_recommendation_score_breakdown,
    plot_residuals,
    plot_window_comparison,
)


def model_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def window_quality(df: pd.DataFrame, rec_row: pd.Series,
                   target_col: str = "yield",
                   fail_col: str = "total_fail_rate") -> dict:
    """In-window vs out-of-window actual yield/fail-rate comparison."""
    f = rec_row["feature_name"]
    rec_lsl, rec_usl = rec_row["recommended_lsl"], rec_row["recommended_usl"]
    mask_in = (df[f] >= rec_lsl) & (df[f] <= rec_usl)
    mask_out = ~mask_in

    out = {
        "feature": f,
        "in_yield_mean": float(df.loc[mask_in, target_col].mean()) if mask_in.any() else float("nan"),
        "out_yield_mean": float(df.loc[mask_out, target_col].mean()) if mask_out.any() else float("nan"),
        "coverage": float(mask_in.mean()),
    }
    if fail_col in df.columns:
        out["in_fail_mean"] = float(df.loc[mask_in, fail_col].mean()) if mask_in.any() else float("nan")
        out["out_fail_mean"] = float(df.loc[mask_out, fail_col].mean()) if mask_out.any() else float("nan")
    return out


def top_n_summary(recommendations: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = ["feature_name", "target_shift", "expected_yield_uplift",
            "coverage_loss", "recommendation_score", "confidence_grade"]
    return recommendations.head(n)[cols].copy()


# ---------------------- HTML evaluation report ----------------------

def _fig_html(fig) -> str:
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="table", border=0, float_format="%.4f")


def generate_html_report(model: Pipeline,
                         test_df: pd.DataFrame,
                         target_col: str,
                         features: list[str],
                         metrics: dict,
                         recommendations: pd.DataFrame,
                         feature_importance_df: pd.DataFrame | None,
                         out_path: Path) -> Path:
    """Write a single-file HTML evaluation report combining model + recommendation views.

    Sections:
      1. Model accuracy (predicted vs actual, residuals, feature importance)
      2. Recommendation quality (impact bubble, score breakdown,
         current-vs-recommended windows, confidence distribution)
      3. Top recommendations table with reason_summary
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = test_df[target_col].to_numpy()
    y_pred = model.predict(test_df[features])

    metrics_table = pd.DataFrame([{
        "model": metrics.get("model", ""),
        "split": metrics.get("split", ""),
        "target": metrics.get("target", target_col),
        "n_train": metrics.get("n_train", ""),
        "n_test": metrics.get("n_test", ""),
        "MAE": metrics.get("MAE", float("nan")),
        "RMSE": metrics.get("RMSE", float("nan")),
        "R2": metrics.get("R2", float("nan")),
    }])

    fig_pa = plot_predicted_vs_actual(y_true, y_pred, f"Predicted vs Actual ({target_col})")
    fig_res = plot_residuals(y_true, y_pred)
    fig_imp = (plot_feature_importance(feature_importance_df)
               if feature_importance_df is not None else None)
    fig_impact = plot_recommendation_impact(recommendations)
    fig_break = plot_recommendation_score_breakdown(recommendations)
    fig_win = plot_window_comparison(recommendations)
    fig_conf = plot_confidence_distribution(recommendations)

    rec_show = recommendations[[
        "feature_name", "current_target", "recommended_target", "target_shift",
        "current_window_yield", "recommended_window_yield",
        "expected_yield_uplift", "coverage_current", "coverage_recommended",
        "coverage_loss", "recommendation_score", "confidence_grade",
        "reason_summary",
    ]].copy()

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Process Window — Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif;
          max-width: 1180px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
  h2 {{ border-bottom: 1px solid #aaa; padding-bottom: 4px; margin-top: 36px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .full {{ grid-column: 1 / span 2; }}
  .table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .table th, .table td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  .note {{ background: #fff7d6; border-left: 4px solid #c9a227; padding: 10px 14px;
           margin: 16px 0; font-size: 14px; }}
  .legend {{ font-size: 13px; color: #555; margin-top: -8px; }}
</style>
</head>
<body>

<h1>Process Window — Evaluation Report</h1>
<p class="legend">L1 → L3 yield 예측 + SPEC window 추천 프로토타입의 모델/추천 품질 평가.
모델 정확도 (1부) 와 추천 품질 (2부) 을 함께 본다.</p>

<div class="note">
  ⚠ 본 추천은 통계적 상관관계 기반이며 <b>인과관계가 아니다</b>.
  실제 SPEC 변경 전 도메인 엔지니어 검토 + pilot lot 검증 필수.
</div>

<h2>1. 모델 정확도</h2>
<p class="legend"><b>지표</b>: MAE = 평균 절대 오차 (작을수록 좋음),
RMSE = 큰 오차에 가중치, R² = 분산 설명력 (1.0 이상적, 0 이면 평균 예측 수준).</p>
{_table_html(metrics_table)}
<div class="grid">
  <div>
    <h3>Predicted vs Actual</h3>
    <p class="legend">점이 빨간 대각선에 모일수록 예측이 정확. 큰 산포 = 노이즈/누락 feature.</p>
    {_fig_html(fig_pa)}
  </div>
  <div>
    <h3>Residual Plot</h3>
    <p class="legend">잔차가 0 주변에서 무작위면 unbiased.
    한쪽으로 치우치거나 특정 예측구간에서 변동성이 커지면 모델 보완 필요.</p>
    {_fig_html(fig_res)}
  </div>
</div>
{"<h3>Feature Importance</h3><p class='legend'>Permutation importance — 해당 feature 셔플 시 yield 예측 성능이 얼마나 떨어지는지. 추천 결과 해석의 1차 근거.</p>" + _fig_html(fig_imp) if fig_imp is not None else ""}

<h2>2. 추천 품질</h2>
<p class="legend">단순한 R² 보다 <b>추천이 yield 개선에 기여하는가</b> 와
<b>현장에 적용 가능한가 (coverage 손실 / segment 편향)</b> 를 동시에 본다.</p>

<h3>Recommendation Impact (uplift × coverage loss)</h3>
<p class="legend">좌상단 (낮은 coverage loss + 높은 uplift) 일수록 가치 있는 추천.
색상은 Confidence (A=초록 최선 / B=주황 / C=회색 reference).</p>
{_fig_html(fig_impact)}

<h3>Score Breakdown</h3>
<p class="legend">각 추천의 final score 가 어떤 구성요소 (uplift + bonus − penalties) 로 나왔는지.
검은 다이아몬드 = 최종 점수.</p>
{_fig_html(fig_break)}

<h3>Current vs Recommended Window</h3>
<p class="legend">feature 별 현재 (회색) vs 추천 (초록) [LSL .. USL] 범위. × = target.
이동 방향, 좁힘/넓힘이 직관적으로 보인다.</p>
{_fig_html(fig_win)}

<h3>Confidence Distribution</h3>
{_fig_html(fig_conf)}

<h2>3. Top recommendations + reason</h2>
<p class="legend">recommendation_score 내림차순. <code>reason_summary</code> 는 공정 엔지니어 검토용 자연어 요약.</p>
{_table_html(rec_show)}

</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
