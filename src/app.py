"""Streamlit UI and CLI entry for the process-window prototype.

Streamlit:  `streamlit run src/app.py`
CLI:        `python -m src.app --cli --generate-sample --train --recommend --top-n 10`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_generator import generate
from .data_loader import build_mart, load_data, load_specs
from .evaluation import generate_html_report, top_n_summary
from .feature_engineering import basic_impute, split_columns
from .modeling import feature_importance, train_and_score
from .window_optimizer import recommend

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")


def _prepare(target_col: str, split: str, kind: str, seed: int):
    l1, l3 = load_data(DATA_DIR)
    mart = build_mart(l1, l3)
    cols = split_columns(mart)
    mart = basic_impute(mart, cols["numeric_features"])
    model, metrics, train_df, test_df = train_and_score(
        mart, target_col, cols["numeric_features"], cols["categorical_features"],
        split=split, kind=kind, seed=seed,
    )
    return mart, cols, model, metrics, train_df, test_df


REPORT_PATH = OUT_DIR / "evaluation_report.html"


# ----------------------------- CLI -----------------------------

def run_cli(args) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate_sample:
        print(f"[gen] generating sample data into {DATA_DIR}/ ...")
        generate(DATA_DIR, seed=args.seed, n_wafers=args.n_wafers)
        print(f"[gen] done.")

    if not (args.train or args.recommend or args.report):
        return

    print(f"[train] model={args.model}, target={args.target}, split={args.split}")
    mart, cols, model, metrics, _, test_df = _prepare(
        target_col=args.target, split=args.split, kind=args.model, seed=args.seed,
    )
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "model_metrics.csv", index=False)
    print(f"[train] metrics: MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.3f}")

    recs = pd.DataFrame()
    if args.recommend or args.report:
        print("[recommend] generating window recommendations ...")
        specs = load_specs(DATA_DIR)
        recs = recommend(
            model, mart, specs,
            numeric_features=cols["numeric_features"],
            categorical_features=cols["categorical_features"],
            target_col=args.target,
            model_r2=metrics["R2"],
            min_coverage=args.min_coverage,
            top_n=args.top_n,
        )
        recs.to_csv(OUT_DIR / "recommendations.csv", index=False)
        print(f"[recommend] saved {len(recs)} rows to {OUT_DIR / 'recommendations.csv'}")
        if not recs.empty:
            print("\n--- Top recommendations ---")
            print(top_n_summary(recs, n=min(5, len(recs))).to_string(index=False))

    if args.report:
        print("\n[report] computing permutation importance + writing HTML report ...")
        features = cols["numeric_features"] + cols["categorical_features"]
        imp = feature_importance(model, test_df[features], test_df[args.target])
        path = generate_html_report(
            model=model, test_df=test_df, target_col=args.target,
            features=features, metrics=metrics,
            recommendations=recs, feature_importance_df=imp,
            out_path=REPORT_PATH,
            full_df=mart, numeric_features=cols["numeric_features"],
        )
        print(f"[report] wrote {path}")


# ----------------------------- Streamlit -----------------------------

def run_streamlit() -> None:
    import streamlit as st
    from .visualization import (
        plot_confidence_distribution, plot_feature_importance, plot_feature_vs_yield,
        plot_predicted_vs_actual, plot_recommendation_impact,
        plot_recommendation_score_breakdown, plot_residuals, plot_response_curve,
        plot_segment_yield, plot_window_comparison,
    )
    from .window_optimizer import _response_curve

    st.set_page_config(page_title="Process Window Prototype", layout="wide")
    st.title("Process Window Prototype")
    st.caption("L1 → L3 수율 예측 + SPEC window 추천 (synthetic data, prototype)")

    if "mart" not in st.session_state:
        st.session_state.mart = None

    tabs = st.tabs(["1. Data", "2. Train", "3. Optimization", "4. Exploration",
                    "5. Evaluation", "Help"])

    # ---- Data ----
    with tabs[0]:
        st.header("Data")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sample data 생성 (synthetic, ~3000 wafer)"):
                generate(DATA_DIR)
                st.success(f"생성 완료: {DATA_DIR}/")
        with c2:
            if st.button("data/ 로드"):
                try:
                    l1, l3 = load_data(DATA_DIR)
                    mart = build_mart(l1, l3)
                    cols = split_columns(mart)
                    mart = basic_impute(mart, cols["numeric_features"])
                    st.session_state.mart = mart
                    st.session_state.cols = cols
                    st.session_state.specs = load_specs(DATA_DIR)
                    st.success(f"Loaded mart: {mart.shape}")
                except FileNotFoundError as e:
                    st.error(f"파일 없음: {e}")

        if st.session_state.mart is not None:
            st.write("mart shape:", st.session_state.mart.shape)
            st.dataframe(st.session_state.mart.head(20))
            st.write("yield/fail rate 요약:")
            st.dataframe(st.session_state.mart[["yield", "total_fail_rate"]].describe())

    # ---- Train ----
    with tabs[1]:
        st.header("Train Baseline Model")
        if st.session_state.mart is None:
            st.info("먼저 Data 탭에서 데이터를 로드하세요.")
        else:
            split = st.selectbox("Split", ["random", "time"])
            kind = st.selectbox("모델", ["hgb", "rf"])
            target = st.selectbox("타겟", ["yield", "total_fail_rate"])
            if st.button("학습 실행"):
                model, metrics, _, test_df = train_and_score(
                    st.session_state.mart, target,
                    st.session_state.cols["numeric_features"],
                    st.session_state.cols["categorical_features"],
                    split=split, kind=kind,
                )
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.target = target
                st.success(
                    f"학습 완료. MAE={metrics['MAE']:.4f}  "
                    f"RMSE={metrics['RMSE']:.4f}  R²={metrics['R2']:.3f}"
                )
                features = (st.session_state.cols["numeric_features"]
                            + st.session_state.cols["categorical_features"])
                with st.spinner("Permutation importance 계산 중..."):
                    imp = feature_importance(model, test_df[features], test_df[target])
                st.plotly_chart(plot_feature_importance(imp), use_container_width=True)

    # ---- Optimization ----
    with tabs[2]:
        st.header("Optimization Mode — SPEC 추천")
        if "model" not in st.session_state:
            st.info("먼저 Train 탭에서 모델을 학습하세요.")
        else:
            min_cov = st.slider("최소 coverage", 0.3, 0.95, 0.6, 0.05)
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=10)
            if st.button("추천 생성"):
                recs = recommend(
                    st.session_state.model, st.session_state.mart, st.session_state.specs,
                    numeric_features=st.session_state.cols["numeric_features"],
                    categorical_features=st.session_state.cols["categorical_features"],
                    target_col=st.session_state.target,
                    model_r2=st.session_state.metrics["R2"],
                    min_coverage=min_cov,
                    top_n=int(top_n),
                )
                st.session_state.recs = recs

            if "recs" in st.session_state:
                recs = st.session_state.recs
                st.dataframe(recs)
                st.download_button("Download recommendations.csv",
                                   data=recs.to_csv(index=False),
                                   file_name="recommendations.csv")
                if not recs.empty:
                    sel = st.selectbox("상세 분석 feature", recs["feature_name"].tolist())
                    row = recs[recs["feature_name"] == sel].iloc[0]
                    cur_spec = {"lsl": row["current_lsl"], "target": row["current_target"],
                                "usl": row["current_usl"]}
                    rec_spec = {"lsl": row["recommended_lsl"], "target": row["recommended_target"],
                                "usl": row["recommended_usl"]}
                    st.write("**Reason**:", row["reason_summary"])
                    st.plotly_chart(plot_feature_vs_yield(
                        st.session_state.mart, sel, st.session_state.target,
                        current_spec=cur_spec, recommended_spec=rec_spec,
                    ), use_container_width=True)
                    curve = _response_curve(
                        st.session_state.model, st.session_state.mart, sel,
                        st.session_state.cols["numeric_features"],
                        st.session_state.cols["categorical_features"],
                    )
                    st.plotly_chart(plot_response_curve(curve, cur_spec, rec_spec, sel),
                                    use_container_width=True)

    # ---- Exploration ----
    with tabs[3]:
        st.header("Exploration Mode — L1↔L3 관계 탐색")
        if st.session_state.mart is None:
            st.info("먼저 Data 탭에서 데이터를 로드하세요.")
        else:
            mart = st.session_state.mart
            feat = st.selectbox("Feature", st.session_state.cols["numeric_features"])
            tgt_options = [c for c in ["yield", "total_fail_rate",
                                       "eds_item_001_fail_rate",
                                       "eds_item_002_fail_rate",
                                       "eds_item_003_fail_rate"] if c in mart.columns]
            tgt = st.selectbox("Target", tgt_options)
            by = st.selectbox("Segment 기준", ["tool_id", "product", "chamber_id", "process_step"])
            st.plotly_chart(plot_feature_vs_yield(mart, feat, tgt),
                            use_container_width=True)
            st.plotly_chart(plot_segment_yield(mart, feat, tgt, by=by),
                            use_container_width=True)

    # ---- Evaluation ----
    with tabs[4]:
        st.header("Evaluation — 모델 정확도 + 추천 품질")
        if "model" not in st.session_state or "recs" not in st.session_state:
            st.info("Train 탭에서 학습하고 Optimization 탭에서 추천을 생성한 뒤 여기서 보세요.")
        else:
            features = (st.session_state.cols["numeric_features"]
                        + st.session_state.cols["categorical_features"])
            target = st.session_state.target
            # use full mart for visual evaluation in UI (synthetic data, OK)
            test_view = st.session_state.mart
            y_true = test_view[target].to_numpy()
            y_pred = st.session_state.model.predict(test_view[features])

            st.subheader("1. 모델 정확도")
            mcol = st.columns(3)
            mcol[0].metric("MAE", f"{st.session_state.metrics['MAE']:.4f}")
            mcol[1].metric("RMSE", f"{st.session_state.metrics['RMSE']:.4f}")
            mcol[2].metric("R²", f"{st.session_state.metrics['R2']:.3f}")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_predicted_vs_actual(y_true, y_pred),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_residuals(y_true, y_pred),
                                use_container_width=True)

            st.subheader("2. 추천 품질")
            recs = st.session_state.recs
            st.plotly_chart(plot_recommendation_impact(recs),
                            use_container_width=True)
            st.plotly_chart(plot_recommendation_score_breakdown(recs),
                            use_container_width=True)
            st.plotly_chart(plot_window_comparison(recs),
                            use_container_width=True)
            st.plotly_chart(plot_confidence_distribution(recs),
                            use_container_width=True)

            st.subheader("3. HTML 리포트 export")
            if st.button("evaluation_report.html 생성"):
                imp = feature_importance(st.session_state.model,
                                         test_view[features], test_view[target])
                path = generate_html_report(
                    model=st.session_state.model, test_df=test_view,
                    target_col=target, features=features,
                    metrics=st.session_state.metrics,
                    recommendations=recs, feature_importance_df=imp,
                    out_path=REPORT_PATH,
                    full_df=st.session_state.mart,
                    numeric_features=st.session_state.cols["numeric_features"],
                )
                st.success(f"저장 완료: {path}")
                with open(path, "rb") as fh:
                    st.download_button("Download evaluation_report.html",
                                       data=fh.read(),
                                       file_name="evaluation_report.html",
                                       mime="text/html")

    # ---- Help ----
    with tabs[5]:
        st.header("사용 가이드")
        st.markdown(
            "1. **Data** 탭: sample 생성 또는 기존 data/ 로드\n"
            "2. **Train** 탭: 모델 학습 (split / 모델 / 타겟 선택)\n"
            "3. **Optimization** 탭: SPEC 추천 생성 → 상세 plot 확인\n"
            "4. **Exploration** 탭: feature와 L3 간 관계 시각화\n"
            "5. **Evaluation** 탭: 모델 정확도 + 추천 품질 시각화, HTML 리포트 export\n\n"
            "> ⚠ **주의**: 본 추천은 통계적 상관관계 기반이며 **인과관계가 아님**.  \n"
            "> 실제 SPEC 변경 전 공정 엔지니어 도메인 검토 + pilot lot 검증 필수."
        )


# ----------------------------- entrypoint -----------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process Window Prototype")
    p.add_argument("--cli", action="store_true", help="Run in CLI mode (no Streamlit).")
    p.add_argument("--generate-sample", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--recommend", action="store_true")
    p.add_argument("--report", action="store_true",
                   help="Write outputs/evaluation_report.html (implies train+recommend).")
    p.add_argument("--target", default="yield", choices=["yield", "total_fail_rate"])
    p.add_argument("--split", default="random", choices=["random", "time"])
    p.add_argument("--model", default="hgb", choices=["hgb", "rf"])
    p.add_argument("--n-wafers", type=int, default=3000)
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cli:
        run_cli(args)
    else:
        run_streamlit()


if __name__ == "__main__":
    main()
