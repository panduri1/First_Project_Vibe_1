from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "predictive_maintenance"
DATA_PATH = OUTPUT_DIR / "processed_predictive_maintenance.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "model_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
EXPERIMENT_SUMMARY_PATH = EXPERIMENT_DIR / "experiment_summary.csv"
EXPERIMENT_STATS_PATH = EXPERIMENT_DIR / "experiment_stat_tests.csv"
EXPERIMENT_FOLD_PATH = EXPERIMENT_DIR / "experiment_fold_results.csv"
EXPERIMENT_OOF_PATH = EXPERIMENT_DIR / "oof_predictions.csv"

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_gap",
    "power_proxy",
    "wear_per_torque",
]
FEATURE_COLS = NUMERIC_COLS + ["Type"]


st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .app-title-banner {
        background: linear-gradient(90deg, #0b5ed7 0%, #1976d2 50%, #0b5ed7 100%);
        color: white;
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 22px rgba(11, 94, 215, 0.22);
    }
    .app-title-banner h1 {
        margin: 0;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        text-align: center;
    }
    .ax-robot {
        min-width: 92px;
        height: 64px;
        border: 2px solid rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(255, 255, 255, 0.14);
        font-weight: 800;
        line-height: 1.05;
    }
    .ax-robot .face {
        font-size: 1.35rem;
    }
    .ax-robot .label {
        font-size: 0.95rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background-color: #e8f1ff;
        color: #0b5ed7;
        border: 1px solid #0b5ed7;
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 700;
        margin-right: 6px;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background-color: #0b5ed7;
        color: white;
        border-color: #0b5ed7;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background-color: #1976d2;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    experiment_summary = pd.read_csv(EXPERIMENT_SUMMARY_PATH) if EXPERIMENT_SUMMARY_PATH.exists() else pd.DataFrame()
    experiment_stats = pd.read_csv(EXPERIMENT_STATS_PATH) if EXPERIMENT_STATS_PATH.exists() else pd.DataFrame()
    experiment_folds = pd.read_csv(EXPERIMENT_FOLD_PATH) if EXPERIMENT_FOLD_PATH.exists() else pd.DataFrame()
    return df, metrics, predictions, feature_importance, experiment_summary, experiment_stats, experiment_folds


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def model_probability_column(model_name: str) -> str:
    return f"{model_name}_probability"


def model_prediction_column(model_name: str) -> str:
    return f"{model_name}_prediction"


def bootstrap_metric_ci(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, metric: str, n_boot: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric == "f1":
            values.append(f1_score(y_true[idx], pred[idx], zero_division=0))
        elif metric == "recall":
            values.append(recall_score(y_true[idx], pred[idx], zero_division=0))
        elif metric == "pr_auc":
            values.append(average_precision_score(y_true[idx], proba[idx]))
        elif metric == "roc_auc":
            values.append(roc_auc_score(y_true[idx], proba[idx]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def mcnemar_p_value(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        return 1.0
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stats.chi2.sf(statistic, 1))


def paired_bootstrap_delta(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    pred_a: np.ndarray,
    proba_b: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int = 500,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(7)
    deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric == "f1":
            score_a = f1_score(y_true[idx], pred_a[idx], zero_division=0)
            score_b = f1_score(y_true[idx], pred_b[idx], zero_division=0)
        elif metric == "pr_auc":
            score_a = average_precision_score(y_true[idx], proba_a[idx])
            score_b = average_precision_score(y_true[idx], proba_b[idx])
        else:
            score_a = recall_score(y_true[idx], pred_a[idx], zero_division=0)
            score_b = recall_score(y_true[idx], pred_b[idx], zero_division=0)
        deltas.append(score_a - score_b)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_experiment_table(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["actual"].to_numpy()
    rows = []
    for _, row in metrics.iterrows():
        model_name = str(row["model"])
        proba = predictions[model_probability_column(model_name)].to_numpy()
        pred = predictions[model_prediction_column(model_name)].to_numpy()
        f1_low, f1_high = bootstrap_metric_ci(y_true, proba, pred, "f1")
        pr_low, pr_high = bootstrap_metric_ci(y_true, proba, pred, "pr_auc")
        rows.append(
            {
                **row.to_dict(),
                "f1_ci95": f"{f1_low:.3f} - {f1_high:.3f}",
                "pr_auc_ci95": f"{pr_low:.3f} - {pr_high:.3f}",
                "false_negative": int(((y_true == 1) & (pred == 0)).sum()),
                "false_positive": int(((y_true == 0) & (pred == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_significance_table(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    best_model = str(metrics.iloc[0]["model"])
    y_true = predictions["actual"].to_numpy()
    best_proba = predictions[model_probability_column(best_model)].to_numpy()
    best_pred = predictions[model_prediction_column(best_model)].to_numpy()
    rows = []
    for model_name in metrics["model"]:
        model_name = str(model_name)
        if model_name == best_model:
            continue
        proba = predictions[model_probability_column(model_name)].to_numpy()
        pred = predictions[model_prediction_column(model_name)].to_numpy()
        f1_delta, f1_low, f1_high = paired_bootstrap_delta(y_true, best_proba, best_pred, proba, pred, "f1")
        pr_delta, pr_low, pr_high = paired_bootstrap_delta(y_true, best_proba, best_pred, proba, pred, "pr_auc")
        rows.append(
            {
                "comparison": f"{best_model} vs {model_name}",
                "mcnemar_p_value": mcnemar_p_value(y_true, best_pred, pred),
                "f1_delta_mean": f1_delta,
                "f1_delta_ci95": f"{f1_low:.3f} - {f1_high:.3f}",
                "pr_auc_delta_mean": pr_delta,
                "pr_auc_delta_ci95": f"{pr_low:.3f} - {pr_high:.3f}",
                "interpretation": "CI가 0을 넘지 않으면 개선이 불안정합니다.",
            }
        )
    return pd.DataFrame(rows)


def clean_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def feature_stat_tests(df: pd.DataFrame, feature_importance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    normal = df[df["Target"] == 0]
    failure = df[df["Target"] == 1]
    importance_map = {
        clean_feature_name(row["feature"]): row["importance"]
        for _, row in feature_importance.iterrows()
    }
    for column in NUMERIC_COLS:
        stat, p_value = stats.mannwhitneyu(normal[column], failure[column], alternative="two-sided")
        rows.append(
            {
                "feature": column,
                "importance": importance_map.get(column, 0),
                "normal_median": normal[column].median(),
                "failure_median": failure[column].median(),
                "median_delta": failure[column].median() - normal[column].median(),
                "cohen_d_failure_minus_normal": cohen_d(normal[column], failure[column]),
                "mannwhitney_p_value": p_value,
            }
        )
    result = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return result


def make_full_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    scored = df.copy()
    scored["best_model_probability"] = model.predict_proba(scored[FEATURE_COLS])[:, 1]
    scored["best_model_prediction"] = (scored["best_model_probability"] >= 0.5).astype(int)
    scored["error_type"] = np.select(
        [
            (scored["Target"] == 1) & (scored["best_model_prediction"] == 0),
            (scored["Target"] == 0) & (scored["best_model_prediction"] == 1),
        ],
        ["False Negative", "False Positive"],
        default="Correct",
    )
    scored["decision_margin"] = (scored["best_model_probability"] - 0.5).abs()
    return scored


def section_header(title: str, text: str) -> None:
    st.subheader(title)
    st.caption(text)


df, metrics, predictions, feature_importance, experiment_summary, experiment_stats, experiment_folds = load_tables()
model = load_model()
best_model = str(metrics.iloc[0]["model"])
scored_full = make_full_predictions(df, model)

st.markdown(
    """
    <div class="app-title-banner">
        <div class="ax-robot"><div class="face">[o_o]</div><div class="label">AX ROBOT</div></div>
        <h1>Machine Predictive Maintenance Dashboard</h1>
        <div class="ax-robot"><div class="face">[o_o]</div><div class="label">AX ROBOT</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("실험 결과, 입력 변수 분석, 오분류 심화 분석을 한 화면에서 검토합니다.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Failure Rate", f"{df['Target'].mean():.2%}")
col3.metric("Best Model", best_model)
col4.metric("Best F1", f"{metrics.iloc[0]['f1']:.3f}")

tab1, tab2, tab3 = st.tabs(["탭1: 실험 결과", "탭2: 입력 변수 분석", "탭3: 오분류 데이터 심화 분석"])

with tab1:
    if not experiment_summary.empty:
        section_header(
            "5-Fold Stratified CV + OOF 비교실험",
            "실제 실행된 19개 실험 조합의 CV 평균/표준편차 결과입니다. 같은 fold split으로 모델/전처리/샘플링/하이퍼파라미터를 비교했습니다.",
        )
        st.dataframe(experiment_summary, use_container_width=True)
        best_exp = experiment_summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Experiment", str(best_exp["experiment_id"]))
        c2.metric("Best Model", str(best_exp["model_name"]))
        c3.metric("Mean F1", f"{best_exp['f1_mean']:.3f}")
        c4.metric("Mean PR-AUC", f"{best_exp['pr_auc_mean']:.3f}")

        fig = px.scatter(
            experiment_summary,
            x="recall_mean",
            y="precision_mean",
            size="f1_mean",
            color="stage",
            hover_data=["experiment_id", "model_name", "sampling_strategy", "outlier_strategy", "hyperparameter_config"],
            title="Experiment Trade-off: Precision vs Recall",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not experiment_stats.empty:
            section_header(
                "OOF 기반 통계검증",
                "baseline(exp001 Logistic Regression) 대비 fold-level Wilcoxon test와 OOF bootstrap CI를 정리했습니다.",
            )
            st.dataframe(experiment_stats, use_container_width=True)
            st.caption("fold가 5개라 Holm 보정 후 p-value는 보수적으로 해석해야 합니다. 효과크기와 CI를 함께 확인하세요.")

        section_header("실험에서 확인된 효과", "실제 CV 실험 결과 기준의 해석입니다.")
        st.markdown(
            """
            - `LightGBM + all_engineered + tuned_deeper`가 F1 기준 최고 조합으로 선택되었습니다.
            - `random_over`는 recall을 끌어올리는 데 유리했지만, 최고 F1은 tuned LightGBM이 더 좋았습니다.
            - `SMOTE`와 `IQR clipping`은 recall은 높일 수 있지만 false positive 증가 가능성이 있어 운영 threshold 검토가 필요합니다.
            - `missing_indicator`는 현재 원본 결측치가 없어서 `median_mode`와 거의 같은 결과를 냈습니다.
            - `stress_missing_5pct`는 성능이 하락하여, 실데이터 결측 발생 시 robust preprocessing이 필요하다는 신호입니다.
            """
        )

        st.download_button(
            "5-fold 실험 요약 CSV 다운로드",
            experiment_summary.to_csv(index=False, encoding="utf-8-sig"),
            file_name="experiment_summary.csv",
            mime="text/csv",
        )

    section_header(
        "Holdout 기준 초기 실험 결과",
        "초기 단일 holdout 실험 결과입니다. 위 5-fold 실험 결과가 더 엄밀한 비교 기준입니다.",
    )
    experiment_table = build_experiment_table(metrics, predictions)
    st.dataframe(experiment_table, use_container_width=True)

    section_header("통계검증", "최고 모델과 다른 모델을 같은 holdout prediction 기준으로 짝지어 비교합니다.")
    significance_table = build_significance_table(metrics, predictions)
    st.dataframe(significance_table, use_container_width=True)
    st.info(
        "현재 통계검증은 단일 holdout prediction 기반입니다. 엄밀한 결론은 다음 단계의 5-fold stratified CV + OOF prediction에서 재검증해야 합니다."
    )

    section_header("방법론별 효과 해석", "이번 실험에서 관찰된 패턴을 요약합니다.")
    st.markdown(
        f"""
        - **LightGBM**: F1 기준 최고 모델입니다. Precision과 recall 균형이 가장 좋습니다.
        - **RandomForest / XGBoost**: recall은 높지만 precision이 LightGBM보다 낮아 false alarm 비용을 확인해야 합니다.
        - **Logistic Regression**: recall은 높지만 precision이 낮습니다. 선형 모델이 위험군을 넓게 잡는 baseline 역할을 합니다.
        - **현재 한계**: 아직 5-fold OOF, sampling, outlier policy, hyperparameter search를 모두 돌린 결과는 아니므로 이 결과는 1차 기준선입니다.
        """
    )

    section_header("향후 추가 실험", "다음 단계에서 우선순위를 두고 확장할 실험입니다.")
    st.markdown(
        """
        1. 5-fold stratified CV + OOF prediction으로 모델별 안정성 검증
        2. LightGBM/XGBoost의 `scale_pos_weight`, `num_leaves`, `max_depth`, `min_child_samples` 탐색
        3. SMOTE, random oversampling, class weight, threshold tuning 비교
        4. 이상치 처리 none vs p01/p99 clipping vs outlier flag 비교
        5. `Target`/`Failure Type` 불일치 27건 처리 정책별 성능 영향 비교
        """
    )

    st.download_button(
        "실험 테이블 CSV 다운로드",
        experiment_table.to_csv(index=False, encoding="utf-8-sig"),
        file_name="experiment_table.csv",
        mime="text/csv",
    )

with tab2:
    section_header(
        "Feature Importance",
        "최고 성능 모델의 feature importance입니다. 중요 변수와 낮은 중요도 변수를 나누어 Target별 차이를 봅니다.",
    )
    feature_importance_display = feature_importance.copy()
    feature_importance_display["clean_feature"] = feature_importance_display["feature"].map(clean_feature_name)
    st.plotly_chart(
        px.bar(
            feature_importance_display.sort_values("importance", ascending=True),
            x="importance",
            y="clean_feature",
            orientation="h",
            title="Feature Importance",
        ),
        use_container_width=True,
    )

    stats_table = feature_stat_tests(df, feature_importance)
    section_header("중요 변수의 통계적 차이", "정상/고장 그룹 간 Mann-Whitney U test와 효과크기(Cohen's d)를 계산합니다.")
    st.dataframe(stats_table, use_container_width=True)

    top_feature = st.selectbox("분포를 볼 변수", stats_table["feature"].tolist())
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.box(df, x="Target", y=top_feature, color="Target", title=f"{top_feature} by Target"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df, x=top_feature, color="Target", nbins=40, barmode="overlay", title=f"{top_feature} Distribution"), use_container_width=True)

    section_header("상관관계 힌트", "센서 변수끼리의 상관관계와 Target과의 방향성을 확인합니다.")
    corr = df[NUMERIC_COLS + ["Target"]].corr()
    fig = ff.create_annotated_heatmap(
        z=corr.round(2).values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu",
        showscale=True,
    )
    fig.update_layout(title="Correlation Heatmap", height=720)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **EDA 인사이트 요약**
        - `power_proxy`, `Tool wear [min]`, `Rotational speed [rpm]`, `Torque [Nm]`는 중요도가 높고, 설비 부하/마모 상태를 설명하는 핵심 축입니다.
        - `temperature_gap`은 열 방출 문제와 연결될 수 있어 고장 유형 분석에서 별도 확인할 가치가 있습니다.
        - `Type` 변수는 중요도가 낮지만, 제품 타입별 base rate 차이가 있을 수 있으므로 완전히 제거하기 전 CV로 확인해야 합니다.
        - 중요도가 낮은 변수라도 threshold 근처 케이스를 설명하는 데 도움을 줄 수 있으므로 해석용으로 유지하는 방안이 합리적입니다.
        """
    )

with tab3:
    best_prob_col = model_probability_column(best_model)
    best_pred_col = model_prediction_column(best_model)
    validation_errors = predictions[predictions["actual"] != predictions[best_pred_col]].copy()
    validation_errors["error_type"] = np.where(validation_errors["actual"] == 1, "False Negative", "False Positive")
    validation_errors["decision_margin"] = (validation_errors[best_prob_col] - 0.5).abs()

    section_header(
        "검증셋 오분류 샘플",
        "최고 성능 모델이 holdout 검증셋에서 틀린 케이스입니다. threshold와 가까운 케이스와 크게 틀린 케이스를 구분합니다.",
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Validation Errors", f"{len(validation_errors):,}")
    col_b.metric("False Negatives", f"{int((validation_errors['error_type'] == 'False Negative').sum()):,}")
    col_c.metric("False Positives", f"{int((validation_errors['error_type'] == 'False Positive').sum()):,}")

    error_type = st.selectbox("오분류 유형", ["All", "False Negative", "False Positive"])
    view_type = st.selectbox("샘플링 기준", ["threshold에 가까운 아쉬운 오분류", "마진이 큰 고확신 오분류", "랜덤 샘플"])
    sample_n = st.slider("표시할 샘플 수", 5, 50, 15)
    display_errors = validation_errors.copy()
    if error_type != "All":
        display_errors = display_errors[display_errors["error_type"] == error_type]
    if view_type == "threshold에 가까운 아쉬운 오분류":
        display_errors = display_errors.sort_values("decision_margin", ascending=True)
    elif view_type == "마진이 큰 고확신 오분류":
        display_errors = display_errors.sort_values("decision_margin", ascending=False)
    else:
        display_errors = display_errors.sample(min(sample_n, len(display_errors)), random_state=42)
    st.dataframe(display_errors.head(sample_n), use_container_width=True)

    section_header("아쉬운 오분류 vs 고확신 오분류 비교", "decision margin 기준 하위/상위 오분류의 feature 평균 차이를 비교합니다.")
    if len(validation_errors) >= 4:
        near = validation_errors.nsmallest(max(3, len(validation_errors) // 3), "decision_margin")
        far = validation_errors.nlargest(max(3, len(validation_errors) // 3), "decision_margin")
        compare = pd.DataFrame(
            {
                "near_threshold_mean": near[NUMERIC_COLS].mean(),
                "high_margin_wrong_mean": far[NUMERIC_COLS].mean(),
            }
        )
        compare["delta_high_minus_near"] = compare["high_margin_wrong_mean"] - compare["near_threshold_mean"]
        st.dataframe(compare, use_container_width=True)
        st.plotly_chart(
            px.bar(compare.reset_index(), x="index", y="delta_high_minus_near", title="High-margin wrong minus near-threshold wrong"),
            use_container_width=True,
        )

    section_header("라벨링 오류 가능성 탐색", "전체 데이터에 최고 모델을 적용해 라벨과 모델 확신이 강하게 충돌하는 행을 찾습니다.")
    suspicious = scored_full[
        ((scored_full["Target"] == 0) & (scored_full["best_model_probability"] >= 0.95))
        | ((scored_full["Target"] == 1) & (scored_full["best_model_probability"] <= 0.05))
        | ((scored_full["Target"] == 0) & (scored_full["Failure Type"] != "No Failure"))
        | ((scored_full["Target"] == 1) & (scored_full["Failure Type"] == "No Failure"))
    ].copy()
    suspicious["label_issue_hint"] = np.select(
        [
            (suspicious["Target"] == 0) & (suspicious["Failure Type"] != "No Failure"),
            (suspicious["Target"] == 1) & (suspicious["Failure Type"] == "No Failure"),
            (suspicious["Target"] == 0) & (suspicious["best_model_probability"] >= 0.95),
            (suspicious["Target"] == 1) & (suspicious["best_model_probability"] <= 0.05),
        ],
        [
            "Target=0 but Failure Type indicates failure",
            "Target=1 but Failure Type says No Failure",
            "Model strongly predicts failure for Target=0",
            "Model strongly predicts normal for Target=1",
        ],
        default="Review",
    )
    st.dataframe(
        suspicious[
            [
                "UDI",
                "Product ID",
                "Type",
                "Target",
                "Failure Type",
                "best_model_probability",
                "best_model_prediction",
                "label_issue_hint",
            ]
            + NUMERIC_COLS
        ].sort_values("best_model_probability", ascending=False).head(100),
        use_container_width=True,
    )

    section_header("모델 고도화 아이디어", "오분류 분석에서 이어지는 다음 개선 과제입니다.")
    st.markdown(
        """
        - **라벨 정책 정리**: `Target`과 `Failure Type` 불일치 27건을 제외/재라벨링/유지 세팅으로 나누어 성능 변화를 비교합니다.
        - **threshold 최적화**: false negative 비용을 크게 둔 cost-based threshold를 선택합니다.
        - **오분류 전용 feature**: threshold 근처 케이스와 고확신 오분류를 구분하는 변수 조합을 추가합니다.
        - **고장 유형 2단계 모델**: 1단계는 고장 여부, 2단계는 고장 유형으로 분리하여 모델 목적을 명확히 합니다.
        - **OOF 기반 재검증**: 지금 앱의 통계검증은 holdout 기반이므로, 5-fold OOF 실험 결과로 같은 탭을 갱신해야 합니다.
        """
    )

    section_header("Single Prediction", "센서값을 직접 입력해 고장 확률을 확인합니다.")
    with st.form("prediction_form"):
        type_value = st.selectbox("Type", sorted(df["Type"].unique()))
        air_temp = st.number_input("Air temperature [K]", value=float(df["Air temperature [K]"].median()))
        process_temp = st.number_input("Process temperature [K]", value=float(df["Process temperature [K]"].median()))
        rpm = st.number_input("Rotational speed [rpm]", value=float(df["Rotational speed [rpm]"].median()))
        torque = st.number_input("Torque [Nm]", value=float(df["Torque [Nm]"].median()))
        tool_wear = st.number_input("Tool wear [min]", value=float(df["Tool wear [min]"].median()))
        submitted = st.form_submit_button("Predict failure risk")

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "Type": type_value,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": process_temp,
                    "Rotational speed [rpm]": rpm,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": tool_wear,
                }
            ]
        )
        row["temperature_gap"] = row["Process temperature [K]"] - row["Air temperature [K]"]
        row["power_proxy"] = row["Rotational speed [rpm]"] * row["Torque [Nm]"]
        row["wear_per_torque"] = row["Tool wear [min]"] / row["Torque [Nm]"].replace(0, pd.NA)
        probability = model.predict_proba(row[FEATURE_COLS])[0, 1]
        st.metric("Failure probability", f"{probability:.2%}")
