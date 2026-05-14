from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "kaggle" / "machine_predictive_maintenance_classification" / "predictive_maintenance.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance"
REPORT_DIR = PROJECT_ROOT / "reports"
DOC_DIR = PROJECT_ROOT / "docs" / "260514"

PROCESSED_PATH = OUTPUT_DIR / "processed_predictive_maintenance.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "model_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
SQLITE_PATH = OUTPUT_DIR / "predictive_maintenance.sqlite"
HTML_REPORT_PATH = REPORT_DIR / "data_gathering_predictive_maintenance_report.html"
STREAMLIT_APP_PATH = REPORT_DIR / "data_gathering_predictive_maintenance_streamlit_app.py"
RUN_LOG_PATH = DOC_DIR / "predictive_maintenance_run_log.md"

ID_COLUMNS = ["UDI", "Product ID"]
TARGET = "Target"
LEAKAGE_COLUMNS = ["Failure Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_gap",
    "power_proxy",
    "wear_per_torque",
]
CATEGORICAL_FEATURES = ["Type"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA_PATH)
    processed = raw.copy()
    processed["temperature_gap"] = processed["Process temperature [K]"] - processed["Air temperature [K]"]
    processed["power_proxy"] = processed["Rotational speed [rpm]"] * processed["Torque [Nm]"]
    processed["wear_per_torque"] = processed["Tool wear [min]"] / processed["Torque [Nm]"].replace(0, pd.NA)

    for column in NUMERIC_FEATURES:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
        processed[column] = processed[column].fillna(processed[column].median())
    processed["Type"] = processed["Type"].fillna(processed["Type"].mode().iloc[0])
    processed[TARGET] = processed[TARGET].astype(int)

    processed.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    return raw, processed


def save_sqlite(raw: pd.DataFrame, processed: pd.DataFrame) -> None:
    with sqlite3.connect(SQLITE_PATH) as conn:
        raw.to_sql("raw_predictive_maintenance", conn, if_exists="replace", index=False)
        processed.to_sql("processed_predictive_maintenance", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_target ON processed_predictive_maintenance(Target)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_type ON processed_predictive_maintenance(Type)")


def save_eda(raw: pd.DataFrame, processed: pd.DataFrame) -> dict[str, int]:
    raw.dtypes.astype(str).rename("dtype").to_csv(OUTPUT_DIR / "dtypes.csv", encoding="utf-8-sig")
    raw.isna().sum().rename("missing_count").to_csv(OUTPUT_DIR / "missing_values.csv", encoding="utf-8-sig")
    processed.describe(include="all").transpose().to_csv(OUTPUT_DIR / "summary_statistics.csv", encoding="utf-8-sig")
    processed[TARGET].value_counts().sort_index().rename("count").to_csv(
        OUTPUT_DIR / "target_distribution.csv", encoding="utf-8-sig"
    )
    processed["Failure Type"].value_counts().rename("count").to_csv(
        OUTPUT_DIR / "failure_type_distribution.csv", encoding="utf-8-sig"
    )

    mismatch = processed[
        ((processed[TARGET] == 0) & (processed["Failure Type"] != "No Failure"))
        | ((processed[TARGET] == 1) & (processed["Failure Type"] == "No Failure"))
    ]
    mismatch.to_csv(OUTPUT_DIR / "target_failure_type_mismatch.csv", index=False, encoding="utf-8-sig")

    plot_target_distribution(processed)
    plot_failure_type_distribution(processed)
    plot_correlation_heatmap(processed)
    plot_feature_boxplots(processed)

    return {
        "rows": len(processed),
        "columns": processed.shape[1],
        "missing_before": int(raw.isna().sum().sum()),
        "missing_after": int(processed.isna().sum().sum()),
        "duplicates": int(processed.duplicated().sum()),
        "mismatch_count": int(len(mismatch)),
    }


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target Distribution")
    save_plot(OUTPUT_DIR / "target_distribution.png")


def plot_failure_type_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4))
    order = df["Failure Type"].value_counts().index
    sns.countplot(data=df, y="Failure Type", order=order)
    plt.title("Failure Type Distribution")
    save_plot(OUTPUT_DIR / "failure_type_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 7))
    corr_cols = NUMERIC_FEATURES + [TARGET]
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Numeric Correlation Heatmap")
    save_plot(OUTPUT_DIR / "correlation_heatmap.png")


def plot_feature_boxplots(df: pd.DataFrame) -> None:
    for column in NUMERIC_FEATURES:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y=column)
        plt.title(f"{column} by Target")
        safe_name = column.lower().replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
        save_plot(OUTPUT_DIR / f"boxplot_{safe_name}.png")


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, str]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = df[features]
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=20,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        ),
    }

    metrics_rows = []
    prediction_frame = x_test.copy()
    prediction_frame["actual"] = y_test.values
    fitted_models: dict[str, Pipeline] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        proba = pipeline.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fitted_models[model_name] = pipeline
        prediction_frame[f"{model_name}_probability"] = proba
        prediction_frame[f"{model_name}_prediction"] = pred
        metrics_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
            }
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(["f1", "recall", "pr_auc"], ascending=False)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_proba = prediction_frame[f"{best_model_name}_probability"].to_numpy()
    best_pred = prediction_frame[f"{best_model_name}_prediction"].to_numpy()

    prediction_frame.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    metrics.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    joblib.dump(best_model, MODEL_PATH)

    save_model_plots(y_test, best_pred, best_proba, best_model_name)
    save_classification_report(y_test, best_pred, best_model_name)
    save_feature_importance(best_model, best_model_name)

    return metrics, prediction_frame, best_model, best_model_name


def save_model_plots(y_true: pd.Series, pred, proba, model_name: str) -> None:
    cm = confusion_matrix(y_true, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    save_plot(OUTPUT_DIR / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    save_plot(OUTPUT_DIR / "roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    save_plot(OUTPUT_DIR / "precision_recall_curve.png")


def save_classification_report(y_true: pd.Series, pred, model_name: str) -> None:
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    with (OUTPUT_DIR / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump({"model": model_name, "report": report}, file, indent=2)


def save_feature_importance(pipeline: Pipeline, model_name: str) -> None:
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = list(preprocessor.get_feature_names_out())
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_[0])
    else:
        return

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top = importance_df.head(15)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top, x="importance", y="feature")
    plt.title(f"Top Feature Importance - {model_name}")
    save_plot(OUTPUT_DIR / "feature_importance.png")


def create_html_report(summary: dict[str, int], metrics: pd.DataFrame, best_model_name: str) -> None:
    target_dist = pd.read_csv(OUTPUT_DIR / "target_distribution.csv").to_html(index=False)
    failure_dist = pd.read_csv(OUTPUT_DIR / "failure_type_distribution.csv").to_html(index=False)
    metrics_html = metrics.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    top_features = ""
    feature_path = OUTPUT_DIR / "feature_importance.csv"
    if feature_path.exists():
        top_features = pd.read_csv(feature_path).head(15).to_html(index=False, float_format=lambda x: f"{x:.4f}")

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Predictive Maintenance Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fafafa; }}
    .card .value {{ font-size: 24px; font-weight: 700; }}
    img {{ max-width: 760px; width: 100%; border: 1px solid #eee; margin: 8px 0 24px; }}
    table {{ border-collapse: collapse; margin: 8px 0 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; }}
  </style>
</head>
<body>
  <h1>Machine Predictive Maintenance Report</h1>
  <p>이 리포트는 Kaggle 데이터셋 <code>shivamb/machine-predictive-maintenance-classification</code>을 기반으로 생성했습니다.</p>
  <div class="cards">
    <div class="card"><div>Rows</div><div class="value">{summary["rows"]:,}</div></div>
    <div class="card"><div>Columns</div><div class="value">{summary["columns"]}</div></div>
    <div class="card"><div>Missing After</div><div class="value">{summary["missing_after"]}</div></div>
    <div class="card"><div>Best Model</div><div class="value">{best_model_name}</div></div>
  </div>

  <h2>Data Overview</h2>
  <p>고장 비율이 낮은 불균형 binary classification 데이터입니다. <code>Failure Type</code>은 결과성 컬럼이므로 1차 예측 feature에서 제외했습니다.</p>
  <h3>Target Distribution</h3>{target_dist}
  <img src="../outputs/data_gathering/predictive_maintenance/target_distribution.png" alt="Target distribution">
  <h3>Failure Type Distribution</h3>{failure_dist}
  <img src="../outputs/data_gathering/predictive_maintenance/failure_type_distribution.png" alt="Failure type distribution">

  <h2>EDA</h2>
  <img src="../outputs/data_gathering/predictive_maintenance/correlation_heatmap.png" alt="Correlation heatmap">

  <h2>Model Performance</h2>
  <p>고장 미탐(False Negative)을 줄이는 것이 중요하므로 recall, F1, PR-AUC를 함께 봐야 합니다.</p>
  {metrics_html}
  <img src="../outputs/data_gathering/predictive_maintenance/confusion_matrix.png" alt="Confusion matrix">
  <img src="../outputs/data_gathering/predictive_maintenance/roc_curve.png" alt="ROC curve">
  <img src="../outputs/data_gathering/predictive_maintenance/precision_recall_curve.png" alt="Precision recall curve">

  <h2>Feature Evidence</h2>
  {top_features}
  <img src="../outputs/data_gathering/predictive_maintenance/feature_importance.png" alt="Feature importance">

  <h2>Logic Tree Summary</h2>
  <ul>
    <li>Root: 설비가 고장 위험 상태인가?</li>
    <li>Branch 1: 공구 마모와 토크 조합이 고장 위험을 높이는지 확인.</li>
    <li>Branch 2: 공정 온도와 공기 온도 차이가 열 관련 고장 위험과 연결되는지 확인.</li>
    <li>Branch 3: 회전속도와 토크로 계산한 power proxy가 과부하 패턴을 설명하는지 확인.</li>
  </ul>
</body>
</html>
"""
    HTML_REPORT_PATH.write_text(html, encoding="utf-8")


def create_streamlit_app() -> None:
    app = f'''from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance" / "processed_predictive_maintenance.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance" / "model_metrics.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance" / "model_predictions.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "data_gathering" / "predictive_maintenance" / "best_model.joblib"

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Machine Predictive Maintenance Dashboard")

df = pd.read_csv(DATA_PATH)
metrics = pd.read_csv(METRICS_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)
model = joblib.load(MODEL_PATH)
best_model = metrics.iloc[0]["model"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{{len(df):,}}")
col2.metric("Failure Rate", f"{{df['Target'].mean():.2%}}")
col3.metric("Best Model", str(best_model))
col4.metric("Best F1", f"{{metrics.iloc[0]['f1']:.3f}}")

st.subheader("Target And Failure Type")
left, right = st.columns(2)
with left:
    st.plotly_chart(px.histogram(df, x="Target", title="Target Distribution"), use_container_width=True)
with right:
    failure_counts = df["Failure Type"].value_counts().reset_index()
    failure_counts.columns = ["Failure Type", "count"]
    st.plotly_chart(px.bar(failure_counts, x="count", y="Failure Type", orientation="h", title="Failure Type Distribution"), use_container_width=True)

st.subheader("EDA")
numeric_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_gap",
    "power_proxy",
    "wear_per_torque",
]
selected_feature = st.selectbox("Feature", numeric_cols)
st.plotly_chart(px.box(df, x="Target", y=selected_feature, color="Target", title=f"{{selected_feature}} by Target"), use_container_width=True)
st.plotly_chart(px.imshow(df[numeric_cols + ["Target"]].corr(), text_auto=".2f", title="Correlation Heatmap"), use_container_width=True)

st.subheader("Model Performance")
st.dataframe(metrics, use_container_width=True)
prob_cols = [c for c in predictions.columns if c.endswith("_probability")]
selected_prob = st.selectbox("Model probability column", prob_cols)
threshold = st.slider("Decision threshold", 0.01, 0.99, 0.50, 0.01)
tmp = predictions.copy()
tmp["threshold_prediction"] = (tmp[selected_prob] >= threshold).astype(int)
cm = pd.crosstab(tmp["actual"], tmp["threshold_prediction"], rownames=["Actual"], colnames=["Predicted"], dropna=False)
st.write("Confusion matrix at selected threshold")
st.dataframe(cm, use_container_width=True)
st.plotly_chart(px.histogram(tmp, x=selected_prob, color="actual", nbins=40, title="Predicted Probability Distribution"), use_container_width=True)

st.subheader("Single Prediction")
with st.form("prediction_form"):
    type_value = st.selectbox("Type", sorted(df["Type"].unique()))
    air_temp = st.number_input("Air temperature [K]", value=float(df["Air temperature [K]"].median()))
    process_temp = st.number_input("Process temperature [K]", value=float(df["Process temperature [K]"].median()))
    rpm = st.number_input("Rotational speed [rpm]", value=float(df["Rotational speed [rpm]"].median()))
    torque = st.number_input("Torque [Nm]", value=float(df["Torque [Nm]"].median()))
    tool_wear = st.number_input("Tool wear [min]", value=float(df["Tool wear [min]"].median()))
    submitted = st.form_submit_button("Predict failure risk")

if submitted:
    row = pd.DataFrame([{{
        "Type": type_value,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }}])
    row["temperature_gap"] = row["Process temperature [K]"] - row["Air temperature [K]"]
    row["power_proxy"] = row["Rotational speed [rpm]"] * row["Torque [Nm]"]
    row["wear_per_torque"] = row["Tool wear [min]"] / row["Torque [Nm]"].replace(0, pd.NA)
    probability = model.predict_proba(row)[0, 1]
    st.metric("Failure probability", f"{{probability:.2%}}")
'''
    STREAMLIT_APP_PATH.write_text(app, encoding="utf-8")


def write_run_log(summary: dict[str, int], metrics: pd.DataFrame, best_model_name: str) -> None:
    lines = [
        "# Predictive Maintenance Execution Log",
        "",
        "## 1. 유저가 요청한 사항",
        "- 다운로드한 Kaggle 예지보전 데이터로 EDA/전처리, ML 모델링, 성능평가 대시보드 생성을 실행.",
        "",
        "## 2. 너가 수행한 사항",
        "- Python 가상환경 `.venv` 생성 및 분석 패키지 설치.",
        "- CSV 로드, 파생변수 생성, 전처리 데이터 저장.",
        "- SQLite raw/processed 테이블 생성.",
        "- EDA 요약 CSV와 차트 생성.",
        "- Logistic Regression, RandomForest, XGBoost, LightGBM 모델 학습 및 평가.",
        "- HTML 리포트와 Streamlit 앱 생성.",
        "",
        "## 3. 그 중 완료된 사항",
        f"- 입력 데이터: `{DATA_PATH.as_posix()}`",
        f"- 행/컬럼 수: `{summary['rows']:,}` rows, `{summary['columns']}` columns",
        f"- 결측치: before `{summary['missing_before']}`, after `{summary['missing_after']}`",
        f"- 중복 행: `{summary['duplicates']}`",
        f"- `Target`/`Failure Type` 불일치: `{summary['mismatch_count']}`",
        f"- Best model: `{best_model_name}`",
        f"- Best F1: `{metrics.iloc[0]['f1']:.4f}`",
        f"- Best recall: `{metrics.iloc[0]['recall']:.4f}`",
        f"- EDA/model outputs: `{OUTPUT_DIR.as_posix()}`",
        f"- HTML report: `{HTML_REPORT_PATH.as_posix()}`",
        f"- Streamlit app: `{STREAMLIT_APP_PATH.as_posix()}`",
        "",
        "## 4. 문제발생 현상 및 향후 해결 방안",
        "- 문제: 고장 클래스 비율이 낮아 accuracy만으로는 모델 성능을 판단하기 어려움.",
        "  - 해결 방안: recall, F1, PR-AUC, threshold tuning 중심으로 평가.",
        "- 문제: `Failure Type`은 `Target`과 연결된 결과 컬럼이라 binary 예측 feature로 쓰면 데이터 누수가 발생함.",
        "  - 해결 방안: 1차 모델 feature에서 제외하고, 2차 고장 유형 분석 타깃으로만 사용.",
        "- 문제: `Target`과 `Failure Type` 불일치 27건이 있음.",
        "  - 해결 방안: 이후 고장 유형 모델링 전에 제외/재라벨링/별도 분석 중 하나로 정책 결정 필요.",
    ]
    RUN_LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw, processed = load_and_preprocess()
    save_sqlite(raw, processed)
    summary = save_eda(raw, processed)
    metrics, _, _, best_model_name = train_models(processed)
    create_html_report(summary, metrics, best_model_name)
    create_streamlit_app()
    write_run_log(summary, metrics, best_model_name)
    print(f"Best model: {best_model_name}")
    print(metrics.to_string(index=False))
    print(f"HTML report: {HTML_REPORT_PATH}")
    print(f"Streamlit app: {STREAMLIT_APP_PATH}")


if __name__ == "__main__":
    main()
