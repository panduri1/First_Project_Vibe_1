from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "kaggle" / "machine_predictive_maintenance_classification" / "predictive_maintenance.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "predictive_maintenance"
DOC_DIR = PROJECT_ROOT / "docs" / "260514"

FOLD_RESULTS_PATH = OUTPUT_DIR / "experiment_fold_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "experiment_summary.csv"
OOF_PATH = OUTPUT_DIR / "oof_predictions.csv"
STATS_PATH = OUTPUT_DIR / "experiment_stat_tests.csv"
CONFIG_PATH = OUTPUT_DIR / "experiment_configs.json"
LOG_PATH = DOC_DIR / "predictive_maintenance_experiment_run_log.md"

TARGET = "Target"
BASE_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED = ["temperature_gap", "power_proxy", "wear_per_torque"]
CATEGORICAL = ["Type"]
ID_COLS = ["UDI", "Product ID"]
LEAKAGE_COLS = ["Failure Type"]


class OutlierPolicyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features: list[str], strategy: str = "none"):
        self.numeric_features = numeric_features
        self.strategy = strategy

    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_: dict[str, tuple[float, float]] = {}
        if self.strategy == "iqr_clip":
            for col in self.numeric_features:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        elif self.strategy == "p01_p99_clip":
            for col in self.numeric_features:
                self.bounds_[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
        elif self.strategy == "outlier_flags":
            for col in self.numeric_features:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        if self.strategy in {"iqr_clip", "p01_p99_clip"}:
            for col, (low, high) in self.bounds_.items():
                X_out[col] = X_out[col].clip(low, high)
        elif self.strategy == "outlier_flags":
            for col, (low, high) in self.bounds_.items():
                X_out[f"{col}__outlier_flag"] = ((X_out[col] < low) | (X_out[col] > high)).astype(int)
        return X_out


@dataclass
class ExperimentConfig:
    experiment_id: str
    stage: str
    model_name: str
    feature_set: str
    missing_strategy: str
    outlier_strategy: str
    sampling_strategy: str
    hyperparameter_config: str
    threshold_rule: str
    model_params: dict[str, Any]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["temperature_gap"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power_proxy"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["wear_per_torque"] = df["Tool wear [min]"] / df["Torque [Nm]"].replace(0, np.nan)
    df["wear_per_torque"] = df["wear_per_torque"].fillna(df["wear_per_torque"].median())
    return df


def build_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("exp001", "baseline", "LogisticRegression", "base", "median_mode", "none", "none", "baseline", "0.5", {"class_weight": "balanced"}),
        ExperimentConfig("exp002", "baseline", "RandomForest", "base", "median_mode", "none", "class_weight", "baseline", "0.5", {"class_weight": "balanced", "max_depth": 8, "n_estimators": 300}),
        ExperimentConfig("exp003", "baseline", "ExtraTrees", "base", "median_mode", "none", "class_weight", "baseline", "0.5", {"class_weight": "balanced", "max_depth": 8, "n_estimators": 300}),
        ExperimentConfig("exp004", "baseline", "LightGBM", "base", "median_mode", "none", "scale_pos_weight", "baseline", "0.5", {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp005", "baseline", "XGBoost", "base", "median_mode", "none", "scale_pos_weight", "baseline", "0.5", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}),
        ExperimentConfig("exp006", "feature", "LightGBM", "all_engineered", "median_mode", "none", "scale_pos_weight", "feature_all", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp007", "feature", "XGBoost", "all_engineered", "median_mode", "none", "scale_pos_weight", "feature_all", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "max_depth": 4}),
        ExperimentConfig("exp008", "sampling", "LightGBM", "all_engineered", "median_mode", "none", "random_over", "over_sampling", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp009", "sampling", "LightGBM", "all_engineered", "median_mode", "none", "random_under", "under_sampling", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp010", "sampling", "LightGBM", "all_engineered", "median_mode", "none", "smote", "smote", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp011", "outlier", "LightGBM", "all_engineered", "median_mode", "p01_p99_clip", "scale_pos_weight", "clip_p01_p99", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp012", "outlier", "LightGBM", "all_engineered", "median_mode", "iqr_clip", "scale_pos_weight", "clip_iqr", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp013", "outlier", "LightGBM", "all_engineered", "median_mode", "outlier_flags", "scale_pos_weight", "outlier_flags", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp014", "missing", "LightGBM", "all_engineered", "missing_indicator", "none", "scale_pos_weight", "missing_indicator", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp015", "missing", "LightGBM", "all_engineered", "stress_missing_5pct", "none", "scale_pos_weight", "stress_missing_5pct", "0.5", {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 20}),
        ExperimentConfig("exp016", "hparam", "LightGBM", "all_engineered", "median_mode", "none", "scale_pos_weight", "tuned_regularized", "0.5", {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 50, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0}),
        ExperimentConfig("exp017", "hparam", "LightGBM", "all_engineered", "median_mode", "none", "scale_pos_weight", "tuned_deeper", "0.5", {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9}),
        ExperimentConfig("exp018", "hparam", "XGBoost", "all_engineered", "median_mode", "none", "scale_pos_weight", "tuned_regularized", "0.5", {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0}),
        ExperimentConfig("exp019", "hparam", "XGBoost", "all_engineered", "median_mode", "none", "scale_pos_weight", "tuned_deeper", "0.5", {"n_estimators": 400, "learning_rate": 0.04, "max_depth": 5, "min_child_weight": 1, "subsample": 0.9, "colsample_bytree": 0.9}),
    ]


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    numeric = BASE_NUMERIC.copy()
    if feature_set == "all_engineered":
        numeric += ENGINEERED
    return numeric, CATEGORICAL.copy()


def inject_missing_if_needed(X: pd.DataFrame, config: ExperimentConfig, fold: int) -> pd.DataFrame:
    if config.missing_strategy != "stress_missing_5pct":
        return X
    X_out = X.copy()
    rng = np.random.default_rng(10_000 + fold)
    numeric, _ = feature_columns(config.feature_set)
    for col in numeric:
        mask = rng.random(len(X_out)) < 0.05
        X_out.loc[mask, col] = np.nan
    return X_out


def make_model(config: ExperimentConfig, scale_pos_weight: float):
    params = dict(config.model_params)
    if config.model_name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, **params)
    if config.model_name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, min_samples_leaf=5, **params)
    if config.model_name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42, n_jobs=-1, min_samples_leaf=5, **params)
    if config.model_name == "LightGBM":
        if config.sampling_strategy == "scale_pos_weight":
            params["scale_pos_weight"] = scale_pos_weight
        return LGBMClassifier(random_state=42, verbosity=-1, **params)
    if config.model_name == "XGBoost":
        if config.sampling_strategy == "scale_pos_weight":
            params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(random_state=42, eval_metric="logloss", **params)
    raise ValueError(f"Unsupported model: {config.model_name}")


def make_sampler(strategy: str):
    if strategy == "random_over":
        return RandomOverSampler(random_state=42)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=42)
    if strategy == "smote":
        return SMOTE(random_state=42, k_neighbors=5)
    return None


def make_pipeline(config: ExperimentConfig, X_train: pd.DataFrame, y_train: pd.Series):
    numeric, categorical = feature_columns(config.feature_set)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    outlier = OutlierPolicyTransformer(numeric, config.outlier_strategy)
    add_indicator = config.missing_strategy == "missing_indicator"
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ]
    )
    model = make_model(config, scale_pos_weight)
    sampler = make_sampler(config.sampling_strategy)
    steps: list[tuple[str, Any]] = [("outlier", outlier), ("preprocess", preprocessor)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def calc_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, fit_seconds: float) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "fit_seconds": fit_seconds,
    }


def run_experiment(df: pd.DataFrame, config: ExperimentConfig, cv: StratifiedKFold):
    numeric, categorical = feature_columns(config.feature_set)
    cols = numeric + categorical
    X = df[cols].copy()
    y = df[TARGET].astype(int)
    fold_rows = []
    oof_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        X_train = inject_missing_if_needed(X_train, config, fold)
        X_valid = inject_missing_if_needed(X_valid, config, fold)
        pipeline = make_pipeline(config, X_train, y_train)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        proba = pipeline.predict_proba(X_valid)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = calc_metrics(y_valid.to_numpy(), proba, pred, fit_seconds)
        fold_rows.append(
            {
                "experiment_id": config.experiment_id,
                "stage": config.stage,
                "model_name": config.model_name,
                "feature_set": config.feature_set,
                "missing_strategy": config.missing_strategy,
                "outlier_strategy": config.outlier_strategy,
                "sampling_strategy": config.sampling_strategy,
                "hyperparameter_config": config.hyperparameter_config,
                "threshold_rule": config.threshold_rule,
                "fold": fold,
                **metrics,
            }
        )
        valid_meta = df.iloc[valid_idx][ID_COLS + LEAKAGE_COLS].copy()
        valid_meta["row_index"] = valid_idx
        valid_meta["actual"] = y_valid.to_numpy()
        valid_meta["predicted_probability"] = proba
        valid_meta["predicted_label"] = pred
        valid_meta["fold"] = fold
        valid_meta["experiment_id"] = config.experiment_id
        valid_meta["model_name"] = config.model_name
        valid_meta["feature_set"] = config.feature_set
        valid_meta["missing_strategy"] = config.missing_strategy
        valid_meta["outlier_strategy"] = config.outlier_strategy
        valid_meta["sampling_strategy"] = config.sampling_strategy
        oof_rows.append(valid_meta)

    return fold_rows, oof_rows


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_negative", "false_positive", "fit_seconds"]
    config_cols = ["experiment_id", "stage", "model_name", "feature_set", "missing_strategy", "outlier_strategy", "sampling_strategy", "hyperparameter_config", "threshold_rule"]
    summary = fold_results.groupby(config_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c]) for col in summary.columns.to_flat_index()]
    summary = summary.sort_values(["f1_mean", "pr_auc_mean", "recall_mean"], ascending=False)
    return summary


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray, metric: str, n_boot: int = 700) -> tuple[float, float]:
    rng = np.random.default_rng(123)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric == "f1":
            values.append(f1_score(y_true[idx], pred[idx], zero_division=0))
        elif metric == "pr_auc":
            values.append(average_precision_score(y_true[idx], proba[idx]))
        elif metric == "recall":
            values.append(recall_score(y_true[idx], pred[idx], zero_division=0))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def statistical_tests(fold_results: pd.DataFrame, oof: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    baseline_id = "exp001"
    best_id = str(summary.iloc[0]["experiment_id"])
    rows = []
    base_folds = fold_results[fold_results["experiment_id"] == baseline_id].sort_values("fold")
    base_oof = oof[oof["experiment_id"] == baseline_id].sort_values("row_index")

    for exp_id in summary["experiment_id"]:
        exp_id = str(exp_id)
        exp_folds = fold_results[fold_results["experiment_id"] == exp_id].sort_values("fold")
        exp_oof = oof[oof["experiment_id"] == exp_id].sort_values("row_index")
        if len(exp_folds) != len(base_folds):
            continue
        f1_delta = exp_folds["f1"].to_numpy() - base_folds["f1"].to_numpy()
        pr_delta = exp_folds["pr_auc"].to_numpy() - base_folds["pr_auc"].to_numpy()
        try:
            f1_p = stats.wilcoxon(f1_delta).pvalue if np.any(f1_delta != 0) else 1.0
        except ValueError:
            f1_p = 1.0
        try:
            pr_p = stats.wilcoxon(pr_delta).pvalue if np.any(pr_delta != 0) else 1.0
        except ValueError:
            pr_p = 1.0
        y_true = exp_oof["actual"].to_numpy()
        proba = exp_oof["predicted_probability"].to_numpy()
        pred = exp_oof["predicted_label"].to_numpy()
        f1_low, f1_high = bootstrap_ci(y_true, proba, pred, "f1")
        pr_low, pr_high = bootstrap_ci(y_true, proba, pred, "pr_auc")
        rows.append(
            {
                "experiment_id": exp_id,
                "best_experiment_id": best_id,
                "baseline_experiment_id": baseline_id,
                "f1_delta_vs_baseline_mean": float(f1_delta.mean()),
                "pr_auc_delta_vs_baseline_mean": float(pr_delta.mean()),
                "wilcoxon_f1_p_value": float(f1_p),
                "wilcoxon_pr_auc_p_value": float(pr_p),
                "oof_f1_ci95_low": f1_low,
                "oof_f1_ci95_high": f1_high,
                "oof_pr_auc_ci95_low": pr_low,
                "oof_pr_auc_ci95_high": pr_high,
            }
        )
    stats_df = pd.DataFrame(rows)
    for col in ["wilcoxon_f1_p_value", "wilcoxon_pr_auc_p_value"]:
        stats_df[f"{col}_holm"] = holm_bonferroni(stats_df[col].to_numpy())
    return stats_df


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        adj = min((m - rank) * p_values[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def write_log(summary: pd.DataFrame, stats_df: pd.DataFrame, n_experiments: int) -> None:
    best = summary.iloc[0]
    lines = [
        "# Predictive Maintenance Comparative Experiment Run Log",
        "",
        "## 1. 유저가 요청한 사항",
        "- EDA 기반 전처리/모델링 전략, 모델 선택, 5-fold stratified CV + OOF 검증, 불균형 샘플링, 결측치/이상치 처리, 하이퍼파라미터 비교실험을 실제 실행.",
        "",
        "## 2. 너가 수행한 사항",
        "- 19개 실험 조합을 구성하고 동일한 5-fold stratified split으로 실행.",
        "- Logistic Regression, RandomForest, ExtraTrees, LightGBM, XGBoost를 비교.",
        "- base feature와 engineered feature를 비교.",
        "- class weight/scale_pos_weight, random over/under sampling, SMOTE를 비교.",
        "- median/mode, missing indicator, stress missing 5%를 비교.",
        "- outlier none, p01/p99 clipping, IQR clipping, outlier flag를 비교.",
        "- LightGBM/XGBoost 하이퍼파라미터 후보를 비교.",
        "- OOF prediction, fold metrics, summary, Wilcoxon/Bootstrap 통계검증 결과를 저장.",
        "",
        "## 3. 그 중 완료된 사항",
        f"- 실행 실험 수: `{n_experiments}`",
        "- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`",
        f"- Best experiment: `{best['experiment_id']}`",
        f"- Best model: `{best['model_name']}`",
        f"- Best config: feature=`{best['feature_set']}`, sampling=`{best['sampling_strategy']}`, missing=`{best['missing_strategy']}`, outlier=`{best['outlier_strategy']}`, hparam=`{best['hyperparameter_config']}`",
        f"- Mean F1: `{best['f1_mean']:.4f}`",
        f"- Mean PR-AUC: `{best['pr_auc_mean']:.4f}`",
        f"- Mean Recall: `{best['recall_mean']:.4f}`",
        f"- Fold results: `{FOLD_RESULTS_PATH.as_posix()}`",
        f"- Summary: `{SUMMARY_PATH.as_posix()}`",
        f"- OOF predictions: `{OOF_PATH.as_posix()}`",
        f"- Statistical tests: `{STATS_PATH.as_posix()}`",
        "",
        "## 4. 문제발생 현상 및 향후 해결 방안",
        "- 문제: fold가 5개라 통계검정의 검정력이 제한적임.",
        "  - 해결 방안: 최종 후보는 repeated 5-fold CV 또는 bootstrap 반복을 추가.",
        "- 문제: stress missing 실험은 원본 데이터 성능 비교가 아니라 robust성 검증용임.",
        "  - 해결 방안: 최종 모델 선정에서는 원본 데이터 실험과 분리해서 해석.",
        "- 문제: sampling은 train fold 내부에서만 적용해야 함.",
        "  - 해결 방안: imbalanced-learn Pipeline으로 fold 내부 sampling을 적용하여 누수를 방지.",
    ]
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    configs = build_configs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows = []
    oof_rows = []
    for config in configs:
        print(f"Running {config.experiment_id}: {config.model_name} / {config.stage}")
        exp_fold_rows, exp_oof_rows = run_experiment(df, config, cv)
        fold_rows.extend(exp_fold_rows)
        oof_rows.extend(exp_oof_rows)

    fold_results = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    summary = summarize_results(fold_results)
    stats_df = statistical_tests(fold_results, oof, summary)
    fold_results.to_csv(FOLD_RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    oof.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")
    stats_df.to_csv(STATS_PATH, index=False, encoding="utf-8-sig")
    CONFIG_PATH.write_text(json.dumps([config.__dict__ for config in configs], indent=2), encoding="utf-8")
    write_log(summary, stats_df, len(configs))
    print("Best experiment")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
