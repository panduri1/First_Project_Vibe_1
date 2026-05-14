# Predictive Maintenance Experiment Plan

## 1. 유저가 요청한 사항
- EDA 기반으로 전처리와 모델링 전략을 더 면밀하게 설계.
- LightGBM, XGBoost 및 기타 모델의 모델 선택/하이퍼파라미터 실험 계획 강화.
- 5-fold stratified CV와 out-of-fold 검증 적용.
- 불균형 데이터 샘플링, 결측치/이상치 처리 기준을 여러 세팅으로 비교.
- 베이스라인부터 점진적으로 강화되는 실험 흐름과 통계적 유의미성 검증 계획 수립.

## 2. 너가 수행한 사항
- 현재 데이터 특성, 기존 LightGBM 베이스라인 결과, 불균형 구조를 바탕으로 실험 설계를 재구성.
- 전처리, 모델, 샘플링, 이상치 처리, threshold tuning, 통계검증을 분리한 실험 프레임워크를 정의.
- 실험 결과를 누적 비교할 수 있는 산출물 구조와 판단 기준을 설계.

## 3. 그 중 완료된 사항
- 강화 실험 계획 문서 작성 완료.
- 5-fold stratified CV + out-of-fold 기반 검증 전략 정의 완료.
- 불균형/결측치/이상치/하이퍼파라미터/통계검증 실험 매트릭스 정의 완료.

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 현재 데이터는 고장 비율이 약 3.39%로 낮아 accuracy 중심 평가는 부적절함.
  - 해결 방안: PR-AUC, recall, F1, balanced accuracy, false negative count를 주요 지표로 사용.
- 문제: `Failure Type`은 `Target`과 직접 연결된 결과성 컬럼이라 binary 예측 feature로 쓰면 데이터 누수가 발생함.
  - 해결 방안: 1차 binary 모델에서는 제외하고, 별도 2차 고장 유형 분석에서만 사용.
- 문제: `Target`과 `Failure Type` 불일치 27건이 있음.
  - 해결 방안: binary 모델은 `Target` 기준으로 유지하되, 고장 유형 분석에서는 세 가지 정책(제외/재라벨링/그대로 유지)을 비교.

## 실험 목표
최종 목표는 단순히 가장 높은 점수를 내는 모델을 찾는 것이 아니라, 다음 질문에 답하는 것입니다.

- 어떤 전처리 전략이 고장 예측 성능을 안정적으로 개선하는가?
- 불균형 데이터에서 recall을 높이면서 false alarm을 어느 정도로 제어할 수 있는가?
- LightGBM/XGBoost 성능 차이가 우연이 아니라 통계적으로 의미 있는가?
- 모델이 어떤 센서/운전 조건을 근거로 고장 위험을 판단하는가?
- 실무 대시보드에서 어떤 threshold를 추천할 수 있는가?

## 데이터 분석 기반 가설
### 데이터 특성
- 전체 10,000건 중 `Target=1`은 339건으로 고장 비율은 약 3.39%.
- `Failure Type`과 `Target` 간 불일치 27건 존재.
- `Type`은 L/M/H 세 가지 제품 타입으로 구성.
- 센서 feature는 모두 수치형이고, 결측치는 현재 원본 기준 0건.

### 모델링 가설
- `Tool wear [min]`, `Torque [Nm]`, `Rotational speed [rpm]`, `temperature_gap`, `power_proxy`가 주요 예측 신호일 가능성이 높음.
- 고장 클래스가 적기 때문에 recall을 높이면 precision이 하락할 가능성이 큼.
- 단일 train/test split보다 stratified CV의 fold별 편차를 확인해야 모델 안정성을 판단할 수 있음.
- tree boosting 계열은 nonlinear interaction을 잘 잡아 Logistic Regression보다 성능이 높을 가능성이 큼.

## 전체 실험 흐름
### Stage 0. Data Audit
- 원본 데이터 shape, dtype, 결측치, 중복 확인.
- `Target` 분포, `Failure Type` 분포, `Type`별 고장률 확인.
- `Target`/`Failure Type` 불일치 27건 별도 저장.
- 수치형 변수별 분포, skewness, outlier rate 확인.

산출물:
- `data_audit_summary.csv`
- `target_failure_type_mismatch.csv`
- `eda_profile.md`

### Stage 1. Baseline
목표: 아무 강화 없이 시작점 성능을 명확히 기록.

실험 세팅:
- Features: `Type`, 온도, 회전속도, 토크, 공구마모
- Exclude: `UDI`, `Product ID`, `Failure Type`
- Model:
  - Logistic Regression
  - RandomForest
  - LightGBM default-ish
  - XGBoost default-ish
- Validation: 5-fold stratified CV
- Metrics:
  - PR-AUC
  - ROC-AUC
  - F1
  - Recall
  - Precision
  - Balanced Accuracy
  - False Negative count

### Stage 2. Feature Engineering
비교할 feature set:

| Feature Set | 설명 |
|---|---|
| `base` | 원본 센서 + `Type` |
| `temp_gap` | `base` + `Process temperature - Air temperature` |
| `power_proxy` | `base` + `Rotational speed * Torque` |
| `wear_ratio` | `base` + `Tool wear / Torque` |
| `all_engineered` | 위 파생변수 전체 |
| `interaction_bins` | 주요 수치형 변수를 분위수 binning 후 interaction 추가 |

판단 기준:
- 평균 성능뿐 아니라 fold별 표준편차가 줄어드는지 확인.
- feature 추가가 특정 fold에서만 좋아지는지, 전체적으로 안정적인지 확인.

### Stage 3. Missing Value Strategy
현재 원본 결측치는 없지만, 실무 데이터 적용 가능성을 위해 결측치 처리 전략을 실험 구조에 포함.

비교 세팅:

| Strategy | 설명 |
|---|---|
| `none` | 결측치 없음. 현재 데이터 기준 baseline |
| `median_mode` | 수치형 median, 범주형 mode |
| `knn_imputer` | KNN imputation |
| `missing_indicator` | 결측 여부 indicator 추가 |
| `stress_missing_5pct` | 임의 5% 결측 주입 후 robust성 비교 |

주의:
- 결측치 주입 실험은 원본 성능 비교가 아니라 모델 robust성 검증용으로 분리.

### Stage 4. Outlier Strategy
수치형 sensor 데이터는 극단값이 실제 고장 신호일 수 있으므로 무조건 제거하면 안 됩니다.

비교 세팅:

| Strategy | 설명 |
|---|---|
| `none` | 이상치 처리 없음 |
| `iqr_clip` | IQR 기준 lower/upper clipping |
| `p01_p99_clip` | 1%/99% winsorization |
| `robust_scaler` | scaling만 robust하게 처리 |
| `outlier_flags` | clipping 없이 이상치 flag 추가 |

판단 기준:
- recall이 떨어지면 위험 신호를 제거했을 가능성이 있으므로 주의.
- 이상치 처리 후 precision만 좋아지고 recall이 떨어지는 경우 실무 목적과 맞는지 검토.

### Stage 5. Imbalanced Learning
비교할 불균형 대응 전략:

| Strategy | 설명 |
|---|---|
| `none` | 별도 처리 없음 |
| `class_weight` | class weight balanced |
| `scale_pos_weight` | XGBoost/LightGBM용 positive class weight |
| `random_undersampling` | 정상 class 일부 downsampling |
| `random_oversampling` | 고장 class oversampling |
| `smote` | SMOTE synthetic oversampling |
| `threshold_tuning` | 모델은 그대로 두고 threshold 조정 |

권장 패키지:
- `imbalanced-learn`

주의:
- oversampling/SMOTE는 반드시 각 CV fold의 train split 안에서만 적용.
- validation fold나 OOF prediction에 sampling이 섞이면 데이터 누수 발생.

### Stage 6. Model Selection
비교 모델:

| Model | 역할 |
|---|---|
| Logistic Regression | 해석 가능한 baseline |
| RandomForest | non-linear baseline |
| ExtraTrees | tree ensemble baseline |
| XGBoost | gradient boosting 후보 |
| LightGBM | gradient boosting 후보 |
| HistGradientBoosting | sklearn native boosting 후보 |

1차 판단 기준:
- PR-AUC 평균
- recall@selected_precision
- F1
- fold 안정성
- inference 속도
- feature importance/해석 용이성

## Hyperparameter Search Plan
### LightGBM
주요 탐색 범위:

| Parameter | Candidate |
|---|---|
| `n_estimators` | 100, 300, 600, 1000 |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 |
| `num_leaves` | 7, 15, 31, 63 |
| `max_depth` | -1, 3, 5, 7 |
| `min_child_samples` | 10, 20, 50, 100 |
| `subsample` | 0.7, 0.85, 1.0 |
| `colsample_bytree` | 0.7, 0.85, 1.0 |
| `reg_alpha` | 0, 0.1, 1 |
| `reg_lambda` | 0, 0.1, 1, 5 |
| `class_weight` | None, balanced |

### XGBoost
주요 탐색 범위:

| Parameter | Candidate |
|---|---|
| `n_estimators` | 100, 300, 600, 1000 |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 |
| `max_depth` | 2, 3, 4, 6 |
| `min_child_weight` | 1, 3, 5, 10 |
| `subsample` | 0.7, 0.85, 1.0 |
| `colsample_bytree` | 0.7, 0.85, 1.0 |
| `gamma` | 0, 0.1, 1 |
| `reg_alpha` | 0, 0.1, 1 |
| `reg_lambda` | 1, 3, 5 |
| `scale_pos_weight` | 1, neg/pos, sqrt(neg/pos) |

### Search Strategy
권장 순서:

1. 넓은 RandomizedSearch
2. 상위 10개 조합 재평가
3. 좁은 grid search
4. threshold tuning
5. 최종 nested CV 또는 repeated CV 검증

## 5-Fold Stratified CV + OOF 검증
### 기본 원칙
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 각 fold마다 train/validation 분리.
- preprocessing, imputation, scaling, sampling은 train fold 안에서 fit.
- validation fold prediction을 모아 OOF prediction 생성.
- 최종 metric은 OOF prediction 전체 기준과 fold별 평균을 모두 기록.

### OOF 산출물
- `experiment_id`
- `fold`
- `row_id`
- `actual`
- `predicted_probability`
- `predicted_label_0_5`
- `threshold_selected_label`
- `model_name`
- `preprocess_config`
- `sampling_config`
- `hyperparameter_config`

## Threshold Tuning
고장 예측에서는 기본 threshold 0.5가 최선이 아닐 가능성이 큽니다.

비교 기준:

| Threshold Rule | 설명 |
|---|---|
| `0.5` | 기본값 |
| `max_f1` | OOF 기준 F1 최대 |
| `recall_90` | recall 90% 이상 중 precision 최대 |
| `precision_50` | precision 50% 이상 중 recall 최대 |
| `cost_based` | FN 비용을 FP보다 크게 둔 비용 최소화 |

비용 기반 예시:
- `Cost = 10 * FN + 1 * FP`
- 현업에서 고장 미탐 비용이 크다면 FN 비용을 더 높임.

## Statistical Validation
### Fold-Level Paired Test
각 실험은 같은 fold split을 사용하므로 paired comparison을 적용합니다.

검정 후보:
- Paired t-test: fold별 metric 차이가 정규성에 크게 어긋나지 않을 때
- Wilcoxon signed-rank test: fold 수가 작고 분포 가정이 부담될 때
- Bootstrap confidence interval: OOF prediction 단위로 metric CI 추정

권장:
- fold가 5개뿐이라 paired t-test 단독 의존은 피하고 Wilcoxon + bootstrap CI를 함께 보고.

### OOF Prediction-Level Test
분류 예측 차이를 비교할 때:
- McNemar test: 두 모델의 맞음/틀림 차이 비교
- DeLong test: ROC-AUC 비교
- Bootstrap PR-AUC/F1 CI: 불균형 데이터에서 PR-AUC 중심 검증

### Multiple Comparison 보정
실험 조합이 많으면 우연히 좋아 보이는 모델이 생길 수 있음.

보정 방법:
- Holm-Bonferroni
- Benjamini-Hochberg FDR

### 보고 기준
각 주요 후보는 아래를 함께 기록:

- mean metric
- std metric
- 95% confidence interval
- baseline 대비 평균 개선폭
- p-value
- adjusted p-value
- effect size

## 실험 매트릭스 우선순위
모든 조합을 전부 돌리면 실험 수가 과도하게 커지므로 단계적으로 확장합니다.

### Round 1. 빠른 기준선
- Feature set: `base`, `all_engineered`
- Models: Logistic Regression, RandomForest, LightGBM, XGBoost
- Sampling: none, class_weight/scale_pos_weight
- CV: 5-fold

### Round 2. 전처리/샘플링 확장
- Best 2 models만 유지
- Outlier: none, p01_p99_clip, outlier_flags
- Sampling: class_weight, random_oversampling, SMOTE, threshold_tuning

### Round 3. 하이퍼파라미터 탐색
- LightGBM/XGBoost RandomizedSearch
- OOF metric 기준 top N 선정
- 통계검증으로 baseline 대비 유의성 확인

### Round 4. 최종 모델 검증
- 고정된 best config로 repeated 5-fold CV 또는 seed 3개 반복.
- 최종 threshold 선택.
- 오분류 케이스 분석.
- Streamlit 대시보드에 최종 모델 반영.

## 결과 테이블 스키마
`experiment_results.csv`

| Column | 설명 |
|---|---|
| `experiment_id` | 실험 ID |
| `stage` | baseline/feature/sampling/hparam/final |
| `model_name` | 모델명 |
| `feature_set` | feature 구성 |
| `missing_strategy` | 결측치 처리 |
| `outlier_strategy` | 이상치 처리 |
| `sampling_strategy` | 불균형 처리 |
| `threshold_rule` | threshold 선택 기준 |
| `fold` | fold 번호 |
| `accuracy` | accuracy |
| `precision` | precision |
| `recall` | recall |
| `f1` | F1 |
| `balanced_accuracy` | balanced accuracy |
| `roc_auc` | ROC-AUC |
| `pr_auc` | PR-AUC |
| `false_negative` | FN count |
| `false_positive` | FP count |
| `fit_seconds` | 학습 시간 |

`experiment_summary.csv`

| Column | 설명 |
|---|---|
| `experiment_id` | 실험 ID |
| `mean_pr_auc` | fold 평균 PR-AUC |
| `std_pr_auc` | fold 표준편차 |
| `mean_f1` | fold 평균 F1 |
| `mean_recall` | fold 평균 recall |
| `ci95_low` | bootstrap 95% CI low |
| `ci95_high` | bootstrap 95% CI high |
| `baseline_delta` | baseline 대비 개선 |
| `p_value` | 검정 p-value |
| `adjusted_p_value` | 다중비교 보정 p-value |

## 최종 선택 기준
1. PR-AUC와 F1이 baseline보다 유의미하게 개선될 것.
2. Recall이 충분히 높고, false negative가 낮을 것.
3. Fold별 성능 편차가 과도하지 않을 것.
4. `Failure Type` 누수 없이 성능이 나올 것.
5. Streamlit에서 threshold 조정과 오분류 분석이 설명 가능할 것.

## 다음 구현 단계
1. `scripts/run_predictive_maintenance_experiments.py` 생성.
2. 공통 CV runner 구현.
3. preprocessing/sampling/model config를 dictionary로 관리.
4. OOF prediction과 fold metrics 저장.
5. 통계검증 모듈 구현.
6. HTML/Streamlit에 실험 비교 탭 추가.
