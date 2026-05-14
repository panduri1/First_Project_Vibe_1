# Machine Predictive Maintenance Project Plan

## 1. 유저가 요청한 사항
- 후보 4번 `shivamb/machine-predictive-maintenance-classification` 데이터셋으로 프로젝트 진행.
- `.env`의 `KAGGLE_API_TOKEN`을 활용해 Kaggle API로 데이터 다운로드.
- 이후 EDA/전처리, ML 모델링, Streamlit 성능평가 대시보드, 오분류 케이스 분석 및 모델 고도화 방향을 계획.
- 다운로드한 데이터를 Markdown 파일로 열람할 수 있게 정리.

## 2. 너가 수행한 사항
- Kaggle 다운로드 스크립트 2종을 생성.
  - `scripts/download_kaggle_dataset.py`
  - `scripts/download_kaggle_dataset.ps1`
- Python이 PATH에 없어 PowerShell 기반 다운로드 방식으로 전환.
- 프로젝트 루트 및 상위 주요 경로에서 `.env` 존재 여부를 확인.
- 선택 데이터셋 기준의 분석/모델링/대시보드 프로젝트 계획을 수립.
- 사용자가 제공한 PowerShell 다운로드 명령을 실행하고 결과 로그를 확인.
- `.env` 생성 후 재시도 요청에 따라 다운로드 명령을 다시 실행하고, 파일 존재 여부와 변수명 감지 여부를 확인.
- `.env` 저장 후 추가 재시도했으며, `KAGGLE_API_TOKEN` 값은 감지되지만 Kaggle 인증에 필요한 username/key 쌍으로는 해석되지 않는 것을 확인.
- `.env`의 `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` 조합을 Kaggle key로 사용할 수 있도록 다운로드 스크립트를 보강하고 재실행.
- 다운로드된 CSV의 행 수와 컬럼명을 확인.
- 남아 있던 문제/추가 문제 항목이 정상 동작을 방해하지 않도록 다운로드 스크립트와 문서 상태를 정리.
- CSV의 샘플 행, 타깃 분포, 고장 유형 분포, 제품 타입 분포, `Target`/`Failure Type` 불일치 건수를 확인.
- 데이터 열람용 Markdown 파일 `docs/260514/machine_predictive_maintenance_dataset.md`를 생성.

## 3. 그 중 완료된 사항
- 선택 데이터셋 확정: `shivamb/machine-predictive-maintenance-classification`
- 데이터 저장 예정 경로 확정:
  - `data/kaggle/machine_predictive_maintenance_classification/`
- 다운로드 자동화 스크립트 생성 완료.
- 프로젝트 진행 계획 수립 완료.
- Kaggle 데이터 다운로드 완료:
  - `data/kaggle/machine_predictive_maintenance_classification/predictive_maintenance.csv`
  - 행 수: 10,000
  - 컬럼 수: 10
- PowerShell 스크립트가 현재 `.env` 형식인 `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` 조합에서도 정상 동작하도록 수정 완료.
- 다운로드 후 임시 `dataset.zip`을 제거하도록 수정 완료.
- 로그 문구를 ASCII 중심으로 통일하여 PowerShell 인코딩 깨짐 재발 가능성을 낮춤.
- 데이터셋 MD 문서 생성 완료:
  - `docs/260514/machine_predictive_maintenance_dataset.md`
- 실행 완료:
  - EDA/model outputs: `outputs/data_gathering/predictive_maintenance/`
  - HTML report: `reports/data_gathering_predictive_maintenance_report.html`
  - Streamlit app: `reports/data_gathering_predictive_maintenance_streamlit_app.py`
  - Streamlit URL: `http://localhost:8501`
  - Best model: `LightGBM`
  - Best F1: `0.8321`
  - Best recall: `0.8382`
- 강화 실험 계획 작성 완료:
  - `docs/260514/predictive_maintenance_experiment_plan.md`
  - 5-fold stratified CV, OOF 검증, 불균형 샘플링, 결측치/이상치 처리 실험, hyperparameter search, 통계검증 계획 포함.
- 강화 비교실험 실제 실행 완료:
  - 실행 스크립트: `scripts/run_predictive_maintenance_experiments.py`
  - 실행 로그: `docs/260514/predictive_maintenance_experiment_run_log.md`
  - 19개 실험 조합, 5-fold stratified CV, OOF prediction 생성.
  - 모델 비교: Logistic Regression, RandomForest, ExtraTrees, LightGBM, XGBoost.
  - 전처리 비교: base/all engineered feature, 결측치 median/missing indicator/stress missing 5%, 이상치 none/p01-p99/IQR/outlier flags.
  - 불균형 비교: class weight/scale_pos_weight, random over/under sampling, SMOTE.
  - hyperparameter 비교: LightGBM/XGBoost tuned 후보.
  - Best CV experiment: `exp017` LightGBM tuned deeper, F1 mean `0.8181`, PR-AUC mean `0.8594`, recall mean `0.7906`.
  - 실험 결과: `outputs/experiments/predictive_maintenance/experiment_summary.csv`
  - OOF 예측: `outputs/experiments/predictive_maintenance/oof_predictions.csv`
  - 통계검증: `outputs/experiments/predictive_maintenance/experiment_stat_tests.csv`
- Streamlit 대시보드 고도화 완료:
  - 탭1: 실험 결과, bootstrap CI, McNemar test, paired bootstrap delta, 방법론 해석, 향후 실험 방안.
  - 탭2: feature importance, Mann-Whitney U test, Cohen's d, 변수별 분포/상관관계 분석.
  - 탭3: validation 오분류 샘플, threshold 근처/고확신 오분류 비교, label issue candidate, 고도화 아이디어.
  - UI: 탭 버튼 파란색 스타일, 파란색 타이틀 배너, 좌우 AX 로봇 아이콘 적용.
  - 탭1에 5-fold CV 실험 결과/OOF 통계검증 결과 연동 완료.
  - 작업 기록: `docs/260514/streamlit_dashboard_enhancement_log.md`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 초기에는 `.env` 부재/빈 파일/토큰 형식 불일치로 Kaggle 인증이 실패했음.
  - 조치: `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` 조합을 지원하도록 스크립트를 수정했고, 현재 다운로드 성공 확인.
- 추가 문제: `python`, `py` 명령이 PATH에 등록되어 있지 않아 Python 스크립트 직접 실행이 불가했음.
  - 조치: PowerShell 스크립트만으로 다운로드가 가능하도록 구현 및 검증. Python 설치/PATH 문제는 다운로드 경로에서는 더 이상 차단 요인이 아님.
- 추가 문제: PowerShell 로그의 한글 인코딩 깨짐이 발생했음.
  - 조치: 다운로드 스크립트 로그 문구를 ASCII 중심으로 통일.
- 추가 문제: 다운로드 후 임시 zip 파일이 남을 수 있었음.
  - 조치: 압축 해제 후 `dataset.zip`을 삭제하도록 수정.
- 향후 과제: 다운로드된 데이터를 기반으로 EDA/전처리, 모델링, Streamlit 대시보드 생성을 진행.
- 추가 확인 사항: `Target`과 `Failure Type` 간 불일치 27건이 있으므로, 모델링 전 처리 정책을 정해야 함.
- 추가 계획: 최종 후보에 대해 repeated 5-fold CV와 cost-based threshold tuning을 추가해 운영 threshold를 확정.

## 데이터셋 개요
- Kaggle ref: `shivamb/machine-predictive-maintenance-classification`
- 목적: 센서/운전 조건을 기반으로 설비 고장 여부와 고장 유형을 예측.
- 예상 주요 컬럼:
  - 제품/장비 정보: `UDI`, `Product ID`, `Type`
  - 공정/설비 센서: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
  - 타깃: `Target`
  - 고장 유형: `Failure Type`

## 핵심 주의점
- `Failure Type`은 `Target`과 직접 연결되는 결과 컬럼이므로, binary failure 예측 feature에 포함하면 데이터 누수가 발생함.
- 성취감 있는 프로젝트를 위해 1차 모델에서는 `Failure Type`, `UDI`, `Product ID`를 제외하고 센서/운전 조건과 `Type`만 사용.
- 2차 분석에서는 `Target=1`인 고장 샘플을 대상으로 `Failure Type`을 별도 타깃으로 활용.

## 프로젝트 계획

### 1. EDA 및 전처리
- 데이터 구조, 결측치, 중복, 컬럼 타입 확인.
- `Target` 클래스 비율과 `Failure Type` 분포 확인.
- 온도 차이, 회전속도, 토크, 공구마모의 분포 및 이상치 분석.
- 파생변수 후보:
  - `temperature_gap = Process temperature - Air temperature`
  - `power_proxy = Rotational speed * Torque`
  - `wear_per_torque = Tool wear / Torque`
- `Product ID`, `UDI`는 식별자 성격이 강하므로 기본 모델 feature에서 제외.

### 2. ML 모델링
- 1단계: `Target` binary classification.
- 2단계: 고장 샘플만 대상으로 `Failure Type` multiclass classification.
- baseline:
  - Logistic Regression
  - RandomForest
  - XGBoost
  - LightGBM
- 불균형 대응:
  - stratified split
  - class weight
  - threshold tuning
  - PR-AUC, recall, F1 중심 평가

### 3. 성능평가 Streamlit 대시보드
- KPI 카드:
  - 전체 행 수, 고장률, best model, validation F1/recall/PR-AUC
- 시각화:
  - 타깃 분포
  - 센서별 분포/boxplot
  - correlation heatmap
  - confusion matrix
  - ROC/PR curve
  - threshold별 precision/recall trade-off
  - feature importance
- 운영 관점 기능:
  - 사용자가 센서값을 입력하면 고장 확률 예측
  - 임계값 조정 시 false alarm/missed failure 변화 확인

### 4. 오분류 케이스 분석 및 고도화
- False Negative: 실제 고장인데 정상으로 예측한 위험 케이스를 최우선 분석.
- False Positive: 정상인데 고장으로 예측한 과잉 정비 케이스 분석.
- 오분류 케이스별 센서 패턴, `Type`, 공구마모 구간, 온도 차이를 비교.
- 개선 방향:
  - threshold를 업무 비용 기준으로 조정
  - 고장 유형별 별도 모델 구성
  - feature engineering 강화
  - class imbalance 대응 강화
  - 데이터 누수 컬럼 포함/제외 실험을 명확히 분리
