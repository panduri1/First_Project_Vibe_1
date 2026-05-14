# Predictive Maintenance Execution Log

## 1. 유저가 요청한 사항
- 다운로드한 Kaggle 예지보전 데이터로 EDA/전처리, ML 모델링, 성능평가 대시보드 생성을 실행.

## 2. 너가 수행한 사항
- Python 가상환경 `.venv` 생성 및 분석 패키지 설치.
- CSV 로드, 파생변수 생성, 전처리 데이터 저장.
- SQLite raw/processed 테이블 생성.
- EDA 요약 CSV와 차트 생성.
- Logistic Regression, RandomForest, XGBoost, LightGBM 모델 학습 및 평가.
- HTML 리포트와 Streamlit 앱 생성.
- Streamlit 대시보드를 로컬 포트 `8501`에서 실행.

## 3. 그 중 완료된 사항
- 입력 데이터: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/data/kaggle/machine_predictive_maintenance_classification/predictive_maintenance.csv`
- 행/컬럼 수: `10,000` rows, `13` columns
- 결측치: before `0`, after `0`
- 중복 행: `0`
- `Target`/`Failure Type` 불일치: `27`
- Best model: `LightGBM`
- Best F1: `0.8321`
- Best recall: `0.8382`
- EDA/model outputs: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/outputs/data_gathering/predictive_maintenance`
- HTML report: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/reports/data_gathering_predictive_maintenance_report.html`
- Streamlit app: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/reports/data_gathering_predictive_maintenance_streamlit_app.py`
- Streamlit URL: `http://localhost:8501`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 고장 클래스 비율이 낮아 accuracy만으로는 모델 성능을 판단하기 어려움.
  - 해결 방안: recall, F1, PR-AUC, threshold tuning 중심으로 평가.
- 문제: `Failure Type`은 `Target`과 연결된 결과 컬럼이라 binary 예측 feature로 쓰면 데이터 누수가 발생함.
  - 해결 방안: 1차 모델 feature에서 제외하고, 2차 고장 유형 분석 타깃으로만 사용.
- 문제: `Target`과 `Failure Type` 불일치 27건이 있음.
  - 해결 방안: 이후 고장 유형 모델링 전에 제외/재라벨링/별도 분석 중 하나로 정책 결정 필요.
