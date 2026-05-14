# Streamlit Dashboard Enhancement Log

## 1. 유저가 요청한 사항
- Streamlit 대시보드 앱을 3개 탭 구조로 구현.
- 탭1/탭2/탭3 버튼을 파란색 버튼 스타일로 변경.
- `Machine Predictive Maintenance Dashboard` 타이틀을 파란색 배경으로 변경하고 양쪽에 AX 로봇 아이콘 추가.
- 탭1 실험 결과:
  - 지금까지의 모든 실험 결과를 통계검증과 함께 실험 테이블로 정리.
  - 시도한 방법론과 효과 해석.
  - 향후 추가 실험 방안 제시.
- 탭2 입력 변수 분석:
  - Feature Importance 기반 중요/비중요 변수 비교.
  - 변수별 통계적 차이와 심화 EDA, 상관관계 힌트 정리.
- 탭3 오분류 데이터 심화 분석:
  - 최고 성능 모델의 오분류 케이스 샘플링 표시.
  - threshold 근처 오분류와 high-margin 오분류 비교.
  - 라벨링 오류 가능성 조사.
  - 모델 고도화 아이디어 제시.

## 2. 너가 수행한 사항
- `reports/data_gathering_predictive_maintenance_streamlit_app.py`를 3개 탭 구조로 재작성.
- 기존 `model_metrics.csv`, `model_predictions.csv`, `feature_importance.csv`, `processed_predictive_maintenance.csv`, `best_model.joblib`을 활용하도록 구현.
- Bootstrap CI, McNemar test, paired bootstrap delta를 앱에서 계산하도록 구현.
- Mann-Whitney U test, Cohen's d 기반 입력 변수 분석을 구현.
- 최고 모델의 전체 데이터 scoring, validation 오분류 샘플링, label issue candidate 탐색을 구현.
- 앱 문법 컴파일과 실행 중인 Streamlit 서버 상태를 확인.
- CSS를 추가하여 탭 버튼, 선택된 탭, hover 상태를 파란색 버튼 스타일로 변경.
- 파란색 타이틀 배너와 좌우 `AX ROBOT` 아이콘을 추가.
- 실제 실행된 5-fold stratified CV 비교실험 결과(`experiment_summary.csv`, `experiment_stat_tests.csv`)를 탭1에 연동.

## 3. 그 중 완료된 사항
- 탭1 `실험 결과` 구현 완료.
- 탭2 `입력 변수 분석` 구현 완료.
- 탭3 `오분류 데이터 심화 분석` 구현 완료.
- 파란색 탭 버튼 스타일 적용 완료.
- 파란색 타이틀 배너 및 좌우 AX 로봇 아이콘 적용 완료.
- 탭1에 19개 실험 조합의 5-fold CV 결과, precision/recall trade-off chart, OOF 기반 통계검증 테이블 표시 완료.
- Streamlit 앱 컴파일 성공.
- 기존 Streamlit 서버 `http://localhost:8501`에서 앱 자동 갱신 가능 상태 확인.

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 초기에는 통계검증이 holdout prediction 기반이었음.
  - 해결 완료: `run_predictive_maintenance_experiments.py`를 구현해 5-fold OOF 실험 결과로 탭1을 갱신.
- 문제: validation prediction에는 원본 `UDI`, `Product ID`, `Failure Type`이 포함되어 있지 않아 검증셋 오분류의 원본 ID 추적이 제한적임.
  - 해결 방안: 다음 실험 스크립트에서 OOF prediction 저장 시 row index와 원본 ID/라벨 정보를 함께 저장.
- 문제: Streamlit 로그에 `use_container_width` deprecation warning이 표시됨.
  - 해결 방안: 동작에는 영향 없으나 후속 정리 시 `width="stretch"` 방식으로 교체.
