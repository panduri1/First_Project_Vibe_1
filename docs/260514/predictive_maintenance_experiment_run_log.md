# Predictive Maintenance Comparative Experiment Run Log

## 1. 유저가 요청한 사항
- EDA 기반 전처리/모델링 전략, 모델 선택, 5-fold stratified CV + OOF 검증, 불균형 샘플링, 결측치/이상치 처리, 하이퍼파라미터 비교실험을 실제 실행.

## 2. 너가 수행한 사항
- 19개 실험 조합을 구성하고 동일한 5-fold stratified split으로 실행.
- Logistic Regression, RandomForest, ExtraTrees, LightGBM, XGBoost를 비교.
- base feature와 engineered feature를 비교.
- class weight/scale_pos_weight, random over/under sampling, SMOTE를 비교.
- median/mode, missing indicator, stress missing 5%를 비교.
- outlier none, p01/p99 clipping, IQR clipping, outlier flag를 비교.
- LightGBM/XGBoost 하이퍼파라미터 후보를 비교.
- OOF prediction, fold metrics, summary, Wilcoxon/Bootstrap 통계검증 결과를 저장.

## 3. 그 중 완료된 사항
- 실행 실험 수: `19`
- CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Best experiment: `exp017`
- Best model: `LightGBM`
- Best config: feature=`all_engineered`, sampling=`scale_pos_weight`, missing=`median_mode`, outlier=`none`, hparam=`tuned_deeper`
- Mean F1: `0.8181`
- Mean PR-AUC: `0.8594`
- Mean Recall: `0.7906`
- Fold results: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/outputs/experiments/predictive_maintenance/experiment_fold_results.csv`
- Summary: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/outputs/experiments/predictive_maintenance/experiment_summary.csv`
- OOF predictions: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/outputs/experiments/predictive_maintenance/oof_predictions.csv`
- Statistical tests: `C:/Users/USER/Desktop/lg_vibe_ml/NEW Project 1/outputs/experiments/predictive_maintenance/experiment_stat_tests.csv`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: fold가 5개라 통계검정의 검정력이 제한적임.
  - 해결 방안: 최종 후보는 repeated 5-fold CV 또는 bootstrap 반복을 추가.
- 문제: stress missing 실험은 원본 데이터 성능 비교가 아니라 robust성 검증용임.
  - 해결 방안: 최종 모델 선정에서는 원본 데이터 실험과 분리해서 해석.
- 문제: sampling은 train fold 내부에서만 적용해야 함.
  - 해결 방안: imbalanced-learn Pipeline으로 fold 내부 sampling을 적용하여 누수를 방지.
