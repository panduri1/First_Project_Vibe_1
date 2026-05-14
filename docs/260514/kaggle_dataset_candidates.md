# Kaggle Dataset Candidate Search

## 1. 유저가 요청한 사항
- `.env` 환경변수의 Kaggle API 토큰을 활용하여 제조업/생산관리/설비/공정/스마트팩토리 중심의 적합한 데이터셋 후보 5개를 찾고 제안.
- 조건: 명확한 분류/회귀 라벨, 다운로드 1,000건 이상, 불명확한 라벨 지양, 적당한 난이도, 3GB 미만 이미지 분류 데이터 허용.
- 선택한 데이터는 이후 Kaggle API로 다운로드하고 프로젝트 계획 수립.

## 2. 너가 수행한 사항
- Kaggle API 공개 검색 엔드포인트와 웹 검색을 통해 후보 데이터셋의 분야, 태스크, 다운로드 수, 용량, 난이도 적합성을 검토.
- 제조 tabular, 설비 센서, 품질검사 이미지 후보를 비교.

## 3. 그 중 완료된 사항
- 다운로드 1,000건 이상 조건을 만족하는 후보 5개를 선별.
- 각 후보별 활용 문제, 모델 활용 가치, 기술적 어려움을 정리.

### 후보 데이터셋 5개

| 우선순위 | Kaggle ref | 데이터 유형 | 태스크 | 다운로드 | 용량 | 추천도 |
|---|---|---|---|---:|---:|---|
| 1 | `ravirajsinh45/real-life-industrial-dataset-of-casting-product` | 주조 제품 이미지 | 양품/불량 이미지 분류 | 26,470 | 약 100 MB | 매우 높음 |
| 2 | `uciml/faulty-steel-plates` | 철강 결함 tabular | 7종 결함 다중분류 | 9,109 | 약 104 KB | 높음 |
| 3 | `brjapon/cwru-bearing-datasets` | 베어링 진동/센서 | 베어링 상태/고장 분류 | 13,974 | 약 42 MB | 높음 |
| 4 | `shivamb/machine-predictive-maintenance-classification` | 설비 센서 tabular | 고장 여부 및 고장 유형 분류 | 48,807 | 약 140 KB | 중상 |
| 5 | `angelolmg/tilda-400-64x64-patches` | 섬유 표면 이미지 패치 | 결함 유형 이미지 분류 | 2,366 | 약 149 MB | 중상 |

### 후보별 요약

1. `casting product image data for quality inspection`
   - 주조 제품 사진을 보고 양품/불량을 판정하는 품질검사 AI를 만들 수 있음.
   - Streamlit에서는 업로드 이미지 예측, 불량 확률, 오분류 이미지 갤러리, Grad-CAM류 시각화를 보여주기 좋음.
   - 단순 CNN으로도 시작 가능하지만, 조명/회전/배경 차이와 데이터 증강 전략에 따라 성능이 달라져 프로젝트 난이도가 적절함.

2. `Faulty Steel Plates`
   - 철판 표면 결함의 위치/면적/밝기 등 수치 특징으로 결함 종류를 분류하는 문제.
   - 데이터가 작고 클래스 불균형이 있어 단순 정확도보다 macro F1, class별 recall, 오분류 매트릭스 분석이 중요함.
   - 제조 tabular EDA, 전처리, 모델 비교, 오류 분석 흐름을 가장 깔끔하게 보여줄 수 있음.

3. `CWRU Bearing Dataset`
   - 모터 베어링 진동 신호를 이용해 정상/결함 상태를 분류하는 설비 진단 문제.
   - 원신호를 통계 특징으로 바꾸는 방식과 spectrogram/CNN 방식 모두 가능해 고도화 여지가 큼.
   - 시계열 분할, 누수 방지, 샘플링 윈도우 설계가 핵심 난점.

4. `Machine Predictive Maintenance Classification`
   - 온도, 회전속도, 토크, 공구마모 등 센서값으로 설비 고장 여부와 고장 유형을 예측.
   - 데이터가 정돈되어 입문용으로 좋지만 synthetic 성격이 있어 너무 쉽게 높은 성능이 나올 수 있음.
   - 단순 binary classification보다 고장 유형 multiclass, threshold tuning, 비용 기반 평가로 난이도를 조절하는 것이 좋음.

5. `TILDA 400 (64X64 patches)`
   - 섬유 표면 패치 이미지에서 결함/텍스처 유형을 분류하는 비전 검사 문제.
   - 이미지가 작아 현재 GPU 환경에서 가벼운 CNN 실험에 적합하고, 결함 유형별 혼동 분석이 쉬움.
   - 같은 질감 내 미세 결함을 구분해야 하므로 augmentation, normalization, 클래스별 recall 관리가 중요함.

## 4. 문제발생 현상 및 향후 해결 방안
- 사용자가 4번 `shivamb/machine-predictive-maintenance-classification`을 선택함.
- Python이 PATH에 등록되어 있지 않아 PowerShell 기반 Kaggle API 다운로드 스크립트로 전환함.
- 현재 프로젝트 루트와 주요 상위 경로에서 `.env`를 찾지 못해 Kaggle 다운로드는 인증 단계에서 실패함.
- 향후 `.env`를 프로젝트 루트에 배치한 뒤 `scripts/download_kaggle_dataset.ps1`을 재실행하면 다운로드를 이어갈 수 있음.
