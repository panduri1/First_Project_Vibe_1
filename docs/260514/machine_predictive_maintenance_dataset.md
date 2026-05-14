# Machine Predictive Maintenance Dataset

## Dataset Source
- Kaggle ref: `shivamb/machine-predictive-maintenance-classification`
- Local CSV: `data/kaggle/machine_predictive_maintenance_classification/predictive_maintenance.csv`
- File name: `predictive_maintenance.csv`

## What This Data Is About
이 데이터는 설비의 운전 조건과 센서값을 이용해 **기계 고장 여부**를 예측하는 예지보전 데이터셋입니다.

현업 관점에서는 다음 질문을 해결하는 데 사용할 수 있습니다.

- 현재 설비 상태가 고장 위험 상태인지 예측할 수 있는가?
- 어떤 조건에서 고장 위험이 커지는가?
- 고장을 놓치는 케이스(False Negative)는 어떤 패턴을 가지는가?
- Streamlit 대시보드에서 설비 상태 입력값을 넣으면 고장 확률을 보여줄 수 있는가?

## Basic Shape
- Rows: `10,000`
- Columns: `10`
- Main task: binary classification
- Primary target: `Target`
- Secondary target: `Failure Type`

## Columns
| Column | Type | Meaning | Modeling Use |
|---|---|---|---|
| `UDI` | ID | 행 단위 고유 번호 | feature에서 제외 |
| `Product ID` | ID | 제품/장비 식별자 | feature에서 제외 권장 |
| `Type` | categorical | 제품 품질/타입 구분 (`L`, `M`, `H`) | feature로 사용 가능 |
| `Air temperature [K]` | numeric | 공기 온도, Kelvin | feature |
| `Process temperature [K]` | numeric | 공정 온도, Kelvin | feature |
| `Rotational speed [rpm]` | numeric | 회전 속도 | feature |
| `Torque [Nm]` | numeric | 토크 | feature |
| `Tool wear [min]` | numeric | 공구 마모 시간 | feature |
| `Target` | binary target | 고장 여부 (`0`: 정상, `1`: 고장) | 1차 타깃 |
| `Failure Type` | categorical target | 고장 유형 또는 정상 상태 | 2차 타깃/분석용 |

## Sample Rows
| UDI | Product ID | Type | Air temperature [K] | Process temperature [K] | Rotational speed [rpm] | Torque [Nm] | Tool wear [min] | Target | Failure Type |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | M14860 | M | 298.1 | 308.6 | 1551 | 42.8 | 0 | 0 | No Failure |
| 2 | L47181 | L | 298.2 | 308.7 | 1408 | 46.3 | 3 | 0 | No Failure |
| 3 | L47182 | L | 298.1 | 308.5 | 1498 | 49.4 | 5 | 0 | No Failure |
| 4 | L47183 | L | 298.2 | 308.6 | 1433 | 39.5 | 7 | 0 | No Failure |
| 5 | L47184 | L | 298.2 | 308.7 | 1408 | 40.0 | 9 | 0 | No Failure |
| 6 | M14865 | M | 298.1 | 308.6 | 1425 | 41.9 | 11 | 0 | No Failure |

## Target Distribution
| Target | Meaning | Count | Ratio |
|---:|---|---:|---:|
| 0 | No machine failure | 9,661 | 96.61% |
| 1 | Machine failure | 339 | 3.39% |

이 데이터는 고장 데이터가 매우 적은 **불균형 분류 문제**입니다. Accuracy만 보면 성능이 좋아 보일 수 있으므로 `recall`, `F1`, `PR-AUC`, confusion matrix를 함께 봐야 합니다.

## Failure Type Distribution
| Failure Type | Count |
|---|---:|
| No Failure | 9,652 |
| Heat Dissipation Failure | 112 |
| Power Failure | 95 |
| Overstrain Failure | 78 |
| Tool Wear Failure | 45 |
| Random Failures | 18 |

## Product Type Distribution
| Type | Count |
|---|---:|
| L | 6,000 |
| M | 2,997 |
| H | 1,003 |

## Important Data Quality Note
`Target`과 `Failure Type`이 완전히 1:1로 맞지는 않습니다.

- `Target = 0`인데 `Failure Type = Random Failures`인 행: `18`
- `Target = 1`인데 `Failure Type = No Failure`인 행: `9`
- Total mismatch: `27`

따라서 1차 모델에서는 `Target`을 기준으로 binary classification을 수행하고, `Failure Type`은 feature에 넣지 않는 것이 좋습니다. `Failure Type`을 feature로 넣으면 정답 정보를 미리 알려주는 데이터 누수가 발생합니다.

## Recommended ML Problem Definition
### Step 1. Binary Failure Prediction
- Goal: `Target`이 1인지 예측
- Features:
  - `Type`
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
- Excluded:
  - `UDI`
  - `Product ID`
  - `Failure Type`

### Step 2. Failure Type Analysis
- Goal: 고장으로 판단된 케이스의 유형 분석
- Target: `Failure Type`
- Scope: 기본적으로 `Target = 1`인 행 중심
- Caveat: `Target`과 `Failure Type`의 불일치 27건을 어떻게 처리할지 정책 결정 필요

## Suggested Derived Features
- `temperature_gap = Process temperature [K] - Air temperature [K]`
- `power_proxy = Rotational speed [rpm] * Torque [Nm]`
- `wear_per_torque = Tool wear [min] / Torque [Nm]`

## Dashboard Ideas
- 전체 고장률 KPI
- 제품 타입별 고장률
- 센서 분포 및 이상치 차트
- 고장/정상별 boxplot
- confusion matrix
- precision-recall curve
- threshold 조정에 따른 missed failure와 false alarm 변화
- 오분류 케이스 테이블

## Next Step
이 MD 파일은 데이터 이해용 문서입니다. 다음 단계에서는 이 CSV를 기준으로 EDA/전처리 산출물, 모델 학습 코드, Streamlit 대시보드를 생성하면 됩니다.
