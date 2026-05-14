# PPTX Summary Deck Log

## 1. 유저가 요청한 사항
- PPTX 스킬을 활용하여 슬라이드 자료 작성 후 업로드.
- 포함 내용:
  - 학습/평가 데이터셋 간단 소개.
  - 가장 성능이 높은 세팅의 특징 소개.
  - 모델이 잘 못 맞히는 케이스 소개.
  - LG CI를 활용한 디자인 테마 설정.

## 2. 너가 수행한 사항
- PPTX 스킬 지침을 확인.
- 실험 요약, feature importance, OOF 오분류 결과를 기반으로 6장 발표자료 구성.
- LG Red, charcoal, white 중심의 LG CI inspired 테마 적용.
- `python-pptx`를 설치하고 PPTX 생성 스크립트 작성.
- 슬라이드 생성 후 텍스트 QA 수행.
- QA 중 feature label truncation 문제를 발견하고 짧은 표시명으로 수정 후 재생성.

## 3. 그 중 완료된 사항
- PPTX 생성 완료:
  - `reports/predictive_maintenance_lg_summary.pptx`
- 생성 스크립트:
  - `scripts/create_predictive_maintenance_pptx.py`
- QA 스크립트:
  - `scripts/qa_pptx_text.py`
- 슬라이드 구성:
  - 1. Title
  - 2. 데이터셋 소개
  - 3. 실험 설계와 최고 성능 세팅
  - 4. 중요 입력 변수
  - 5. 오분류 케이스
  - 6. 모델 고도화 방향

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 현재 환경에서 LibreOffice/Poppler 기반 slide image 변환 도구가 PATH에서 확인되지 않아 이미지 기반 QA는 수행하지 못함.
  - 해결 방안: PowerPoint 또는 Streamlit/GitHub에서 파일을 열어 시각 검수 후 필요 시 spacing/레이아웃 보정.
- 문제: 텍스트 QA 결과 콘솔 한글 출력이 깨져 보임.
  - 해결 방안: PPTX 내부는 Unicode로 생성되었고, 콘솔 code page 문제로 판단. 실제 PowerPoint에서 한글 렌더링 확인 필요.
- 문제: 4번 슬라이드 feature 카드 텍스트가 길어 일부 잘릴 수 있음.
  - 해결 완료: `Rotational speed [rpm]` 등 긴 feature명을 `RPM`, `Power proxy`, `Tool wear` 등 짧은 표시명으로 변경.
