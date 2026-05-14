# First_Project_Vibe_1 Deploy Issue

## 1. 유저가 요청한 사항
- Streamlit Cloud에서 `First_Project_Vibe_1` repo를 deploy하려고 하는데 `main` branch를 설정할 수 없는 이유 확인.

## 2. 너가 수행한 사항
- GitHub CLI로 `panduri1/First_Project_Vibe_1` repository metadata 확인.
- branch 목록 확인.
- repository contents 존재 여부 확인.
- 사용자의 요청에 따라 `First_Project_Vibe_1`을 public 상태로 유지한 채 현재 프로젝트를 `main` branch로 push.

## 3. 그 중 완료된 사항
- `First_Project_Vibe_1` repo 상태 확인 완료.
- 초기 확인 결과:
  - URL: `https://github.com/panduri1/First_Project_Vibe_1`
  - Visibility: `PUBLIC`
  - `isEmpty`: `true`
  - default branch: empty
  - branch list: empty
- 조치 후 확인 결과:
  - URL: `https://github.com/panduri1/First_Project_Vibe_1`
  - Visibility: `PUBLIC`
  - `isEmpty`: `false`
  - default branch: `main`
  - branch list: `main`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: `First_Project_Vibe_1`은 비어 있는 repository라 `main` branch가 아직 생성되지 않음.
- 그래서 Streamlit Cloud에서 branch dropdown에 `main`이 표시되지 않음.
- 해결 완료:
  - 현재 프로젝트를 `First_Project_Vibe_1`에 push하여 `main` branch를 생성함.
  - 이제 Streamlit Cloud에서 `panduri1/First_Project_Vibe_1` repo와 `main` branch를 선택할 수 있어야 함.
- Streamlit deploy 설정:
  - Repository: `panduri1/First_Project_Vibe_1`
  - Branch: `main`
  - Main file path: `reports/data_gathering_predictive_maintenance_streamlit_app.py`
