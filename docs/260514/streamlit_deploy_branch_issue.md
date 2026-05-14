# Streamlit Deploy Branch Issue

## 1. 유저가 요청한 사항
- Streamlit.io에서 app deploy 시 `NEW Project 1`은 조회되지만 `main` branch가 조회되지 않는 이유 확인.

## 2. 너가 수행한 사항
- 로컬 git branch와 remote URL 확인.
- GitHub repository metadata 확인.
- GitHub 원격 branch 목록 확인.
- `panduri1` 계정의 repo 목록 확인.

## 3. 그 중 완료된 사항
- 현재 로컬 프로젝트 remote:
  - `https://github.com/panduri1/predictive-maintenance-dashboard.git`
- 현재 로컬 branch:
  - `main`
- GitHub remote branch:
  - `main`
- GitHub repo visibility:
  - `PRIVATE`
- `panduri1` 계정 repo 목록에서 확인된 현재 프로젝트 repo:
  - `predictive-maintenance-dashboard`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: Streamlit에서 `NEW Project 1`을 선택하고 있다면, 그것은 현재 push된 GitHub repo 이름과 다름.
- 원인 가능성: `NEW Project 1`은 로컬 폴더명이고, 실제 GitHub repo는 `predictive-maintenance-dashboard`임.
- 해결 방안:
  - Streamlit Cloud에서 repo는 `panduri1/predictive-maintenance-dashboard`를 선택.
  - branch는 `main` 선택.
  - app file path는 `reports/data_gathering_predictive_maintenance_streamlit_app.py` 입력.
  - private repo가 보이지 않으면 Streamlit Cloud의 GitHub App 권한에서 private repo 접근 권한을 다시 부여.
