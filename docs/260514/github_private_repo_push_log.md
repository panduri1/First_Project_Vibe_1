# GitHub Private Repo Push Log

## 1. 유저가 요청한 사항
- GitHub private repository를 만들고 현재 프로젝트를 `main` branch에 push.

## 2. 너가 수행한 사항
- Git/GitHub CLI 설치 여부를 확인.
- Git과 GitHub CLI가 없어 `winget`으로 설치.
- GitHub CLI 인증 상태를 확인.
- `.env`, `.venv` 등 민감/대용량 로컬 환경 파일을 제외하기 위한 `.gitignore`를 추가.

## 3. 그 중 완료된 사항
- Git 설치 완료.
- GitHub CLI 설치 완료.
- `.gitignore` 생성 완료.

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: GitHub CLI가 아직 로그인되어 있지 않아 private repo 생성과 push는 인증 전까지 진행할 수 없음.
- 해결 방안: `gh auth login`으로 GitHub 인증을 완료한 뒤 repo 생성, commit, push를 이어서 수행.
