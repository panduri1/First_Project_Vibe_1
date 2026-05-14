# GitHub Private Repo Push Log

## 1. 유저가 요청한 사항
- GitHub private repository를 만들고 현재 프로젝트를 `main` branch에 push.

## 2. 너가 수행한 사항
- Git/GitHub CLI 설치 여부를 확인.
- Git과 GitHub CLI가 없어 `winget`으로 설치.
- GitHub CLI 인증 상태를 확인.
- `.env`, `.venv` 등 민감/대용량 로컬 환경 파일을 제외하기 위한 `.gitignore`를 추가.
- GitHub CLI device login을 재시도하여 인증 완료.
- 로컬 git repository를 `main` branch로 초기화.
- Git 전역 설정은 변경하지 않고 GitHub noreply 계정 정보를 커밋 환경변수로만 사용.
- 프로젝트 파일을 커밋하고 GitHub private repository를 생성 후 push.

## 3. 그 중 완료된 사항
- Git 설치 완료.
- GitHub CLI 설치 완료.
- `.gitignore` 생성 완료.
- GitHub 인증 완료: `panduri1`
- 로컬 커밋 완료: `1b7daee`
- Private repository 생성 완료:
  - `https://github.com/panduri1/predictive-maintenance-dashboard`
- `main` branch push 완료.
- Remote tracking 설정 완료: `main...origin/main`

## 4. 문제발생 현상 및 향후 해결 방안
- 문제: 첫 GitHub CLI device login은 만료/실패로 종료됨.
  - 해결: 새 device code로 재로그인하여 성공.
- 문제: Git 사용자명이 설정되어 있지 않아 첫 commit 시도가 실패함.
  - 해결: Git 전역 설정은 변경하지 않고 `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `GIT_COMMITTER_NAME`, `GIT_COMMITTER_EMAIL` 환경변수로 커밋 수행.
- 문제: `.env`에 Kaggle 인증값이 포함되어 있어 커밋하면 안 됨.
  - 해결: `.gitignore`에 `.env`를 추가하고 ignore 상태를 확인한 뒤 커밋.
