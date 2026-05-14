from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REF = "shivamb/machine-predictive-maintenance-classification"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kaggle" / "machine_predictive_maintenance_classification"
LOG_PATH = PROJECT_ROOT / "docs" / "260514" / "kaggle_download_log.md"


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'").strip('"')
        values[key.strip()] = value
    return values


def _load_kaggle_credentials() -> tuple[str | None, str | None, list[str]]:
    notes: list[str] = []
    env_values = _parse_env_file(PROJECT_ROOT / ".env")

    username = os.getenv("KAGGLE_USERNAME") or env_values.get("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY") or env_values.get("KAGGLE_KEY")
    token = os.getenv("KAGGLE_API_TOKEN") or env_values.get("KAGGLE_API_TOKEN")

    if username and key:
        notes.append("Found KAGGLE_USERNAME/KAGGLE_KEY credentials.")
        return username, key, notes

    if token:
        token = token.strip()
        if username:
            notes.append("Found KAGGLE_USERNAME with KAGGLE_API_TOKEN used as key.")
            return username, token, notes

        try:
            parsed = json.loads(token)
            username = parsed.get("username")
            key = parsed.get("key")
            if username and key:
                notes.append("Found KAGGLE_API_TOKEN JSON credentials.")
                return username, key, notes
        except json.JSONDecodeError:
            pass

        if ":" in token:
            username, key = token.split(":", 1)
            if username and key:
                notes.append("Found KAGGLE_API_TOKEN username:key credentials.")
                return username, key, notes

        notes.append("KAGGLE_API_TOKEN exists but could not be parsed as Kaggle username/key.")
        return None, None, notes

    notes.append("No Kaggle credentials found in .env or process environment variables.")
    return None, None, notes


def _ensure_kaggle_package() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])


def _write_log(lines: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    lines = [
        "# Kaggle Download Log",
        "",
        f"- Dataset: `{DATASET_REF}`",
        f"- Output directory: `{OUTPUT_DIR.as_posix()}`",
    ]

    username, key, notes = _load_kaggle_credentials()
    lines.extend(f"- {note}" for note in notes)

    if not username or not key:
        lines.append("- Status: download skipped because credentials were not available.")
        _write_log(lines)
        return 2

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        _ensure_kaggle_package()
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET_REF, path=str(OUTPUT_DIR), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"- Status: download failed - `{type(exc).__name__}`")
        lines.append(f"- Detail: `{str(exc)[:500]}`")
        _write_log(lines)
        return 1

    downloaded_files = sorted(p.relative_to(OUTPUT_DIR).as_posix() for p in OUTPUT_DIR.rglob("*") if p.is_file())
    lines.append("- Status: download completed.")
    lines.append(f"- Downloaded file count: {len(downloaded_files)}")
    for file_name in downloaded_files[:20]:
        lines.append(f"  - `{file_name}`")
    if len(downloaded_files) > 20:
        lines.append(f"  - ... and {len(downloaded_files) - 20} more files")
    _write_log(lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
