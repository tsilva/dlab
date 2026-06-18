from __future__ import annotations

import os
from pathlib import Path

ENV_PREFIXES = ("AWS_", "WANDB_")
ENV_KEYS = {
    "CHECKPOINT_BUCKET_URI",
}
REMOTE_ENV_MARKERS = (
    "MODAL_APP_ID",
    "MODAL_FUNCTION_CALL_ID",
    "MODAL_TASK_ID",
)


def load_experiment_env(dotenv_path: str | Path = ".env") -> None:
    """Load selected experiment secrets/config from .env without echoing values."""
    if any(os.environ.get(marker) for marker in REMOTE_ENV_MARKERS):
        return

    path = Path(dotenv_path)
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key.startswith(ENV_PREFIXES) and key not in ENV_KEYS:
            continue
        os.environ.setdefault(key, _dotenv_value(value))


def _dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
