from __future__ import annotations

import os

from src.utils.env import load_experiment_env


def test_load_experiment_env_loads_only_allowed_keys(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=r2-access-key",
                "export AWS_SECRET_ACCESS_KEY='r2-secret-key'",
                'CHECKPOINT_BUCKET_URI="s3://wandb"',
                "WANDB_API_KEY=wandb-key",
                "UNRELATED_SECRET=ignore-me",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CHECKPOINT_BUCKET_URI",
        "WANDB_API_KEY",
        "UNRELATED_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)

    load_experiment_env(env_path)

    assert os.environ["AWS_ACCESS_KEY_ID"] == "r2-access-key"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "r2-secret-key"
    assert os.environ["CHECKPOINT_BUCKET_URI"] == "s3://wandb"
    assert os.environ["WANDB_API_KEY"] == "wandb-key"
    assert "UNRELATED_SECRET" not in os.environ


def test_load_experiment_env_does_not_override_exported_env(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("CHECKPOINT_BUCKET_URI=s3://from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_BUCKET_URI", "s3://from-shell")

    load_experiment_env(env_path)

    assert os.environ["CHECKPOINT_BUCKET_URI"] == "s3://from-shell"


def test_load_experiment_env_skips_dotenv_inside_modal_worker(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("CHECKPOINT_BUCKET_URI=s3://from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("CHECKPOINT_BUCKET_URI", raising=False)
    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")

    load_experiment_env(env_path)

    assert "CHECKPOINT_BUCKET_URI" not in os.environ
