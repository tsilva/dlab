from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.execution.experiment import (
    _ModalVolumeCommitCallback,
    _ResumeRestoreEventCallback,
    _early_stopping_metrics,
    _resolve_resume_checkpoint,
    _resume_metrics,
    _run_selection_validation,
    _run_tta_evaluation_if_enabled,
    _wandb_run_id,
)
from src.trainers import ResearchLitModule


def test_selection_validation_reloads_best_checkpoint_and_promotes_val_metrics() -> None:
    calls = {}

    class FakeTrainer:
        checkpoint_callback = SimpleNamespace(best_model_path="/tmp/best.ckpt")

        def validate(self, lit_module, *, datamodule, ckpt_path, weights_only, verbose):
            calls["lit_module"] = lit_module
            calls["datamodule"] = datamodule
            calls["ckpt_path"] = ckpt_path
            calls["weights_only"] = weights_only
            calls["verbose"] = verbose
            return [{"val/loss": 0.42, "val/acc": 0.91}]

    cfg = OmegaConf.create(
        {
            "trainer": {"enable_checkpointing": True},
            "evaluation": {"selection": {"enabled": True, "ckpt_path": "best"}},
        }
    )

    metrics = _run_selection_validation(
        cfg,
        FakeTrainer(),
        lit_module=object(),
        datamodule=object(),
    )

    assert calls["ckpt_path"] == "best"
    assert calls["weights_only"] is False
    assert calls["verbose"] is False
    assert metrics["val/loss"] == 0.42
    assert metrics["val/acc"] == 0.91
    assert metrics["evaluation/selected/val/loss"] == 0.42
    assert metrics["evaluation/selected/val/acc"] == 0.91
    assert metrics["evaluation/selected_checkpoint_path"] == "/tmp/best.ckpt"


def test_selection_validation_skips_when_no_best_checkpoint_exists() -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {"enable_checkpointing": True},
            "evaluation": {"selection": {"enabled": True, "ckpt_path": "best"}},
        }
    )
    trainer = SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=""))

    assert _run_selection_validation(cfg, trainer, object(), object()) == {}


def test_tta_evaluation_loads_selected_checkpoint_and_logs_metrics(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "optimizer": {"name": "adamw", "lr": 0.001},
            "loss": {
                "label_smoothing": 0.0,
                "mixup": {"enabled": False},
                "cutmix": {"enabled": False},
            },
            "dataset": {"name": "cifar10"},
            "trainer": {"enable_checkpointing": True, "gradient_clip_val": 0.0},
            "gradient_flow": {"enabled": False},
            "evaluation": {
                "tta": {
                    "enabled": True,
                    "split": "val",
                    "ckpt_path": "best",
                    "transforms": ["identity", "hflip"],
                    "weights_only": False,
                }
            },
        }
    )
    lit_module = ResearchLitModule(nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2)), cfg)
    checkpoint_path = tmp_path / "best.ckpt"
    torch.save({"state_dict": lit_module.state_dict()}, checkpoint_path)
    x = torch.randn(4, 3, 4, 4)
    y = torch.randint(0, 2, (4,))
    datamodule = SimpleNamespace(
        val_dataloader=lambda: DataLoader(TensorDataset(x, y), batch_size=2)
    )
    logged = []
    trainer = SimpleNamespace(
        checkpoint_callback=SimpleNamespace(best_model_path=str(checkpoint_path)),
        global_step=123,
        loggers=[SimpleNamespace(log_metrics=lambda metrics, step=None: logged.append((metrics, step)))],
    )

    metrics = _run_tta_evaluation_if_enabled(cfg, trainer, lit_module, datamodule)

    assert metrics["evaluation/tta/val/num_views"] == 2.0
    assert metrics["evaluation/tta/val/num_examples"] == 4.0
    assert 0.0 <= metrics["evaluation/tta/val/acc"] <= 1.0
    assert metrics["evaluation/tta_checkpoint_path"] == str(checkpoint_path)
    assert logged
    assert logged[0][1] == 123
    assert "evaluation/tta_checkpoint_path" not in logged[0][0]


def test_early_stopping_metrics_snapshot_callback_state() -> None:
    callback = SimpleNamespace(stopped_epoch=51, wait_count=8, patience=8)

    assert _early_stopping_metrics(callback) == {
        "early_stopping/stopped_epoch": 51,
        "early_stopping/stopped": True,
        "early_stopping/wait_count": 8,
        "early_stopping/patience": 8,
    }


def test_resolve_resume_checkpoint_uses_existing_last_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    last_checkpoint = checkpoint_dir / "last.ckpt"
    last_checkpoint.write_bytes(b"checkpoint")
    cfg = OmegaConf.create(
        {
            "trainer": {"enable_checkpointing": True},
            "resume": {"enabled": True, "ckpt_path": "last", "require_checkpoint": False},
        }
    )

    assert _resolve_resume_checkpoint(cfg, checkpoint_dir) == str(last_checkpoint)
    assert _resume_metrics(str(last_checkpoint)) == {
        "resume/started_from_checkpoint": True,
        "resume/checkpoint_path": str(last_checkpoint),
    }


def test_resolve_resume_checkpoint_skips_missing_last_by_default(tmp_path) -> None:
    cfg = OmegaConf.create(
        {
            "trainer": {"enable_checkpointing": True},
            "resume": {"enabled": True, "ckpt_path": "last", "require_checkpoint": False},
        }
    )

    assert _resolve_resume_checkpoint(cfg, tmp_path / "checkpoints") is None
    assert _resume_metrics(None) == {"resume/started_from_checkpoint": False}


def test_modal_volume_commit_callback_commits_configured_volume(monkeypatch) -> None:
    calls = []

    class FakeVolume:
        def commit(self):
            calls.append("commit")

    fake_modal = SimpleNamespace(
        Volume=SimpleNamespace(
            from_name=lambda name: calls.append(("from_name", name)) or FakeVolume()
        )
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    callback = _ModalVolumeCommitCallback("dlab-training-runs")
    callback.on_validation_end(trainer=object(), pl_module=object())
    callback.on_fit_end(trainer=object(), pl_module=object())

    assert calls == [("from_name", "dlab-training-runs"), "commit", "commit"]


def test_modal_volume_commit_callback_does_not_raise_on_commit_failure(monkeypatch) -> None:
    class FakeVolume:
        def commit(self):
            raise RuntimeError("temporary commit failure")

    fake_modal = SimpleNamespace(
        Volume=SimpleNamespace(from_name=lambda name: FakeVolume())
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    callback = _ModalVolumeCommitCallback("dlab-training-runs")

    callback.on_exception(trainer=object(), pl_module=object(), exception=KeyboardInterrupt())


def test_resume_restore_event_callback_logs_at_restored_step() -> None:
    calls = []

    class FakeLogger:
        def log_metrics(self, metrics, step=None):
            calls.append((metrics, step))

    trainer = SimpleNamespace(
        loggers=[FakeLogger()],
        global_step=1688,
        current_epoch=2,
    )
    callback = _ResumeRestoreEventCallback("/vol/dlab/outputs/run/checkpoints/last.ckpt")

    callback.on_train_start(trainer=trainer, pl_module=object())
    callback.on_train_start(trainer=trainer, pl_module=object())

    assert calls == [
        (
            {
                "resume/event": 1.0,
                "resume/restored_from_checkpoint": 1.0,
                "resume/restored_epoch": 2,
                "resume/restored_global_step": 1688,
            },
            1688,
        )
    ]


def test_wandb_run_id_uses_explicit_id_when_configured() -> None:
    cfg = OmegaConf.create(
        {
            "experiment_name": "resume-test",
            "wandb": {
                "project": "dlab",
                "entity": "tsilva",
                "id": "manual-run-id",
                "stable_id": True,
            },
        }
    )

    assert _wandb_run_id(cfg) == "manual-run-id"


def test_wandb_run_id_is_stable_for_experiment_identity() -> None:
    cfg = OmegaConf.create(
        {
            "experiment_name": "cifar10-densenet_seed1337",
            "wandb": {
                "project": "dlab",
                "entity": "tsilva",
                "id": None,
                "stable_id": True,
            },
        }
    )

    first = _wandb_run_id(cfg)
    second = _wandb_run_id(cfg)

    assert first == second
    assert first is not None
    assert first.startswith("dlab-")
    assert len(first) == 29


def test_wandb_run_id_can_use_wandb_generated_ids() -> None:
    cfg = OmegaConf.create(
        {
            "experiment_name": "cifar10-densenet_seed1337",
            "wandb": {
                "project": "dlab",
                "entity": "tsilva",
                "id": None,
                "stable_id": False,
            },
        }
    )

    assert _wandb_run_id(cfg) is None
