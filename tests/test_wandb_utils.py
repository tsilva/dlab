from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from src.utils.wandb import _artifact_aliases, _class_names, log_wandb_post_run, wandb_tags


def test_class_names_use_underlying_subset_dataset_classes() -> None:
    datamodule = SimpleNamespace(
        val_data=SimpleNamespace(dataset=SimpleNamespace(classes=["cat", "dog"]))
    )

    assert _class_names(datamodule, "custom") == ["cat", "dog"]


def test_class_names_fallback_for_fashion_mnist() -> None:
    assert _class_names(SimpleNamespace(), "fashion_mnist") == [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]


def test_wandb_tags_are_shortened_to_wandb_limit() -> None:
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "dataset": {"name": "cifar10"},
            "model": {"name": "resnet18"},
            "optimizer": {"name": "adamw"},
            "run": {
                "project": "cifar10_beat_baseline",
                "tags": ["diagnostic"],
                "stage": "03_dataset_difficulty",
                "study": "cifar10_resnet18_cifar_stem_aug_cosine75_ls005_jitter_medium_earlystop",
                "sweep_name": None,
            },
        }
    )

    tags = wandb_tags(cfg)

    assert all(len(tag) <= 64 for tag in tags)
    assert "cifar10_beat_baseline" in tags
    assert any(tag.startswith("cifar10_resnet18_cifar_stem_aug_cosine75") for tag in tags)


def test_artifact_aliases_include_research_project_keys() -> None:
    cfg = OmegaConf.create(
        {
            "run": {
                "project": "cifar10_beat_baseline",
                "stage": "03_dataset_difficulty",
                "study": "001_resnet18_baseline",
                "group": "cifar10-resnet18-baseline",
            }
        }
    )

    aliases = _artifact_aliases(cfg)

    assert aliases == [
        "group-cifar10-resnet18-baseline",
        "latest",
        "project-cifar10_beat_baseline",
        "stage-03_dataset_difficulty",
        "study-001_resnet18_baseline",
    ]


def test_wandb_post_run_summary_includes_run_target(monkeypatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.summary = {}

        def finish(self) -> None:
            self.summary["finished"] = True

    fake_run = FakeRun()
    monkeypatch.setattr("src.utils.wandb._active_wandb_run", lambda: fake_run)
    cfg = OmegaConf.create(
        {
            "wandb": {
                "enabled": True,
                "log_tables": False,
                "log_artifacts": False,
            },
            "run_target": {
                "name": "modal:L4",
                "provider": "modal",
                "launcher": "modal",
                "is_remote": True,
                "hostname": "modal-host",
                "platform": {
                    "system": "Linux",
                    "release": "6.1",
                    "machine": "x86_64",
                    "python": "3.12",
                },
                "cpu": {"count_logical": 8, "brand": "test-cpu"},
                "memory": {"total_gb": 30.0},
                "gpu": {
                    "requested": "L4",
                    "count": 1,
                    "names": ["NVIDIA L4"],
                    "cuda_available": True,
                },
                "container": {"modal_function_call_id": "fc-123"},
            },
        }
    )
    trainer = SimpleNamespace(global_step=10, current_epoch=1, callback_metrics={})
    lit_module = SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))

    log_wandb_post_run(
        cfg=cfg,
        trainer=trainer,
        lit_module=lit_module,
        datamodule=object(),
        run_dir="outputs/test",
        report_path=None,
        elapsed_seconds=2.0,
    )

    assert fake_run.summary["target/provider"] == "modal"
    assert fake_run.summary["target/gpu_requested"] == "L4"
    assert fake_run.summary["target/modal_function_call_id"] == "fc-123"
    assert fake_run.summary["finished"] is True
