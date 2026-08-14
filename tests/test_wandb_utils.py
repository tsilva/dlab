from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.utils.wandb import (
    _artifact_aliases,
    _build_s3_artifact_uri,
    _class_names,
    _log_example_table,
    _log_run_artifact,
    _wandb_artifact_storage_uri,
    log_wandb_post_run,
    wandb_tags,
)


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


def test_build_s3_artifact_uri_sanitizes_artifact_path() -> None:
    assert (
        _build_s3_artifact_uri(
            "s3://dlab-checkpoints/dlab",
            "mnist/rnn run-output",
            "checkpoints/epoch=2-step=30.ckpt",
        )
        == "s3://dlab-checkpoints/dlab/mnist-rnn-run-output/checkpoints/epoch-2-step-30.ckpt"
    )


def test_wandb_artifact_storage_uri_appends_research_track(monkeypatch) -> None:
    monkeypatch.setenv("CHECKPOINT_BUCKET_URI", "s3://wandb")
    cfg = OmegaConf.create(
        {
            "wandb": {"artifact_storage_uri": None},
            "run": {"project": "cifar10_beat_baseline"},
        }
    )

    assert _wandb_artifact_storage_uri(cfg) == "s3://wandb/cifar10_beat_baseline"


def test_wandb_artifact_storage_uri_substitutes_research_track_placeholder(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CHECKPOINT_BUCKET_URI", "s3://wandb/{research_track_id}")
    cfg = OmegaConf.create(
        {
            "wandb": {"artifact_storage_uri": None},
            "run": {"project": "sequence_modeling_basics"},
        }
    )

    assert _wandb_artifact_storage_uri(cfg) == "s3://wandb/sequence_modeling_basics"


def test_log_run_artifact_uses_s3_references_when_storage_uri_is_set(
    monkeypatch,
    tmp_path: Path,
) -> None:
    logged = {}
    uploads = []

    class FakeArtifact:
        def __init__(self, name: str, type: str, metadata: dict) -> None:
            self.name = name
            self.type = type
            self.metadata = metadata
            self.references = []
            self.files = []

        def add_reference(self, uri: str, name: str) -> None:
            self.references.append((uri, name))

        def add_file(self, path: str, name: str) -> None:
            self.files.append((path, name))

    class FakeWandb:
        Artifact = FakeArtifact

    class FakeRun:
        def log_artifact(self, artifact: FakeArtifact, aliases: list[str]) -> None:
            logged["artifact"] = artifact
            logged["aliases"] = aliases

    run_dir = tmp_path / "run"
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "nested"
    checkpoints_dir.mkdir(parents=True)
    metrics_dir.mkdir()
    (run_dir / "config.yaml").write_text("seed: 1337\n", encoding="utf-8")
    (metrics_dir / "metrics.csv").write_text("epoch,acc\n0,0.1\n", encoding="utf-8")
    (checkpoints_dir / "last.ckpt").write_text("checkpoint", encoding="utf-8")
    report_path = tmp_path / "report.md"
    report_path.write_text("# Report\n", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)
    monkeypatch.setattr(
        "src.utils.wandb._upload_s3_artifact",
        lambda source_path, destination_uri: uploads.append((source_path, destination_uri)),
    )
    cfg = OmegaConf.create(
        {
            "experiment_name": "mnist/rnn run",
            "wandb": {"artifact_storage_uri": "s3://wandb"},
            "run": {
                "project": "sequence_modeling_basics",
                "stage": "04_sequence_modeling",
                "study": "003_pixel_rnn_diagnostic_rerun",
                "group": "mnist-rnn",
            },
        }
    )

    _log_run_artifact(cfg, FakeRun(), run_dir, str(report_path))

    artifact = logged["artifact"]
    assert artifact.files == []
    assert artifact.metadata["artifact_storage"] == {
        "mode": "s3_reference",
        "base_uri": "s3://wandb/sequence_modeling_basics",
    }
    assert sorted(name for _, name in artifact.references) == [
        "checkpoints/last.ckpt",
        "config.yaml",
        "metrics/nested/metrics.csv",
        "report.md",
    ]
    assert [destination for _, destination in uploads] == [
        "s3://wandb/sequence_modeling_basics/mnist-rnn-run-run/config.yaml",
        "s3://wandb/sequence_modeling_basics/mnist-rnn-run-run/metrics/nested/metrics.csv",
        "s3://wandb/sequence_modeling_basics/mnist-rnn-run-run/checkpoints/last.ckpt",
        "s3://wandb/sequence_modeling_basics/mnist-rnn-run-run/report.md",
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


def test_log_example_table_handles_multilabel_targets(monkeypatch) -> None:
    logged = {}

    class FakeTable:
        def __init__(self, columns: list[str]) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *values: object) -> None:
            self.rows.append(values)

    class FakeWandb:
        Table = FakeTable
        Image = staticmethod(lambda image: ("image", tuple(image.shape)))

    class FakeRun:
        def log(self, payload: dict) -> None:
            logged.update(payload)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [
                    [2.0, -2.0, 0.0],
                    [-2.0, 2.0, 2.0],
                ],
                device=x.device,
            )

    x = torch.zeros(2, 1, 4, 4)
    y = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    datamodule = SimpleNamespace(
        setup=lambda stage: None,
        val_dataloader=lambda: iter([(x, y)]),
        val_data=SimpleNamespace(classes=["a", "b", "c"]),
    )
    lit_module = SimpleNamespace(model=FakeModel(), device=torch.device("cpu"))
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "dataset": {"name": "chestmnist", "target_type": "multi_label_binary"},
            "loss": {"threshold": 0.5},
            "wandb": {"table_max_examples": 2, "table_split": "val"},
        }
    )
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    _log_example_table(cfg, FakeRun(), lit_module, datamodule)

    table = logged["examples/predictions"]
    assert table.columns == [
        "index",
        "image",
        "labels",
        "predictions",
        "positive_labels",
        "predicted_positive_labels",
        "mean_probability",
    ]
    assert table.rows[0][2] == [1, 0, 1]
    assert table.rows[0][3] == [1, 0, 1]
    assert table.rows[0][4] == ["a", "c"]
    assert table.rows[1][5] == ["b", "c"]
