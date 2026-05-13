from __future__ import annotations

from pathlib import Path

from src.roadmap import RoadmapStage, resolve_study, roadmap_commands


def test_resolve_study_prefers_sweep_configs() -> None:
    study = resolve_study("mlp_lr", configs_dir=Path("configs"))

    assert study.kind == "sweep"
    assert study.config_path == Path("configs/sweep/mlp_lr.yaml")


def test_roadmap_commands_plan_sweeps_and_experiments() -> None:
    stage = RoadmapStage(
        id="example",
        goal="test",
        studies=["mlp_lr", "mnist_mlp"],
    )

    commands = roadmap_commands(stage, backend="wandb", launcher="modal")

    assert commands[0][-3:] == ["sweep=mlp_lr", "backend=wandb", "launcher=modal"]
    assert commands[1][-2:] == ["experiment=mnist_mlp", "launcher=modal"]
