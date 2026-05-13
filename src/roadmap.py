from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class RoadmapStudy:
    id: str
    kind: str
    config_path: Path


@dataclass(frozen=True)
class RoadmapStage:
    id: str
    goal: str
    studies: list[str]


def load_roadmap(path: str | Path = "configs/roadmap/default.yaml") -> list[RoadmapStage]:
    cfg = OmegaConf.load(path)
    stages = []
    for stage in cfg.get("stages", []):
        stages.append(
            RoadmapStage(
                id=str(stage.id),
                goal=str(stage.goal),
                studies=[str(study) for study in stage.get("studies", [])],
            )
        )
    return stages


def resolve_study(study_id: str, configs_dir: str | Path = "configs") -> RoadmapStudy:
    configs_path = Path(configs_dir)
    sweep_path = configs_path / "sweep" / f"{study_id}.yaml"
    if sweep_path.exists():
        return RoadmapStudy(study_id, "sweep", sweep_path)
    experiment_path = configs_path / "experiment" / f"{study_id}.yaml"
    if experiment_path.exists():
        return RoadmapStudy(study_id, "experiment", experiment_path)
    raise FileNotFoundError(f"No sweep or experiment config found for study '{study_id}'.")


def find_stage(stages: list[RoadmapStage], stage_id: str) -> RoadmapStage:
    for stage in stages:
        if stage.id == stage_id:
            return stage
    known = ", ".join(stage.id for stage in stages)
    raise KeyError(f"Unknown roadmap stage '{stage_id}'. Available: {known}")


def roadmap_commands(
    stage: RoadmapStage,
    *,
    backend: str = "local",
    launcher: str | None = None,
) -> list[list[str]]:
    commands = []
    for study_id in stage.studies:
        study = resolve_study(study_id)
        if study.kind == "sweep":
            command = [sys.executable, "sweep.py", f"sweep={study.id}", f"backend={backend}"]
        else:
            command = [sys.executable, "train.py", f"experiment={study.id}"]
        if launcher:
            command.append(f"launcher={launcher}")
        commands.append(command)
    return commands


def run_stage(
    stage: RoadmapStage,
    *,
    backend: str = "local",
    launcher: str | None = None,
    dry_run: bool = False,
) -> list[int]:
    exit_codes = []
    for command in roadmap_commands(stage, backend=backend, launcher=launcher):
        print(" ".join(command), flush=True)
        if dry_run:
            exit_codes.append(0)
            continue
        completed = subprocess.run(command, check=False)
        exit_codes.append(completed.returncode)
        if completed.returncode != 0:
            break
    return exit_codes


def roadmap_table(stages: list[RoadmapStage]) -> list[dict[str, Any]]:
    return [
        {"stage": stage.id, "goal": stage.goal, "studies": ", ".join(stage.studies)}
        for stage in stages
    ]
