from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "setup_wandb_workspaces.py"
    spec = importlib.util.spec_from_file_location("setup_wandb_workspaces", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_workspace_specs_create_stage_specific_views() -> None:
    module = _load_module()

    specs = module.build_workspace_specs(stage="01_mlp_basics")

    assert sorted(specs) == ["evaluation", "forensics", "sweeps", "training"]
    assert specs["training"].name == "Training monitor - 01_mlp_basics"
    assert specs["training"].sections[0].is_open is True
    assert specs["training"].sections[0].pinned is True
    assert specs["training"].sections[0].panels[0].kwargs["y"] == [
        "train/loss_step",
        "train/loss_epoch",
    ]


def test_workspace_specs_cover_logged_media_and_forensics_keys() -> None:
    module = _load_module()

    specs = module.build_workspace_specs()
    evaluation_media = [
        key
        for section in specs["evaluation"].sections
        for panel in section.panels
        for key in panel.kwargs.get("media_keys", [])
    ]
    forensic_metrics = [
        panel.kwargs.get("metric")
        for section in specs["forensics"].sections
        for panel in section.panels
    ]

    assert "examples/predictions" in evaluation_media
    assert "examples/reconstructions" in evaluation_media
    assert "errors/val_misclassification_count_total" in forensic_metrics
