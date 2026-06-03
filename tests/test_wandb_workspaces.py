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

    assert sorted(specs) == [
        "evaluation",
        "forensics",
        "gradient_debug",
        "gradients",
        "sweeps",
        "training",
    ]
    assert specs["training"].name == "Training monitor - 01_mlp_basics"
    assert specs["gradients"].name == "Gradient diagnostics - 01_mlp_basics"
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


def test_gradient_workspace_covers_flow_and_clipping_metrics() -> None:
    module = _load_module()

    specs = module.build_workspace_specs()
    panels = [
        panel
        for section in specs["gradients"].sections
        for panel in section.panels
    ]
    line_metrics = [
        metric
        for panel in panels
        for metric in panel.kwargs.get("y", [])
    ]
    scalar_metrics = [
        panel.kwargs.get("metric")
        for panel in panels
        if panel.kind == "ScalarChart"
    ]

    assert specs["gradients"].sections[0].name == "Clipping pressure"
    assert specs["gradients"].sections[0].pinned is True
    assert "train/grad_clip/raw_norm_step" in line_metrics
    assert "train/grad_clip/clip_coef_step" in line_metrics
    assert "train/grad_clip/was_clipped_epoch" in line_metrics
    assert "train/grad_flow/first_layer_norm_step" in line_metrics
    assert "train/grad_flow/dead_layers_step" in line_metrics
    assert "resume/event" in line_metrics
    assert "train/grad_clip/was_clipped_epoch" in scalar_metrics


def test_minimal_gradient_debug_workspace_prioritizes_core_diagnostics() -> None:
    module = _load_module()

    specs = module.build_workspace_specs()
    spec = specs["gradient_debug"]
    section_names = [section.name for section in spec.sections]
    line_metrics = [
        metric
        for section in spec.sections
        for panel in section.panels
        for metric in panel.kwargs.get("y", [])
    ]
    outcome_metrics = [
        metric
        for panel in spec.sections[1].panels
        for metric in panel.kwargs.get("metrics", [])
    ]
    media_keys = [
        key
        for section in spec.sections
        for panel in section.panels
        for key in panel.kwargs.get("media_keys", [])
    ]
    scalar_metrics = [
        panel.kwargs.get("metric")
        for section in spec.sections
        for panel in section.panels
        if panel.kind == "ScalarChart"
    ]

    assert spec.name == "Minimal gradient debug"
    assert section_names == [
        "1. High-leverage graph scan",
        "2. Outcome cards",
        "3. Secondary failure checks",
        "4. Error analysis",
        "5. Context",
    ]
    assert spec.sections[0].pinned is True
    assert spec.sections[0].smoothing_type is None
    assert all(panel.kind == "LinePlot" for panel in spec.sections[0].panels)
    assert spec.sections[0].panels[0].kwargs["title"] == "Train vs validation accuracy"
    assert spec.sections[0].panels[0].kwargs["y"] == ["train/acc_epoch", "val/acc"]
    assert spec.sections[0].panels[1].kwargs["title"] == "Train vs validation loss"
    assert spec.sections[0].panels[1].kwargs["y"] == ["train/loss_epoch", "val/loss"]
    assert spec.sections[0].panels[2].kwargs["title"] == "Raw norm vs clip threshold"
    assert spec.sections[0].panels[3].kwargs["title"] == "Clip coefficient kept"
    assert spec.sections[0].panels[4].kwargs["title"] == "First vs last layer grad norm"
    assert spec.sections[0].panels[5].kwargs["title"] == "First-to-last grad ratio"
    assert [panel.kind for panel in spec.sections[1].panels] == ["BarPlot", "BarPlot", "BarPlot"]
    assert spec.sections[1].panels[0].kwargs["title"] == "Final train vs validation accuracy"
    assert spec.sections[1].panels[0].kwargs["metrics"] == ["train/acc_epoch", "val/acc"]
    assert spec.sections[1].panels[1].kwargs["title"] == "Final train vs validation loss"
    assert spec.sections[1].panels[1].kwargs["metrics"] == ["train/loss_epoch", "val/loss"]
    assert spec.sections[1].panels[2].kwargs["title"] == "Final clipping summary"
    assert "val/acc" in line_metrics
    assert "val/loss" in line_metrics
    assert "train/grad_clip/raw_norm_step" in line_metrics
    assert "train/grad_clip/clip_coef_step" in line_metrics
    assert "train/grad_flow/first_to_last_ratio_step" in line_metrics
    assert "train/grad_flow/dead_layers_step" in line_metrics
    assert "resume/event" in line_metrics
    assert "val/acc" in outcome_metrics
    assert "train/acc_epoch" in outcome_metrics
    assert "val/loss" in outcome_metrics
    assert "train/loss_epoch" in outcome_metrics
    assert "train/grad_clip/clip_coef_epoch" in outcome_metrics
    assert "train/grad_clip/was_clipped_epoch" in outcome_metrics
    assert "errors/val_misclassifications" in media_keys
    assert "errors/val_confusion_matrix" in media_keys
    assert "errors/val_misclassification_count_logged" in scalar_metrics
    assert "errors/val_misclassification_count_total" in scalar_metrics
    assert "generalization/loss_gap" not in scalar_metrics
    assert "checkpoint/best_score" not in scalar_metrics
