from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

PROJECT_DEFAULT = "dlab"
VIEW_CHOICES = ("training", "evaluation", "forensics", "gradient_debug", "gradients", "sweeps")


@dataclass(frozen=True)
class PanelSpec:
    kind: str
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class SectionSpec:
    name: str
    panels: list[PanelSpec]
    is_open: bool = True
    pinned: bool = False
    columns: int = 3
    rows: int = 2
    smoothing_type: str | None = None
    smoothing_weight: int | None = None


@dataclass(frozen=True)
class WorkspaceSpec:
    name: str
    sections: list[SectionSpec]
    max_runs: int = 20


def build_workspace_specs(
    stage: str | None = None,
    run_project: str | None = None,
) -> dict[str, WorkspaceSpec]:
    suffix_parts = [part for part in (run_project, stage) if part]
    suffix = f" - {' - '.join(suffix_parts)}" if suffix_parts else ""
    return {
        "training": WorkspaceSpec(
            name=f"Training monitor{suffix}",
            sections=[
                SectionSpec(
                    name="Live health",
                    pinned=True,
                    panels=[
                        line("Train loss", ["train/loss_step", "train/loss_epoch"]),
                        line("Validation loss", ["val/loss"]),
                        line("Validation accuracy", ["val/acc"]),
                        line("Learning rate", ["train/lr"]),
                        line("Gradient norm", ["train/grad_norm"]),
                        scalar("Best checkpoint score", "checkpoint/best_score"),
                    ],
                ),
                SectionSpec(
                    name="Runtime",
                    panels=[
                        scalar("Runtime seconds", "runtime/seconds"),
                        scalar("Steps per second", "runtime/steps_per_second"),
                        scalar("Global step", "trainer/global_step"),
                        scalar("Trainable params", "params/trainable"),
                    ],
                ),
                SectionSpec(
                    name="Gradient flow",
                    is_open=False,
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("First / middle / last layer grad norm", [
                            "train/grad_flow/first_layer_norm_step",
                            "train/grad_flow/middle_layer_norm_step",
                            "train/grad_flow/last_layer_norm_step",
                        ]),
                        line("Min / max layer grad norm", [
                            "train/grad_flow/min_layer_norm_step",
                            "train/grad_flow/max_layer_norm_step",
                        ]),
                        line(
                            "First-to-last grad ratio",
                            ["train/grad_flow/first_to_last_ratio_step"],
                        ),
                        line("Dead gradient layers", ["train/grad_flow/dead_layers_step"]),
                    ],
                ),
                SectionSpec(
                    name="Gradient clipping",
                    is_open=False,
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("Raw norm vs clip threshold", [
                            "train/grad_clip/raw_norm_step",
                            "train/grad_clip/threshold_step",
                        ]),
                        line("Clip coefficient", ["train/grad_clip/clip_coef_step"]),
                        line("Steps clipped", ["train/grad_clip/was_clipped_step"]),
                        line(
                            "Estimated clipped norm",
                            ["train/grad_clip/clipped_norm_estimate_step"],
                        ),
                    ],
                ),
            ],
        ),
        "evaluation": WorkspaceSpec(
            name=f"Evaluation monitor{suffix}",
            sections=[
                SectionSpec(
                    name="Validation and test",
                    pinned=True,
                    panels=[
                        line("Validation loss", ["val/loss"]),
                        line("Validation accuracy", ["val/acc"]),
                        line("Test loss", ["test/loss"]),
                        line("Test accuracy", ["test/acc"]),
                        scalar("Generalization loss gap", "generalization/loss_gap"),
                        scalar("Best checkpoint score", "checkpoint/best_score"),
                    ],
                ),
                SectionSpec(
                    name="Reconstruction objectives",
                    panels=[
                        line("Reconstruction loss", ["train/recon_loss_epoch", "val/recon_loss"]),
                        line("KL loss", ["train/kl_loss_epoch", "val/kl_loss"]),
                        line("VQ loss", ["train/vq_loss_epoch", "val/vq_loss"]),
                        line("Codebook health", [
                            "train/codebook_perplexity_epoch",
                            "val/codebook_perplexity",
                            "train/codebook_utilization_epoch",
                            "val/codebook_utilization",
                        ]),
                    ],
                ),
                SectionSpec(
                    name="Examples",
                    panels=[
                        media("Predictions", ["examples/predictions"]),
                        media("Reconstructions", ["examples/reconstructions", "reconstructions"]),
                        media("Latent traversals", ["latent_traversals"]),
                    ],
                ),
            ],
        ),
        "forensics": WorkspaceSpec(
            name=f"Forensics{suffix}",
            max_runs=50,
            sections=[
                SectionSpec(
                    name="Failure summary",
                    pinned=True,
                    panels=[
                        scalar(
                            "Validation error count logged",
                            "errors/val_misclassification_count_logged",
                        ),
                        scalar(
                            "Validation error count total",
                            "errors/val_misclassification_count_total",
                        ),
                        scalar(
                            "Test error count logged",
                            "errors/test_misclassification_count_logged",
                        ),
                        scalar(
                            "Test error count total",
                            "errors/test_misclassification_count_total",
                        ),
                        scalar("Early stopped", "early_stopping/stopped"),
                        scalar("Early stopping wait", "early_stopping/wait_count"),
                    ],
                ),
                SectionSpec(
                    name="Misclassifications",
                    panels=[
                        media("Validation misclassifications", ["errors/val_misclassifications"]),
                        media("Test misclassifications", ["errors/test_misclassifications"]),
                        PanelSpec("RunComparer", {"diff_only": "split"}),
                    ],
                ),
                SectionSpec(
                    name="Model and output artifacts",
                    is_open=False,
                    panels=[
                        media("Prediction examples", ["examples/predictions"]),
                        media("Filters", ["filters"]),
                        PanelSpec("RunComparer", {}),
                    ],
                ),
            ],
        ),
        "gradients": WorkspaceSpec(
            name=f"Gradient diagnostics{suffix}",
            max_runs=50,
            sections=[
                SectionSpec(
                    name="Clipping pressure",
                    pinned=True,
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("Raw norm vs clip threshold", [
                            "train/grad_clip/raw_norm_step",
                            "train/grad_clip/threshold_step",
                            "train/grad_norm",
                        ]),
                        line("Clip coefficient", [
                            "train/grad_clip/clip_coef_step",
                            "train/grad_clip/clip_coef_epoch",
                        ]),
                        line("Clipping frequency", [
                            "train/grad_clip/was_clipped_step",
                            "train/grad_clip/was_clipped_epoch",
                        ]),
                        line("Estimated clipped norm", [
                            "train/grad_clip/clipped_norm_estimate_step",
                            "train/grad_clip/clipped_norm_estimate_epoch",
                        ]),
                        scalar("Mean clip coefficient", "train/grad_clip/clip_coef_epoch"),
                        scalar("Mean clipped-step fraction", "train/grad_clip/was_clipped_epoch"),
                    ],
                ),
                SectionSpec(
                    name="Layer gradient flow",
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("First / middle / last layer grad norm", [
                            "train/grad_flow/first_layer_norm_step",
                            "train/grad_flow/middle_layer_norm_step",
                            "train/grad_flow/last_layer_norm_step",
                        ]),
                        line("Min / max layer grad norm", [
                            "train/grad_flow/min_layer_norm_step",
                            "train/grad_flow/max_layer_norm_step",
                        ]),
                        line("Layer balance ratios", [
                            "train/grad_flow/first_to_last_ratio_step",
                            "train/grad_flow/min_to_max_ratio_step",
                        ]),
                        line("Dead gradient layers", ["train/grad_flow/dead_layers_step"]),
                        scalar("Layer count", "train/grad_flow/layer_count_epoch"),
                        scalar("Final dead layers", "train/grad_flow/dead_layers_epoch"),
                    ],
                ),
                SectionSpec(
                    name="Optimization context",
                    panels=[
                        line("Train and validation loss", [
                            "train/loss_step",
                            "train/loss_epoch",
                            "val/loss",
                        ]),
                        line("Train and validation accuracy", [
                            "train/acc_step",
                            "train/acc_epoch",
                            "val/acc",
                        ]),
                        line("Learning rate", ["train/lr", "lr-Adam", "lr-AdamW", "lr-SGD"]),
                        line("Generalization gap", ["generalization/loss_gap"]),
                        line("Resume event", ["resume/event"]),
                    ],
                ),
                SectionSpec(
                    name="Post-hoc probes",
                    is_open=False,
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("Post-hoc train gradient norms", [
                            "posthoc/grad_flow/train/total_param_grad_norm",
                            "posthoc/grad_flow/train/first_layer_norm",
                            "posthoc/grad_flow/train/last_layer_norm",
                        ]),
                        line("Post-hoc val gradient norms", [
                            "posthoc/grad_flow/val/total_param_grad_norm",
                            "posthoc/grad_flow/val/first_layer_norm",
                            "posthoc/grad_flow/val/last_layer_norm",
                        ]),
                        line("Post-hoc dead layers", [
                            "posthoc/grad_flow/train/dead_layers_1e-8",
                            "posthoc/grad_flow/val/dead_layers_1e-8",
                        ]),
                    ],
                ),
            ],
        ),
        "gradient_debug": WorkspaceSpec(
            name=f"Minimal gradient debug{suffix}",
            max_runs=20,
            sections=[
                SectionSpec(
                    name="1. High-leverage graph scan",
                    pinned=True,
                    panels=[
                        line("Train vs validation accuracy", ["train/acc_epoch", "val/acc"]),
                        line("Train vs validation loss", ["train/loss_epoch", "val/loss"]),
                        line("Raw norm vs clip threshold", [
                            "train/grad_clip/raw_norm_step",
                            "train/grad_clip/threshold_step",
                        ]),
                        line("Clip coefficient kept", ["train/grad_clip/clip_coef_step"]),
                        line("First vs last layer grad norm", [
                            "train/grad_flow/first_layer_norm_step",
                            "train/grad_flow/last_layer_norm_step",
                        ]),
                        line(
                            "First-to-last grad ratio",
                            ["train/grad_flow/first_to_last_ratio_step"],
                        ),
                        line("Resume event", ["resume/event"]),
                    ],
                ),
                SectionSpec(
                    name="2. Outcome cards",
                    panels=[
                        bar(
                            "Final train vs validation accuracy",
                            ["train/acc_epoch", "val/acc"],
                            range_x=(0.0, 1.0),
                        ),
                        bar(
                            "Final train vs validation loss",
                            ["train/loss_epoch", "val/loss"],
                        ),
                        bar(
                            "Final clipping summary",
                            [
                                "train/grad_clip/clip_coef_epoch",
                                "train/grad_clip/was_clipped_epoch",
                            ],
                            range_x=(0.0, 1.0),
                        ),
                    ],
                ),
                SectionSpec(
                    name="3. Secondary failure checks",
                    smoothing_type="exponential",
                    smoothing_weight=20,
                    panels=[
                        line("Clipped step fraction", ["train/grad_clip/was_clipped_step"]),
                        line("Min vs max layer grad norm", [
                            "train/grad_flow/min_layer_norm_step",
                            "train/grad_flow/max_layer_norm_step",
                        ]),
                        line("Dead gradient layers", ["train/grad_flow/dead_layers_step"]),
                    ],
                ),
                SectionSpec(
                    name="4. Error analysis",
                    panels=[
                        media("Validation misclassifications", ["errors/val_misclassifications"]),
                        media("Validation confusion matrix", ["errors/val_confusion_matrix"]),
                        scalar(
                            "Validation errors logged",
                            "errors/val_misclassification_count_logged",
                        ),
                        scalar(
                            "Validation errors total",
                            "errors/val_misclassification_count_total",
                        ),
                    ],
                ),
                SectionSpec(
                    name="5. Context",
                    is_open=False,
                    panels=[
                        line("Learning rate", ["train/lr", "lr-Adam", "lr-AdamW", "lr-SGD"]),
                        scalar("Trainable params", "params/trainable"),
                        scalar("Runtime seconds", "runtime/seconds"),
                        PanelSpec("RunComparer", {"diff_only": "split"}),
                    ],
                ),
            ],
        ),
        "sweeps": WorkspaceSpec(
            name=f"Sweep comparison{suffix}",
            max_runs=100,
            sections=[
                SectionSpec(
                    name="Best run scan",
                    pinned=True,
                    columns=2,
                    rows=2,
                    panels=[
                        PanelSpec(
                            "ParallelCoordinatesPlot",
                            {
                                "title": "Hyperparameters vs validation loss",
                                "columns": [
                                    {"metric": "optimizer.lr", "display_name": "LR", "log": True},
                                    {"metric": "dataset.batch_size", "display_name": "Batch size"},
                                    {"metric": "model.width", "display_name": "Width"},
                                    {"metric": "model.depth", "display_name": "Depth"},
                                    {
                                        "metric": "val/loss",
                                        "display_name": "Val loss",
                                        "inverted": True,
                                    },
                                    {"metric": "val/acc", "display_name": "Val acc"},
                                ],
                            },
                        ),
                        PanelSpec("ParameterImportancePlot", {"with_respect_to": "val/loss"}),
                        PanelSpec("ScatterPlot", {
                            "title": "Params vs validation loss",
                            "x": "params/trainable",
                            "y": "val/loss",
                            "log_x": True,
                            "regression": True,
                        }),
                        PanelSpec("RunComparer", {"diff_only": "split"}),
                    ],
                ),
                SectionSpec(
                    name="Objective curves",
                    panels=[
                        line("Validation loss", ["val/loss"]),
                        line("Validation accuracy", ["val/acc"]),
                        line("Train loss", ["train/loss_epoch"]),
                        line("Generalization gap", ["generalization/loss_gap"]),
                    ],
                ),
            ],
        ),
    }


def line(title: str, y: list[str]) -> PanelSpec:
    return PanelSpec("LinePlot", {"title": title, "x": "Step", "y": y})


def scalar(title: str, metric: str) -> PanelSpec:
    return PanelSpec("ScalarChart", {"title": title, "metric": metric, "groupby_aggfunc": "mean"})


def bar(
    title: str,
    metrics: list[str],
    range_x: tuple[float | None, float | None] = (None, None),
) -> PanelSpec:
    return PanelSpec(
        "BarPlot",
        {
            "title": title,
            "metrics": metrics,
            "range_x": range_x,
            "groupby_aggfunc": "mean",
        },
    )


def media(title: str, media_keys: list[str]) -> PanelSpec:
    return PanelSpec(
        "MediaBrowser",
        {"title": title, "media_keys": media_keys, "num_columns": 4, "mode": "grid"},
    )


def save_workspaces(
    *,
    entity: str,
    project: str,
    specs: list[WorkspaceSpec],
    stage: str | None,
    run_project: str | None,
) -> list[str]:
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise RuntimeError(
            "Install the W&B Workspace API first: uv run --with wandb-workspaces "
            "python scripts/setup_wandb_workspaces.py --entity <entity>"
        ) from exc

    urls = []
    for spec in specs:
        workspace = ws.Workspace(
            name=spec.name,
            entity=entity,
            project=project,
            sections=[
                instantiate_section(ws, wr, section)
                for section in spec.sections
            ],
            settings=ws.WorkspaceSettings(
                x_axis="Step",
                group_by_prefix="first",
                max_runs=spec.max_runs,
                remove_legends_from_panels=False,
                tooltip_number_of_runs="default",
                tooltip_color_run_names=True,
                auto_expand_panel_search_results=True,
            ),
            runset_settings=runset_settings(ws, stage=stage, run_project=run_project),
            auto_generate_panels=False,
        )
        saved = workspace.save()
        urls.append(getattr(saved, "url", None) or getattr(workspace, "url", ""))
    return urls


def instantiate_section(ws: Any, wr: Any, section: SectionSpec) -> Any:
    kwargs = {
        "name": section.name,
        "panels": [instantiate_panel(wr, panel) for panel in section.panels],
        "is_open": section.is_open,
        "pinned": section.pinned,
        "layout_settings": ws.SectionLayoutSettings(
            columns=section.columns,
            rows=section.rows,
        ),
    }
    panel_settings = section_panel_settings(ws, section)
    if panel_settings is not None:
        kwargs["panel_settings"] = panel_settings
    return ws.Section(**kwargs)


def instantiate_panel(wr: Any, panel: PanelSpec) -> Any:
    kwargs = dict(panel.kwargs)
    if panel.kind == "ParallelCoordinatesPlot":
        kwargs["columns"] = [
            wr.ParallelCoordinatesPlotColumn(**column) for column in kwargs["columns"]
        ]
    return getattr(wr, panel.kind)(**kwargs)


def section_panel_settings(ws: Any, section: SectionSpec) -> Any | None:
    if section.smoothing_type is None:
        return None
    return ws.SectionPanelSettings(
        smoothing_type=section.smoothing_type,
        smoothing_weight=section.smoothing_weight or 0,
    )


def runset_settings(ws: Any, stage: str | None, run_project: str | None = None) -> Any:
    kwargs = {
        "pinned_columns": [
            "run:displayName",
            "config:run.project",
            "config:run.stage",
            "config:run.study",
            "config:dataset.name",
            "config:model.name",
            "config:optimizer.lr",
            "summary:val/loss",
            "summary:val/acc",
            "summary:checkpoint/best_score",
        ],
    }
    filters = []
    if run_project:
        filters.append(f"Config('run.project') = '{run_project}'")
    if stage:
        filters.append(f"Config('run.stage') = '{stage}'")
    if filters:
        kwargs["filters"] = " AND ".join(filters)
    return ws.RunsetSettings(**kwargs)


def describe_specs(specs: list[WorkspaceSpec]) -> str:
    lines = []
    for spec in specs:
        lines.append(f"{spec.name}:")
        for section in spec.sections:
            status = "open" if section.is_open else "closed"
            pinned = ", pinned" if section.pinned else ""
            lines.append(f"  - {section.name} ({status}{pinned}): {len(section.panels)} panels")
        lines.append("")
    return "\n".join(lines).rstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create curated W&B saved workspace views.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", PROJECT_DEFAULT))
    parser.add_argument("--stage", default=None, help="Optional run.stage value to filter views.")
    parser.add_argument(
        "--run-project",
        default=None,
        help="Optional run.project value to filter views.",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=VIEW_CHOICES,
        help="View to create. Repeat for multiple views. Defaults to all views.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned views only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs_by_name = build_workspace_specs(stage=args.stage, run_project=args.run_project)
    selected = args.view or list(VIEW_CHOICES)
    specs = [specs_by_name[name] for name in selected]

    if args.dry_run:
        print(describe_specs(specs))
        return

    if not args.entity:
        raise SystemExit("Provide --entity or set WANDB_ENTITY.")

    urls = save_workspaces(
        entity=args.entity,
        project=args.project,
        specs=specs,
        stage=args.stage,
        run_project=args.run_project,
    )
    for url in urls:
        print(url)


if __name__ == "__main__":
    main()
