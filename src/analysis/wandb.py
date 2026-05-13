from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_COLUMNS = [
    "name",
    "state",
    "stage",
    "study",
    "dataset",
    "model",
    "optimizer",
    "lr",
    "batch_size",
    "seed",
    "val/loss",
    "val/acc",
    "train/loss_epoch",
    "generalization/loss_gap",
    "params/trainable",
    "runtime/seconds",
]


def load_wandb_study_runs(
    *,
    project: str,
    study: str,
    entity: str | None = None,
    stage: str | None = None,
) -> pd.DataFrame:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B study analysis requires the wandb package.") from exc

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    rows = []
    for run in api.runs(path):
        row = _run_row(run)
        if row.get("study") != study:
            continue
        if stage and row.get("stage") != stage:
            continue
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_wandb_study(
    *,
    project: str,
    study: str,
    entity: str | None = None,
    stage: str | None = None,
    metric: str = "val/loss",
    goal: str = "minimize",
) -> pd.DataFrame:
    df = load_wandb_study_runs(project=project, entity=entity, study=study, stage=stage)
    if df.empty:
        return df
    sort_ascending = goal != "maximize"
    if metric in df.columns:
        df = df.sort_values(metric, ascending=sort_ascending, na_position="last")
    return df[[column for column in DEFAULT_COLUMNS if column in df.columns]]


def write_wandb_study_report(
    df: pd.DataFrame,
    *,
    study: str,
    metric: str = "val/loss",
    goal: str = "minimize",
    output_dir: str | Path = "reports",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / f"wandb-study-{study}.md"

    if df.empty:
        body = [
            f"# W&B Study: {study}",
            "",
            "No matching runs found.",
            "",
        ]
        report_path.write_text("\n".join(body), encoding="utf-8")
        return report_path

    best = df.iloc[0].to_dict()
    metric_value = best.get(metric, "n/a")
    changed_columns = _likely_changed_columns(df)
    grouped = _grouped_metric_summary(df, metric, changed_columns, goal)

    body = [
        f"# W&B Study: {study}",
        "",
        "## Best Run",
        "",
        f"- Name: `{best.get('name')}`",
        f"- `{metric}`: {metric_value}",
        f"- Dataset/model: `{best.get('dataset')}` / `{best.get('model')}`",
        f"- Optimizer/LR: `{best.get('optimizer')}` / `{best.get('lr')}`",
        "",
        "## Top Runs",
        "",
        _markdown_table(df.head(10)),
        "",
        "## Parameter Patterns",
        "",
        grouped if grouped else "No grouped parameter summary available.",
        "",
        "## Notes",
        "",
        "- What changed?",
        "- What pattern is visible?",
        "- What should be tried next?",
        "",
    ]
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def _run_row(run: Any) -> dict[str, Any]:
    config = dict(getattr(run, "config", {}) or {})
    summary = dict(getattr(run, "summary", {}) or {})
    run_cfg = config.get("run", {}) if isinstance(config.get("run"), dict) else {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    optimizer_cfg = config.get("optimizer", {}) if isinstance(config.get("optimizer"), dict) else {}

    row = {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "state": getattr(run, "state", None),
        "stage": run_cfg.get("stage"),
        "study": run_cfg.get("study"),
        "sweep_name": run_cfg.get("sweep_name"),
        "dataset": dataset_cfg.get("name"),
        "batch_size": dataset_cfg.get("batch_size"),
        "model": model_cfg.get("name"),
        "optimizer": optimizer_cfg.get("name"),
        "lr": optimizer_cfg.get("lr"),
        "seed": config.get("seed"),
    }
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        row[key] = value
    return row


def _likely_changed_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "dataset",
        "batch_size",
        "model",
        "optimizer",
        "lr",
        "seed",
        "params/trainable",
    ]
    return [
        column
        for column in candidates
        if column in df.columns and df[column].nunique(dropna=True) > 1
    ]


def _grouped_metric_summary(
    df: pd.DataFrame,
    metric: str,
    columns: list[str],
    goal: str,
) -> str:
    if metric not in df.columns or not columns:
        return ""
    lines = []
    ascending = goal != "maximize"
    for column in columns:
        grouped = (
            df[[column, metric]]
            .dropna()
            .groupby(column, dropna=True)[metric]
            .agg(["count", "mean", "min", "max"])
            .sort_values("mean", ascending=ascending)
        )
        if grouped.empty:
            continue
        lines.extend([f"### {column}", "", _markdown_table(grouped.reset_index()), ""])
    return "\n".join(lines).rstrip()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.astype(object).where(pd.notna(df), "")
    headers = [str(column) for column in text_df.columns]
    rows = [[str(value) for value in row] for row in text_df.to_numpy().tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)
