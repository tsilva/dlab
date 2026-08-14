from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_STUDIES = [
    "001_wide_cnn_val_acc_selection",
    "fashion_mnist_cnn_wide_confirm_3seed",
    "fashion_mnist_cnn_capacity_reg_val",
    "fashion_mnist_cnn_schedule_confirm_3seed",
    "fashion_mnist_cnn_schedule_val",
    "fashion_mnist_cnn_transfer_val_3seed",
    "fashion_mnist_cnn_gradient_flow",
    "fashion_mnist_cnn_residual_deep_val",
    "fashion_mnist_cnn_deep_bn_tune_val",
    "fashion_mnist_cnn_bn_ema_ablation",
]

SIGNATURE_KEYS = [
    "model.name",
    "model.params.channels",
    "model.params.convs_per_stage",
    "model.params.batch_norm",
    "model.params.residual",
    "model.params.dropout",
    "optimizer.name",
    "optimizer.lr",
    "optimizer.weight_decay",
    "optimizer.scheduler.name",
    "loss.label_smoothing",
    "weight_averaging.name",
    "checkpoint.monitor",
    "checkpoint.mode",
    "dataset.batch_size",
    "dataset.augmentation.enabled",
]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    url: str
    state: str
    study: str | None
    sweep_name: str | None
    seed: int | None
    val_acc: float | None
    val_loss: float | None
    selected_val_acc: float | None
    selected_val_loss: float | None
    config: dict[str, Any]

    @property
    def ranking_metric(self) -> float | None:
        return self.selected_val_acc if self.selected_val_acc is not None else self.val_acc

    @property
    def tie_break_loss(self) -> float | None:
        return self.selected_val_loss if self.selected_val_loss is not None else self.val_loss

    @property
    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple((key, _stable_value(nested_get(self.config, key))) for key in SIGNATURE_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Fashion-MNIST CNN-like W&B runs by validation accuracy."
    )
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--dataset", default="fashion_mnist")
    parser.add_argument("--studies", nargs="*", default=DEFAULT_STUDIES)
    parser.add_argument("--output-dir", default="outputs/fashion_mnist_cnn_val_acc")
    args = parser.parse_args()

    runs = fetch_runs(
        entity=args.entity,
        project=args.project,
        dataset=args.dataset,
        studies=args.studies,
    )
    ranked = rank_runs(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ranked(output_dir / "ranked.csv", ranked)
    write_group_summary(output_dir / "groups.csv", ranked)
    write_markdown(output_dir / "ranked.md", ranked)

    print(f"runs={len(ranked)}")
    if ranked:
        best = ranked[0]
        print(
            "best="
            f"{best.ranking_metric:.6f} "
            f"loss={best.tie_break_loss:.6f} "
            f"run={best.name} "
            f"url={best.url}"
        )


def fetch_runs(entity: str, project: str, dataset: str, studies: list[str]) -> list[RunRecord]:
    import wandb

    api = wandb.Api(timeout=30)
    path = f"{entity}/{project}"
    filters: dict[str, Any] = {
        "config.dataset.name": dataset,
        "config.model.name": "cnn",
    }
    if studies:
        filters["config.run.study"] = {"$in": studies}

    raw_runs = api.runs(path, filters=filters, per_page=200)
    records = []
    for run in raw_runs:
        config = dict(run.config or {})
        summary = run.summary or {}
        if nested_get(config, "dataset.name") != dataset:
            continue
        if nested_get(config, "model.name") != "cnn":
            continue
        study = nested_get(config, "run.study")
        if studies and study not in studies:
            continue
        records.append(
            RunRecord(
                run_id=run.id,
                name=run.name,
                url=run.url,
                state=run.state,
                study=study,
                sweep_name=nested_get(config, "run.sweep_name"),
                seed=_optional_int(nested_get(config, "seed")),
                val_acc=_optional_float(summary.get("val/acc")),
                val_loss=_optional_float(summary.get("val/loss")),
                selected_val_acc=_optional_float(summary.get("evaluation/selected/val/acc")),
                selected_val_loss=_optional_float(summary.get("evaluation/selected/val/loss")),
                config=config,
            )
        )
    return records


def rank_runs(runs: list[RunRecord]) -> list[RunRecord]:
    candidates = [run for run in runs if run.state == "finished" and run.ranking_metric is not None]
    return sorted(candidates, key=run_sort_key)


def write_ranked(path: Path, ranked: list[RunRecord]) -> None:
    rows = [ranked_row(rank, run) for rank, run in enumerate(ranked, start=1)]
    write_csv(path, rows, fieldnames=list(rows[0]) if rows else ["rank"])


def write_group_summary(path: Path, ranked: list[RunRecord]) -> None:
    groups = grouped_runs(ranked)

    rows = []
    for signature, runs in groups.items():
        metrics = [run.ranking_metric for run in runs if run.ranking_metric is not None]
        losses = [run.tie_break_loss for run in runs if run.tie_break_loss is not None]
        if not metrics:
            continue
        row = {
            "n": len(metrics),
            "mean_val_acc": f"{statistics.mean(metrics):.6f}",
            "std_val_acc": f"{statistics.stdev(metrics):.6f}" if len(metrics) > 1 else "",
            "max_val_acc": f"{max(metrics):.6f}",
            "mean_val_loss": f"{statistics.mean(losses):.6f}" if losses else "",
            "seeds": " ".join(
                str(run.seed) for run in sorted(runs, key=lambda item: item.seed or -1)
            ),
            "run_ids": " ".join(run.run_id for run in runs),
        }
        row.update({key: value for key, value in signature})
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["mean_val_acc"]), float(row["mean_val_loss"] or 999.0)))
    fieldnames = (
        ["n", "mean_val_acc", "std_val_acc", "max_val_acc", "mean_val_loss", "seeds", "run_ids"]
        + SIGNATURE_KEYS
    )
    write_csv(path, rows, fieldnames=fieldnames)


def write_markdown(path: Path, ranked: list[RunRecord]) -> None:
    group_rows = grouped_rows(ranked)
    lines = [
        "# Fashion-MNIST CNN Validation Accuracy Ranking",
        "",
        "Metric: `evaluation/selected/val/acc` when present, otherwise `val/acc`.",
        "",
        "## Top Config Groups",
        "",
        (
            "| Rank | N | Mean Val Acc | Std | Max | Mean Val Loss | Channels | Dropout | "
            "WD | Monitor | Runs |"
        ),
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for index, row in enumerate(group_rows[:20], start=1):
        lines.append(
            f"| {index} | {row['n']} | {row['mean_val_acc']} | {row['std_val_acc']} | "
            f"{row['max_val_acc']} | {row['mean_val_loss']} | "
            f"{row.get('model.params.channels', '')} | {row.get('model.params.dropout', '')} | "
            f"{row.get('optimizer.weight_decay', '')} | "
            f"{row.get('checkpoint.monitor', '')}/{row.get('checkpoint.mode', '')} | "
            f"{row['run_ids']} |"
        )

    lines.extend(
        [
            "",
            "## Top Runs",
            "",
            "| Rank | Val Acc | Val Loss | Seed | Study | Run |",
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    for rank, run in enumerate(ranked[:30], start=1):
        val_acc = _format_float(run.ranking_metric)
        val_loss = _format_float(run.tie_break_loss)
        lines.append(
            f"| {rank} | {val_acc} | {val_loss} | {run.seed} | {run.study} | "
            f"[{run.run_id}]({run.url}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def grouped_rows(ranked: list[RunRecord]) -> list[dict[str, str]]:
    groups = grouped_runs(ranked)
    rows = []
    for signature, runs in groups.items():
        metrics = [run.ranking_metric for run in runs if run.ranking_metric is not None]
        losses = [run.tie_break_loss for run in runs if run.tie_break_loss is not None]
        if not metrics:
            continue
        row = {
            "n": str(len(metrics)),
            "mean_val_acc": f"{statistics.mean(metrics):.6f}",
            "std_val_acc": f"{statistics.stdev(metrics):.6f}" if len(metrics) > 1 else "",
            "max_val_acc": f"{max(metrics):.6f}",
            "mean_val_loss": f"{statistics.mean(losses):.6f}" if losses else "",
            "run_ids": " ".join(run.run_id for run in runs),
        }
        row.update({key: value for key, value in signature})
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (-float(row["mean_val_acc"]), float(row["mean_val_loss"] or 999.0)),
    )


def ranked_row(rank: int, run: RunRecord) -> dict[str, str]:
    row = {
        "rank": str(rank),
        "run_id": run.run_id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "study": str(run.study),
        "sweep_name": str(run.sweep_name),
        "seed": str(run.seed),
        "val_acc": _format_float(run.ranking_metric),
        "val_loss": _format_float(run.tie_break_loss),
    }
    row.update({key: _stable_value(nested_get(run.config, key)) for key in SIGNATURE_KEYS})
    return row


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_runs(
    ranked: list[RunRecord],
) -> dict[tuple[tuple[str, str], ...], list[RunRecord]]:
    best_by_signature_seed: dict[tuple[tuple[tuple[str, str], ...], int | None], RunRecord] = {}
    for run in ranked:
        key = (run.signature, run.seed)
        current = best_by_signature_seed.get(key)
        if current is None or run_sort_key(run) < run_sort_key(current):
            best_by_signature_seed[key] = run

    groups: dict[tuple[tuple[str, str], ...], list[RunRecord]] = {}
    for run in best_by_signature_seed.values():
        groups.setdefault(run.signature, []).append(run)
    for runs in groups.values():
        runs.sort(key=run_sort_key)
    return groups


def run_sort_key(run: RunRecord) -> tuple[float, float]:
    return (-(run.ranking_metric or 0.0), run.tie_break_loss or 999.0)


def nested_get(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "[" + ", ".join(_stable_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
