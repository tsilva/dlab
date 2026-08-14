from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execution.launchers import format_override_value

METRIC = "evaluation/selected/val/acc"
CONFIG_KEYS = [
    "model.params.rnn_type",
    "model.params.hidden_dim",
    "model.params.num_layers",
    "model.params.dropout",
    "model.params.bidirectional",
    "model.params.pooling",
    "model.params.sequence_axis",
    "optimizer.lr",
    "optimizer.weight_decay",
    "optimizer.scheduler.name",
    "loss.label_smoothing",
    "dataset.batch_size",
    "dataset.augmentation.enabled",
    "trainer.gradient_clip_val",
]


@dataclass(frozen=True)
class RunRecord:
    name: str
    url: str
    state: str
    seed: int | None
    metric: float | None
    val_loss: float | None
    config: dict[str, Any]

    @property
    def signature(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, nested_get(self.config, key)) for key in CONFIG_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank and confirm MNIST RNN W&B runs.")
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--run-project", default="mnist_rnn_val_acc")
    parser.add_argument("--study", default="001_rnn_architecture_search")
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--source-sweep", default=None)
    parser.add_argument("--confirm-sweep", default="mnist_rnn_val_acc_confirm_top")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/mnist_rnn_val_acc_search")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = fetch_runs(args.entity, args.project, args.run_project, args.study)
    ranked = rank_runs(runs, args.metric, source_sweep=args.source_sweep)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(output_dir, ranked, args.metric)

    print(f"ranked_runs={len(ranked)}")
    if ranked:
        best = ranked[0]
        print(f"best_metric={best.metric:.6f} run={best.name} url={best.url}")

    if args.confirm:
        commands = confirmation_commands(
            ranked,
            top_k=args.top_k,
            seeds=args.seeds,
            metric=args.metric,
            confirm_sweep=args.confirm_sweep,
        )
        write_commands(output_dir / "confirmation_commands.txt", commands)
        run_commands(commands, parallel=max(1, args.parallel), dry_run=args.dry_run)

        refreshed = fetch_runs(args.entity, args.project, args.run_project, args.study)
        refreshed_ranked = rank_runs(refreshed, args.metric, source_sweep=None)
        write_summary(output_dir, refreshed_ranked, args.metric, suffix="post_confirm")
        write_group_summary(output_dir, refreshed_ranked, args.metric)


def fetch_runs(entity: str, project: str, run_project: str, study: str) -> list[RunRecord]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}"
    filters = {
        "config.run.project": run_project,
        "config.run.study": study,
        "state": "finished",
    }
    try:
        raw_runs = list(api.runs(path, filters=filters, per_page=200))
    except Exception as exc:
        print(f"filtered W&B query failed, falling back to client-side filter: {exc}")
        raw_runs = list(api.runs(path, per_page=200))

    records = []
    for run in raw_runs:
        config = dict(run.config or {})
        if nested_get(config, "run.project") != run_project:
            continue
        if nested_get(config, "run.study") != study:
            continue
        records.append(
            RunRecord(
                name=run.name,
                url=run.url,
                state=run.state,
                seed=_optional_int(nested_get(config, "seed")),
                metric=_optional_float(run.summary.get(METRIC)),
                val_loss=_optional_float(
                    run.summary.get("evaluation/selected/val/loss", run.summary.get("val/loss"))
                ),
                config=config,
            )
        )
    return records


def rank_runs(
    runs: Iterable[RunRecord],
    metric: str,
    source_sweep: str | None,
) -> list[RunRecord]:
    candidates = []
    for run in runs:
        if source_sweep is not None and nested_get(run.config, "run.sweep_name") != source_sweep:
            continue
        value = _optional_float(run.metric if metric == METRIC else nested_get(run.config, metric))
        if value is None:
            continue
        candidates.append(run)
    return sorted(candidates, key=lambda item: (-float(item.metric or 0.0), item.val_loss or 999.0))


def confirmation_commands(
    ranked: list[RunRecord],
    top_k: int,
    seeds: list[int],
    metric: str,
    confirm_sweep: str,
) -> list[list[str]]:
    selected = unique_signatures(ranked)[:top_k]
    commands = []
    for index, run in enumerate(selected):
        for seed in seeds:
            overrides = [
                "experiment=mnist_rnn_val_acc_search",
                f"seed={seed}",
                f"run.sweep_name={format_override_value(confirm_sweep)}",
                f"run.sweep_index={index}",
                "wandb.entity=tsilva",
                "wandb.project=dlab",
                "wandb.stable_id=false",
                "wandb.resume=never",
            ]
            for key, value in run.signature:
                if value is not None:
                    overrides.append(f"{key}={format_override_value(value)}")
            commands.append([sys.executable, "train.py", *overrides])
    return commands


def unique_signatures(ranked: list[RunRecord]) -> list[RunRecord]:
    seen = set()
    unique = []
    for run in ranked:
        if run.signature in seen:
            continue
        seen.add(run.signature)
        unique.append(run)
    return unique


def run_commands(commands: list[list[str]], parallel: int, dry_run: bool) -> None:
    if dry_run:
        for command in commands:
            print(" ".join(command))
        return
    if parallel <= 1:
        for command in commands:
            print(" ".join(command), flush=True)
            subprocess.run(command, check=True)
        return

    active: list[subprocess.Popen] = []
    pending = list(commands)
    while pending or active:
        while pending and len(active) < parallel:
            command = pending.pop(0)
            print(" ".join(command), flush=True)
            active.append(subprocess.Popen(command))
        next_active = []
        for process in active:
            code = process.poll()
            if code is None:
                next_active.append(process)
            elif code != 0:
                raise subprocess.CalledProcessError(code, process.args)
        active = next_active


def write_summary(
    output_dir: Path,
    ranked: list[RunRecord],
    metric: str,
    suffix: str = "ranked",
) -> None:
    csv_path = output_dir / f"{suffix}.csv"
    md_path = output_dir / f"{suffix}.md"
    rows = [summary_row(index, run) for index, run in enumerate(ranked, start=1)]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as handle:
        handle.write(f"# MNIST RNN Search Ranking\n\nMetric: `{metric}`\n\n")
        handle.write(
            "| Rank | Metric | Val Loss | Seed | Run | Type | Hidden | Layers | Bidir | "
            "Pool | Axis | LR | WD | Scheduler | Dropout | LS | Batch | Aug | Clip |\n"
        )
        handle.write("|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---|---:|\n")
        for row in rows[:50]:
            handle.write(
                f"| {row['rank']} | {row['metric']} | {row['val_loss']} | {row['seed']} | "
                f"[{row['run']}]({row['url']}) | {row['rnn_type']} | {row['hidden_dim']} | "
                f"{row['num_layers']} | {row['bidirectional']} | {row['pooling']} | "
                f"{row['sequence_axis']} | {row['lr']} | {row['weight_decay']} | "
                f"{row['scheduler']} | {row['dropout']} | {row['label_smoothing']} | "
                f"{row['batch_size']} | {row['augmentation']} | {row['gradient_clip_val']} |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


def write_group_summary(output_dir: Path, ranked: list[RunRecord], metric: str) -> None:
    groups: dict[tuple[tuple[str, Any], ...], list[RunRecord]] = defaultdict(list)
    for run in ranked:
        groups[run.signature].append(run)
    summaries = []
    for signature, runs in groups.items():
        values = [float(run.metric) for run in runs if run.metric is not None]
        if not values:
            continue
        config = dict(signature)
        summaries.append(
            {
                "mean_metric": sum(values) / len(values),
                "best_metric": max(values),
                "n": len(values),
                "config": config,
                "runs": runs,
            }
        )
    summaries.sort(key=lambda item: (-item["mean_metric"], -item["best_metric"]))
    path = output_dir / "group_summary.md"
    with path.open("w") as handle:
        handle.write(f"# MNIST RNN Config Summary\n\nMetric: `{metric}`\n\n")
        handle.write(
            "| Rank | Mean | Best | N | Type | Hidden | Layers | Bidir | Pool | Axis | "
            "LR | WD | Scheduler | Dropout | LS | Batch | Aug | Clip |\n"
        )
        handle.write("|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|---|---:|\n")
        for index, item in enumerate(summaries[:30], start=1):
            config = item["config"]
            handle.write(
                f"| {index} | {item['mean_metric']:.6f} | {item['best_metric']:.6f} | "
                f"{item['n']} | {config.get('model.params.rnn_type')} | "
                f"{config.get('model.params.hidden_dim')} | "
                f"{config.get('model.params.num_layers')} | "
                f"{config.get('model.params.bidirectional')} | "
                f"{config.get('model.params.pooling')} | "
                f"{config.get('model.params.sequence_axis')} | {config.get('optimizer.lr')} | "
                f"{config.get('optimizer.weight_decay')} | "
                f"{config.get('optimizer.scheduler.name')} | "
                f"{config.get('model.params.dropout')} | {config.get('loss.label_smoothing')} | "
                f"{config.get('dataset.batch_size')} | "
                f"{config.get('dataset.augmentation.enabled')} | "
                f"{config.get('trainer.gradient_clip_val')} |\n"
            )
    print(f"wrote {path}")


def write_commands(path: Path, commands: list[list[str]]) -> None:
    with path.open("w") as handle:
        for command in commands:
            handle.write(" ".join(command) + "\n")
    print(f"wrote {path}")


def summary_row(rank: int, run: RunRecord) -> dict[str, Any]:
    return {
        "rank": rank,
        "metric": _format_float(run.metric),
        "val_loss": _format_float(run.val_loss),
        "seed": run.seed,
        "run": run.name,
        "url": run.url,
        "rnn_type": nested_get(run.config, "model.params.rnn_type"),
        "hidden_dim": nested_get(run.config, "model.params.hidden_dim"),
        "num_layers": nested_get(run.config, "model.params.num_layers"),
        "dropout": nested_get(run.config, "model.params.dropout"),
        "bidirectional": nested_get(run.config, "model.params.bidirectional"),
        "pooling": nested_get(run.config, "model.params.pooling"),
        "sequence_axis": nested_get(run.config, "model.params.sequence_axis"),
        "lr": nested_get(run.config, "optimizer.lr"),
        "weight_decay": nested_get(run.config, "optimizer.weight_decay"),
        "scheduler": nested_get(run.config, "optimizer.scheduler.name"),
        "label_smoothing": nested_get(run.config, "loss.label_smoothing"),
        "batch_size": nested_get(run.config, "dataset.batch_size"),
        "augmentation": nested_get(run.config, "dataset.augmentation.enabled"),
        "gradient_clip_val": nested_get(run.config, "trainer.gradient_clip_val"),
    }


def nested_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
