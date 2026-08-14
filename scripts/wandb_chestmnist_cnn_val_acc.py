from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execution.launchers import format_override_value

VAL_METRIC = "evaluation/selected/val/acc"
VAL_LOSS = "evaluation/selected/val/loss"
POSITIVE_RATE = "evaluation/selected/val/pred_positive_rate"
DEFAULT_SWEEPS = [
    "chestmnist_cnn_val_acc_phase1_sweep",
    "chestmnist_resnet18_val_acc_phase1_sweep",
    "chestmnist_cnn_val_acc_confirm_top",
    "chestmnist_cnn_val_acc_threshold_phase2_sweep",
    "chestmnist_cnn_val_acc_high_threshold_phase3",
    "chestmnist_cnn_val_acc_scalar_threshold_phase4",
]
SIGNATURE_KEYS = [
    "model.name",
    "model.params.channels",
    "model.params.convs_per_stage",
    "model.params.batch_norm",
    "model.params.residual",
    "model.params.dropout",
    "model.params.model_name",
    "model.params.pretrained",
    "model.params.stem",
    "optimizer.name",
    "optimizer.lr",
    "optimizer.weight_decay",
    "optimizer.scheduler.name",
    "loss.name",
    "loss.threshold",
    "checkpoint.monitor",
    "checkpoint.mode",
    "dataset.batch_size",
    "dataset.augmentation.enabled",
    "dataset.augmentation.random_affine.degrees",
    "dataset.augmentation.random_affine.translate",
    "dataset.augmentation.random_affine.scale",
]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    url: str
    state: str
    seed: int | None
    sweep_name: str | None
    val_acc: float | None
    val_loss: float | None
    positive_rate: float | None
    checkpoint_path: str | None
    config: dict[str, Any]

    @property
    def signature(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, nested_get(self.config, key)) for key in SIGNATURE_KEYS)

    @property
    def metric(self) -> float | None:
        return self.val_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank ChestMNIST CNN-like W&B runs by validation accuracy."
    )
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--run-project", default="chestmnist_cnn_val_acc")
    parser.add_argument("--study", default=None)
    parser.add_argument("--sweeps", nargs="*", default=DEFAULT_SWEEPS)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="*", default=[2024, 9001])
    parser.add_argument("--confirm-sweep", default="chestmnist_cnn_val_acc_confirm_top")
    parser.add_argument("--output-dir", default="outputs/chestmnist_cnn_val_acc")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = fetch_runs(
        entity=args.entity,
        project=args.project,
        run_project=args.run_project,
        study=args.study,
        sweeps=args.sweeps,
    )
    ranked = rank_runs(runs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ranked(output_dir / "ranked.csv", ranked)
    write_group_summary(output_dir / "groups.csv", ranked)
    write_markdown(output_dir / "ranked.md", ranked)

    print(f"ranked_runs={len(ranked)}")
    if ranked:
        best = ranked[0]
        print(
            "best="
            f"{best.metric:.6f} "
            f"loss={_format_float(best.val_loss)} "
            f"positive_rate={_format_float(best.positive_rate)} "
            f"run={best.name} "
            f"url={best.url}"
        )

    if args.confirm:
        commands = confirmation_commands(
            ranked,
            top_k=args.top_k,
            seeds=args.seeds,
            confirm_sweep=args.confirm_sweep,
        )
        write_commands(output_dir / "confirmation_commands.txt", commands)
        run_commands(commands, parallel=max(1, args.parallel), dry_run=args.dry_run)


def fetch_runs(
    *,
    entity: str,
    project: str,
    run_project: str,
    study: str | None,
    sweeps: list[str],
) -> list[RunRecord]:
    import wandb

    api = wandb.Api(timeout=30)
    filters: dict[str, Any] = {
        "config.run.project": run_project,
        "config.dataset.name": "chestmnist",
        "state": "finished",
    }
    if study is not None:
        filters["config.run.study"] = study

    raw_runs = api.runs(f"{entity}/{project}", filters=filters, per_page=300)
    records = []
    for run in raw_runs:
        config = dict(run.config or {})
        if nested_get(config, "run.project") != run_project:
            continue
        if nested_get(config, "dataset.name") != "chestmnist":
            continue
        sweep_name = nested_get(config, "run.sweep_name")
        if sweeps and sweep_name not in sweeps:
            continue
        summary = run.summary or {}
        records.append(
            RunRecord(
                run_id=run.id,
                name=run.name,
                url=run.url,
                state=run.state,
                seed=_optional_int(nested_get(config, "seed")),
                sweep_name=sweep_name,
                val_acc=_summary_float(summary, VAL_METRIC, "val/acc"),
                val_loss=_summary_float(summary, VAL_LOSS, "val/loss"),
                positive_rate=_summary_float(summary, POSITIVE_RATE, "val/pred_positive_rate"),
                checkpoint_path=_optional_str(
                    summary.get("evaluation/selected_checkpoint_path")
                    or summary.get("checkpoint/best_model_path")
                ),
                config=config,
            )
        )
    return records


def rank_runs(runs: Iterable[RunRecord]) -> list[RunRecord]:
    candidates = [run for run in runs if run.state == "finished" and run.metric is not None]
    return sorted(candidates, key=run_sort_key)


def confirmation_commands(
    ranked: list[RunRecord],
    *,
    top_k: int,
    seeds: list[int],
    confirm_sweep: str,
) -> list[list[str]]:
    selected = unique_signatures(ranked)[:top_k]
    commands = []
    for index, run in enumerate(selected):
        for seed in seeds:
            overrides = [
                f"experiment={experiment_for_run(run)}",
                f"seed={seed}",
                f"run.sweep_name={format_override_value(confirm_sweep)}",
                f"run.sweep_index={index}",
                "wandb.entity=tsilva",
                "wandb.project=dlab",
                "wandb.stable_id=false",
                "wandb.resume=never",
                "evaluation.test.enabled=false",
            ]
            for key, value in run.signature:
                if value is not None:
                    overrides.append(f"{key}={format_override_value(value)}")
            commands.append([sys.executable, "train.py", *overrides])
    return commands


def experiment_for_run(run: RunRecord) -> str:
    model_name = nested_get(run.config, "model.name")
    if model_name == "resnet18":
        return "chestmnist_resnet18_val_acc_search"
    return "chestmnist_cnn_val_acc_search"


def write_ranked(path: Path, ranked: list[RunRecord]) -> None:
    rows = [ranked_row(index, run) for index, run in enumerate(ranked, start=1)]
    write_csv(path, rows, fieldnames=list(rows[0]) if rows else ["rank"])


def write_group_summary(path: Path, ranked: list[RunRecord]) -> None:
    rows = group_rows(ranked)
    fieldnames = [
        "n",
        "mean_val_acc",
        "std_val_acc",
        "max_val_acc",
        "mean_val_loss",
        "mean_positive_rate",
        "seeds",
        "run_ids",
        *SIGNATURE_KEYS,
    ]
    write_csv(path, rows, fieldnames=fieldnames)


def write_markdown(path: Path, ranked: list[RunRecord]) -> None:
    groups = group_rows(ranked)
    lines = [
        "# ChestMNIST CNN Validation Accuracy Ranking",
        "",
        "Metric: `evaluation/selected/val/acc` when present, otherwise `val/acc`.",
        "Test metrics are intentionally excluded from selection.",
        "",
        "## Top Config Groups",
        "",
        (
            "| Rank | N | Mean Val Acc | Std | Max | Mean Val Loss | "
            "Mean Positive Rate | Model | Key Params | Runs |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(groups[:20], start=1):
        lines.append(
            f"| {index} | {row['n']} | {row['mean_val_acc']} | {row['std_val_acc']} | "
            f"{row['max_val_acc']} | {row['mean_val_loss']} | "
            f"{row['mean_positive_rate']} | {row.get('model.name', '')} | "
            f"`{config_label(row)}` | {row['run_ids']} |"
        )

    lines.extend(
        [
            "",
            "## Top Runs",
            "",
            "| Rank | Val Acc | Val Loss | Positive Rate | Seed | Sweep | Run |",
            "|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for rank, run in enumerate(ranked[:40], start=1):
        lines.append(
            f"| {rank} | {_format_float(run.val_acc)} | {_format_float(run.val_loss)} | "
            f"{_format_float(run.positive_rate)} | {run.seed} | {run.sweep_name} | "
            f"[{run.run_id}]({run.url}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ranked_row(rank: int, run: RunRecord) -> dict[str, str]:
    row = {
        "rank": str(rank),
        "run_id": run.run_id,
        "name": run.name,
        "url": run.url,
        "state": run.state,
        "seed": str(run.seed),
        "sweep_name": str(run.sweep_name),
        "val_acc": _format_float(run.val_acc),
        "val_loss": _format_float(run.val_loss),
        "positive_rate": _format_float(run.positive_rate),
        "checkpoint_path": run.checkpoint_path or "",
    }
    row.update({key: _stable_value(nested_get(run.config, key)) for key in SIGNATURE_KEYS})
    return row


def group_rows(ranked: list[RunRecord]) -> list[dict[str, str]]:
    rows = []
    for signature, runs in grouped_runs(ranked).items():
        metrics = [run.val_acc for run in runs if run.val_acc is not None]
        losses = [run.val_loss for run in runs if run.val_loss is not None]
        positive_rates = [run.positive_rate for run in runs if run.positive_rate is not None]
        if not metrics:
            continue
        row = {
            "n": str(len(metrics)),
            "mean_val_acc": f"{statistics.mean(metrics):.6f}",
            "std_val_acc": f"{statistics.stdev(metrics):.6f}" if len(metrics) > 1 else "",
            "max_val_acc": f"{max(metrics):.6f}",
            "mean_val_loss": f"{statistics.mean(losses):.6f}" if losses else "",
            "mean_positive_rate": (
                f"{statistics.mean(positive_rates):.6f}" if positive_rates else ""
            ),
            "seeds": " ".join(str(run.seed) for run in sorted(runs, key=seed_sort_key)),
            "run_ids": " ".join(run.run_id for run in runs),
        }
        row.update({key: _stable_value(value) for key, value in signature})
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (-float(row["mean_val_acc"]), float(row["mean_val_loss"] or 999.0)),
    )


def grouped_runs(ranked: list[RunRecord]) -> dict[tuple[tuple[str, str], ...], list[RunRecord]]:
    best_by_signature_seed: dict[tuple[tuple[tuple[str, str], ...], int | None], RunRecord] = {}
    for run in ranked:
        key = (_signature_key(run), run.seed)
        current = best_by_signature_seed.get(key)
        if current is None or run_sort_key(run) < run_sort_key(current):
            best_by_signature_seed[key] = run

    groups: dict[tuple[tuple[str, str], ...], list[RunRecord]] = defaultdict(list)
    for run in best_by_signature_seed.values():
        groups[_signature_key(run)].append(run)
    for runs in groups.values():
        runs.sort(key=run_sort_key)
    return groups


def unique_signatures(ranked: list[RunRecord]) -> list[RunRecord]:
    seen = set()
    unique = []
    for run in ranked:
        signature = _signature_key(run)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(run)
    return unique


def _signature_key(run: RunRecord) -> tuple[tuple[str, str], ...]:
    return tuple((key, _stable_value(value)) for key, value in run.signature)


def run_sort_key(run: RunRecord) -> tuple[float, float, float]:
    positive_rate_penalty = abs((run.positive_rate or 0.0) - 0.1)
    return (-(run.val_acc or 0.0), run.val_loss or 999.0, positive_rate_penalty)


def seed_sort_key(run: RunRecord) -> int:
    return -1 if run.seed is None else run.seed


def config_label(row: Mapping[str, str]) -> str:
    model = row.get("model.name")
    if model == "resnet18":
        return (
            f"resnet18 lr={row.get('optimizer.lr')} wd={row.get('optimizer.weight_decay')} "
            f"aug={row.get('dataset.augmentation.enabled')}"
        )
    return (
        f"cnn channels={row.get('model.params.channels')} "
        f"dropout={row.get('model.params.dropout')} "
        f"wd={row.get('optimizer.weight_decay')} "
        f"aug={row.get('dataset.augmentation.enabled')}"
    )


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


def write_commands(path: Path, commands: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(" ".join(command) + "\n")
    print(f"wrote {path}")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def nested_get(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _summary_float(summary: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _optional_float(summary.get(key))
        if value is not None:
            return value
    return None


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


def _optional_str(value: Any) -> str | None:
    if value in {None, "", "none", "null"}:
        return None
    return str(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _stable_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_stable_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ",".join(f"{key}:{_stable_value(value[key])}" for key in sorted(value))
        return "{" + items + "}"
    return str(value)


if __name__ == "__main__":
    main()
