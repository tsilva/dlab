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

VAL_METRIC = "evaluation/selected/val/acc"
TEST_METRIC = "test/acc"
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
    sweep_name: str | None
    sweep_index: int | None
    val_acc: float | None
    val_loss: float | None
    test_acc: float | None
    test_loss: float | None
    checkpoint_path: str | None
    config: dict[str, Any]

    @property
    def signature(self) -> tuple[tuple[str, Any], ...]:
        return tuple((key, nested_get(self.config, key)) for key in CONFIG_KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and rank MNIST RNN test evals.")
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--run-project", default="mnist_rnn_val_acc")
    parser.add_argument("--study", default="001_rnn_architecture_search")
    parser.add_argument("--confirm-sweep", default="mnist_rnn_val_acc_confirm_top")
    parser.add_argument("--eval-sweep", default="mnist_rnn_val_acc_test_eval_top")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/mnist_rnn_val_acc_test_eval")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = fetch_runs(args.entity, args.project, args.run_project, args.study)
    candidates = select_candidate_runs(runs, args.confirm_sweep, args.top_k)
    write_candidate_summary(output_dir, candidates)
    print(f"candidate_runs={len(candidates)}")

    if args.evaluate:
        commands = evaluation_commands(
            candidates,
            entity=args.entity,
            project=args.project,
            eval_sweep=args.eval_sweep,
        )
        write_commands(output_dir / "test_eval_commands.txt", commands)
        run_commands(commands, parallel=max(1, args.parallel), dry_run=args.dry_run)

    refreshed = fetch_runs(args.entity, args.project, args.run_project, args.study)
    eval_runs = [
        run
        for run in refreshed
        if run.sweep_name == args.eval_sweep and run.test_acc is not None
    ]
    write_test_summary(output_dir, eval_runs)
    if eval_runs:
        best = rank_groups(eval_runs, TEST_METRIC)[0]
        print(
            "best_test_mean="
            f"{best['mean_test_acc']:.6f} "
            f"config={config_label(dict(best['signature']))}"
        )


def fetch_runs(entity: str, project: str, run_project: str, study: str) -> list[RunRecord]:
    import wandb

    api = wandb.Api()
    raw_runs = list(
        api.runs(
            f"{entity}/{project}",
            filters={"config.run.project": run_project, "config.run.study": study},
            per_page=300,
        )
    )
    records = []
    for run in raw_runs:
        config = dict(run.config or {})
        if nested_get(config, "run.project") != run_project:
            continue
        if nested_get(config, "run.study") != study:
            continue
        run_cfg = nested_get(config, "run", {}) or {}
        records.append(
            RunRecord(
                name=run.name,
                url=run.url,
                state=run.state,
                seed=_optional_int(nested_get(config, "seed")),
                sweep_name=run_cfg.get("sweep_name"),
                sweep_index=_optional_int(run_cfg.get("sweep_index")),
                val_acc=_optional_float(run.summary.get(VAL_METRIC)),
                val_loss=_optional_float(run.summary.get("evaluation/selected/val/loss")),
                test_acc=_optional_float(run.summary.get(TEST_METRIC)),
                test_loss=_optional_float(run.summary.get("test/loss")),
                checkpoint_path=_optional_str(
                    run.summary.get("evaluation/selected_checkpoint_path")
                    or run.summary.get("checkpoint/best_model_path")
                ),
                config=config,
            )
        )
    return records


def select_candidate_runs(
    runs: Iterable[RunRecord],
    confirm_sweep: str,
    top_k: int,
) -> list[RunRecord]:
    train_runs = [
        run
        for run in runs
        if run.state == "finished"
        and run.val_acc is not None
        and run.checkpoint_path
        and run.sweep_name != "mnist_rnn_val_acc_test_eval_top"
    ]
    grouped: dict[tuple[tuple[str, Any], ...], list[RunRecord]] = defaultdict(list)
    for run in train_runs:
        grouped[run.signature].append(run)

    groups = []
    for signature, group_runs in grouped.items():
        values = [float(run.val_acc) for run in group_runs if run.val_acc is not None]
        if len(values) < 3:
            continue
        groups.append(
            {
                "signature": signature,
                "runs": sorted(group_runs, key=lambda item: int(item.seed or -1)),
                "mean_val_acc": sum(values) / len(values),
                "best_val_acc": max(values),
            }
        )
    groups.sort(key=lambda item: (-item["mean_val_acc"], -item["best_val_acc"]))
    selected = []
    for group in groups[:top_k]:
        for run in group["runs"]:
            selected.append(run)
    return selected


def evaluation_commands(
    candidates: list[RunRecord],
    entity: str,
    project: str,
    eval_sweep: str,
) -> list[list[str]]:
    signature_rank = {
        signature: index
        for index, signature in enumerate(
            dict.fromkeys(run.signature for run in candidates)
        )
    }
    commands = []
    for run in candidates:
        artifact = f"{entity}/{project}/{run.name}-run:latest"
        checkpoint_file = checkpoint_artifact_file(run)
        overrides = [
            "experiment=mnist_rnn_val_acc_search",
            f"seed={run.seed}",
            "run.stage=05_test_evaluation",
            (
                "run.question='What is the held-out test performance of the "
                "validation-confirmed MNIST RNN candidate?'"
            ),
            (
                "run.goal='Audit validation-confirmed RNN candidates on the MNIST "
                "test split after selection.'"
            ),
            (
                "run.expected_pattern='Test accuracy should broadly track three-seed "
                "selected validation accuracy; if not, treat the result as an audit, "
                "not a clean selection signal.'"
            ),
            f"run.sweep_name={format_override_value(eval_sweep)}",
            f"run.sweep_index={signature_rank[run.signature]}",
            "wandb.entity=tsilva",
            "wandb.project=dlab",
            "wandb.stable_id=false",
            "wandb.resume=never",
            "wandb.log_artifacts=false",
            "evaluation.only=true",
            "evaluation.selection.enabled=true",
            "evaluation.test.enabled=true",
            f"evaluation.checkpoint.artifact={format_override_value(artifact)}",
            f"evaluation.checkpoint.file={format_override_value(checkpoint_file)}",
            "trainer.enable_checkpointing=false",
            "early_stopping.enabled=false",
            "gradient_flow.enabled=false",
            "sequence_diagnostics.enabled=true",
            "reports.enabled=true",
        ]
        for key, value in run.signature:
            if value is not None:
                overrides.append(f"{key}={format_override_value(value)}")
        commands.append([sys.executable, "train.py", *overrides])
    return commands


def checkpoint_artifact_file(run: RunRecord) -> str:
    if not run.checkpoint_path:
        raise ValueError(f"Run has no checkpoint path: {run.name}")
    checkpoint_name = Path(run.checkpoint_path).name
    return f"checkpoints/{checkpoint_name}"


def rank_groups(runs: list[RunRecord], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, Any], ...], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.signature].append(run)
    rows = []
    for signature, group_runs in grouped.items():
        test_values = [float(run.test_acc) for run in group_runs if run.test_acc is not None]
        val_values = [float(run.val_acc) for run in group_runs if run.val_acc is not None]
        if not test_values:
            continue
        rows.append(
            {
                "signature": signature,
                "runs": sorted(group_runs, key=lambda item: int(item.seed or -1)),
                "mean_test_acc": sum(test_values) / len(test_values),
                "best_test_acc": max(test_values),
                "mean_val_acc": sum(val_values) / len(val_values) if val_values else None,
                "n": len(test_values),
            }
        )
    rows.sort(key=lambda item: (-item["mean_test_acc"], -item["best_test_acc"]))
    return rows


def write_candidate_summary(output_dir: Path, candidates: list[RunRecord]) -> None:
    path = output_dir / "candidate_runs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "url",
                "seed",
                "val_acc",
                "val_loss",
                "artifact",
                "checkpoint_file",
                *summary_config_keys(),
            ],
        )
        writer.writeheader()
        for run in candidates:
            row = {
                "run": run.name,
                "url": run.url,
                "seed": run.seed,
                "val_acc": _format_float(run.val_acc),
                "val_loss": _format_float(run.val_loss),
                "artifact": f"tsilva/dlab/{run.name}-run:latest",
                "checkpoint_file": checkpoint_artifact_file(run),
            }
            row.update(summary_config(run))
            writer.writerow(row)
    print(f"wrote {path}")


def write_test_summary(output_dir: Path, eval_runs: list[RunRecord]) -> None:
    csv_path = output_dir / "test_eval_runs.csv"
    group_path = output_dir / "test_eval_group_summary.md"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "url",
                "seed",
                "test_acc",
                "test_loss",
                "selected_val_acc",
                "selected_val_loss",
                *summary_config_keys(),
            ],
        )
        writer.writeheader()
        for run in sorted(eval_runs, key=lambda item: (-float(item.test_acc or 0), item.name)):
            row = {
                "run": run.name,
                "url": run.url,
                "seed": run.seed,
                "test_acc": _format_float(run.test_acc),
                "test_loss": _format_float(run.test_loss),
                "selected_val_acc": _format_float(run.val_acc),
                "selected_val_loss": _format_float(run.val_loss),
            }
            row.update(summary_config(run))
            writer.writerow(row)
    groups = rank_groups(eval_runs, TEST_METRIC)
    with group_path.open("w") as handle:
        handle.write("# MNIST RNN Test Evaluation Summary\n\n")
        handle.write("| Rank | Mean Test Acc | Best Test Acc | N | Config |\n")
        handle.write("|---:|---:|---:|---:|---|\n")
        for index, group in enumerate(groups, start=1):
            handle.write(
                f"| {index} | {group['mean_test_acc']:.6f} | "
                f"{group['best_test_acc']:.6f} | "
                f"{group['n']} | "
                f"`{config_label(dict(group['signature']))}` |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {group_path}")


def write_commands(path: Path, commands: list[list[str]]) -> None:
    with path.open("w") as handle:
        for command in commands:
            handle.write(" ".join(command) + "\n")
    print(f"wrote {path}")


def run_commands(commands: list[list[str]], parallel: int, dry_run: bool) -> None:
    if dry_run:
        for command in commands:
            print(" ".join(command))
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


def summary_config(run: RunRecord) -> dict[str, Any]:
    config = dict(run.signature)
    return {summary_key(key): value for key, value in config.items()}


def summary_config_keys() -> list[str]:
    return [summary_key(key) for key in CONFIG_KEYS]


def summary_key(key: str) -> str:
    return {
        "model.params.rnn_type": "rnn_type",
        "model.params.hidden_dim": "hidden_dim",
        "model.params.num_layers": "num_layers",
        "model.params.dropout": "dropout",
        "model.params.bidirectional": "bidirectional",
        "model.params.pooling": "pooling",
        "model.params.sequence_axis": "sequence_axis",
        "optimizer.lr": "lr",
        "optimizer.weight_decay": "weight_decay",
        "optimizer.scheduler.name": "scheduler",
        "loss.label_smoothing": "label_smoothing",
        "dataset.batch_size": "batch_size",
        "dataset.augmentation.enabled": "augmentation",
        "trainer.gradient_clip_val": "gradient_clip_val",
    }[key]


def config_label(config: Mapping[str, Any]) -> str:
    return (
        f"{config.get('model.params.rnn_type')} "
        f"w{config.get('model.params.hidden_dim')} "
        f"d{config.get('model.params.num_layers')} "
        f"bidir={config.get('model.params.bidirectional')} "
        f"pool={config.get('model.params.pooling')} "
        f"axis={config.get('model.params.sequence_axis')} "
        f"lr={config.get('optimizer.lr')} "
        f"wd={config.get('optimizer.weight_decay')} "
        f"sched={config.get('optimizer.scheduler.name')} "
        f"do={config.get('model.params.dropout')} "
        f"ls={config.get('loss.label_smoothing')} "
        f"bs={config.get('dataset.batch_size')} "
        f"aug={config.get('dataset.augmentation.enabled')} "
        f"clip={config.get('trainer.gradient_clip_val')}"
    )


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
