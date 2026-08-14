from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.execution.launchers import format_override_value

VAL_METRIC = "evaluation/selected/val/acc"
VAL_LOSS = "evaluation/selected/val/loss"
TEST_METRIC = "test/acc"
TEST_LOSS = "test/loss"
CONFIG_KEYS = [
    "model.params.channels",
    "model.params.convs_per_stage",
    "model.params.batch_norm",
    "model.params.residual",
    "model.params.dropout",
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
    name: str
    url: str
    state: str
    seed: int | None
    sweep_name: str | None
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
    parser = argparse.ArgumentParser(
        description="Run and rank Fashion-MNIST CNN held-out test audits."
    )
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--run-project", default="fashion_mnist_cnn_val_acc")
    parser.add_argument("--study", default="001_wide_cnn_val_acc_selection")
    parser.add_argument("--confirm-sweep", default="fashion_mnist_cnn_val_acc_confirm_3seed_sweep")
    parser.add_argument("--eval-sweep", default="fashion_mnist_cnn_val_acc_test_eval_top")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/fashion_mnist_cnn_val_acc_test_eval")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = fetch_runs(args.entity, args.project, args.run_project, args.study)
    candidates = select_candidate_runs(runs, args.confirm_sweep, args.top_k)
    write_candidate_summary(output_dir, candidates, args.entity, args.project)
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
        run for run in refreshed if run.sweep_name == args.eval_sweep and run.test_acc is not None
    ]
    write_test_summary(output_dir, eval_runs)
    if eval_runs:
        best = rank_groups(eval_runs)[0]
        print(
            "best_test_mean="
            f"{best['mean_test_acc']:.6f} "
            f"config={config_label(dict(best['signature']))}"
        )


def fetch_runs(entity: str, project: str, run_project: str, study: str) -> list[RunRecord]:
    import wandb

    api = wandb.Api(timeout=30)
    raw_runs = api.runs(
        f"{entity}/{project}",
        filters={"config.run.project": run_project, "config.run.study": study},
        per_page=200,
    )
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
                sweep_name=nested_get(config, "run.sweep_name"),
                val_acc=_optional_float(run.summary.get(VAL_METRIC, run.summary.get("val/acc"))),
                val_loss=_optional_float(run.summary.get(VAL_LOSS, run.summary.get("val/loss"))),
                test_acc=_optional_float(run.summary.get(TEST_METRIC)),
                test_loss=_optional_float(run.summary.get(TEST_LOSS)),
                checkpoint_path=_optional_str(
                    run.summary.get("evaluation/selected_checkpoint_path")
                    or run.summary.get("checkpoint/best_model_path")
                ),
                config=config,
            )
        )
    return records


def select_candidate_runs(
    runs: list[RunRecord],
    confirm_sweep: str,
    top_k: int,
) -> list[RunRecord]:
    train_runs = [
        run
        for run in runs
        if run.state == "finished"
        and run.sweep_name == confirm_sweep
        and run.val_acc is not None
        and run.checkpoint_path
    ]
    grouped: dict[tuple[tuple[str, Any], ...], list[RunRecord]] = defaultdict(list)
    for run in train_runs:
        grouped[run.signature].append(run)

    groups = []
    for signature, group_runs in grouped.items():
        values = [float(run.val_acc) for run in group_runs if run.val_acc is not None]
        losses = [float(run.val_loss) for run in group_runs if run.val_loss is not None]
        if len(values) < 3:
            continue
        groups.append(
            {
                "signature": signature,
                "runs": sorted(group_runs, key=lambda item: int(item.seed or -1)),
                "mean_val_acc": sum(values) / len(values),
                "mean_val_loss": sum(losses) / len(losses) if losses else 999.0,
                "best_val_acc": max(values),
            }
        )
    groups.sort(
        key=lambda item: (
            -float(item["mean_val_acc"]),
            float(item["mean_val_loss"]),
            -float(item["best_val_acc"]),
        )
    )

    selected = []
    for group in groups[:top_k]:
        selected.extend(group["runs"])
    return selected


def evaluation_commands(
    candidates: list[RunRecord],
    entity: str,
    project: str,
    eval_sweep: str,
) -> list[list[str]]:
    signature_rank = {
        signature: index
        for index, signature in enumerate(dict.fromkeys(run.signature for run in candidates))
    }
    commands = []
    for run in candidates:
        artifact = f"{entity}/{project}/{run.name}-run:latest"
        checkpoint_file = checkpoint_artifact_file(run)
        question = (
            "What is the held-out test performance of the validation-confirmed "
            "Fashion-MNIST CNN candidate?"
        )
        goal = (
            "Audit the selected Fashion-MNIST CNN on the test split after "
            "validation selection."
        )
        expected_pattern = (
            "Test accuracy should broadly track selected validation accuracy; "
            "treat surprises as audit findings, not hyperparameter search signals."
        )
        overrides = [
            "experiment=fashion_mnist_cnn_val_acc_wide",
            f"seed={run.seed}",
            "run.stage=05_test_evaluation",
            f"run.question={format_override_value(question)}",
            f"run.goal={format_override_value(goal)}",
            f"run.expected_pattern={format_override_value(expected_pattern)}",
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
    return f"checkpoints/{Path(run.checkpoint_path).name}"


def rank_groups(runs: list[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[tuple[str, Any], ...], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[run.signature].append(run)

    rows = []
    for signature, group_runs in grouped.items():
        test_values = [float(run.test_acc) for run in group_runs if run.test_acc is not None]
        test_losses = [float(run.test_loss) for run in group_runs if run.test_loss is not None]
        val_values = [float(run.val_acc) for run in group_runs if run.val_acc is not None]
        if not test_values:
            continue
        rows.append(
            {
                "signature": signature,
                "runs": sorted(group_runs, key=lambda item: int(item.seed or -1)),
                "mean_test_acc": sum(test_values) / len(test_values),
                "mean_test_loss": sum(test_losses) / len(test_losses) if test_losses else None,
                "best_test_acc": max(test_values),
                "mean_val_acc": sum(val_values) / len(val_values) if val_values else None,
                "n": len(test_values),
            }
        )
    rows.sort(key=lambda item: (-item["mean_test_acc"], item["mean_test_loss"] or 999.0))
    return rows


def write_candidate_summary(
    output_dir: Path,
    candidates: list[RunRecord],
    entity: str,
    project: str,
) -> None:
    path = output_dir / "candidate_runs.csv"
    fieldnames = [
        "run",
        "url",
        "seed",
        "val_acc",
        "val_loss",
        "artifact",
        "checkpoint_file",
        *summary_config_keys(),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in candidates:
            row = {
                "run": run.name,
                "url": run.url,
                "seed": run.seed,
                "val_acc": _format_float(run.val_acc),
                "val_loss": _format_float(run.val_loss),
                "artifact": f"{entity}/{project}/{run.name}-run:latest",
                "checkpoint_file": checkpoint_artifact_file(run),
            }
            row.update(summary_config(run))
            writer.writerow(row)
    print(f"wrote {path}")


def write_test_summary(output_dir: Path, eval_runs: list[RunRecord]) -> None:
    csv_path = output_dir / "test_eval_runs.csv"
    group_path = output_dir / "test_eval_group_summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
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

    groups = rank_groups(eval_runs)
    with group_path.open("w", encoding="utf-8") as handle:
        handle.write("# Fashion-MNIST CNN Test Evaluation Summary\n\n")
        handle.write("| Rank | Mean Test Acc | Mean Test Loss | Best Test Acc | N | Config |\n")
        handle.write("|---:|---:|---:|---:|---:|---|\n")
        for index, group in enumerate(groups, start=1):
            handle.write(
                f"| {index} | {group['mean_test_acc']:.6f} | "
                f"{_format_float(group['mean_test_loss'])} | "
                f"{group['best_test_acc']:.6f} | {group['n']} | "
                f"`{config_label(dict(group['signature']))}` |\n"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {group_path}")


def write_commands(path: Path, commands: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
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
    return key.replace(".", "_")


def config_label(config: dict[str, Any]) -> str:
    channels = config.get("model.params.channels")
    dropout = config.get("model.params.dropout")
    weight_decay = config.get("optimizer.weight_decay")
    monitor = config.get("checkpoint.monitor")
    return f"channels={channels}, dropout={dropout}, wd={weight_decay}, monitor={monitor}"


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
