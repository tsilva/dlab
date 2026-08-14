from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

from src.datasets import build_datamodule
from src.models import build_model
from src.trainers import ResearchLitModule


@dataclass(frozen=True)
class ThresholdResult:
    label: str
    config_path: str
    checkpoint_path: str
    val_acc_at_0p5: float
    val_positive_rate_at_0p5: float
    best_scalar_threshold: float
    best_scalar_val_acc: float
    best_scalar_positive_rate: float
    best_per_class_val_acc: float
    best_per_class_positive_rate: float
    per_class_thresholds: list[float]
    per_class_acc: list[float]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune scalar and per-class ChestMNIST thresholds on the validation split "
            "for one or more existing checkpoints."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        help="Candidate label, resolved config.yaml path, and checkpoint path.",
    )
    parser.add_argument(
        "--wandb-run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_ID"),
        help="Candidate label and W&B run id. Downloads the run-output artifact.",
    )
    parser.add_argument("--entity", default="tsilva")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--output-dir", default="outputs/chestmnist_cnn_val_acc/threshold_tuning")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if not args.candidate and not args.wandb_run:
        parser.error("At least one --candidate or --wandb-run is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    local_candidates = [
        (label, Path(config_path), Path(checkpoint_path))
        for label, config_path, checkpoint_path in args.candidate or []
    ]
    if args.wandb_run:
        artifact_root = output_dir / "artifacts"
        for label, run_id in args.wandb_run:
            local_candidates.append(
                download_wandb_candidate(
                    label=label,
                    run_id=run_id,
                    entity=args.entity,
                    project=args.project,
                    artifact_root=artifact_root,
                )
            )

    results = [
        evaluate_candidate(label, Path(config_path), Path(checkpoint_path), device)
        for label, config_path, checkpoint_path in local_candidates
    ]
    results = sorted(results, key=lambda item: item.best_per_class_val_acc, reverse=True)

    write_json(output_dir / "threshold_tuning.json", results)
    write_csv(output_dir / "threshold_tuning.csv", results)
    if results:
        best = results[0]
        print(
            "best="
            f"{best.label} "
            f"per_class_val_acc={best.best_per_class_val_acc:.6f} "
            f"scalar_threshold={best.best_scalar_threshold:.6f} "
            f"scalar_val_acc={best.best_scalar_val_acc:.6f} "
            f"positive_rate={best.best_per_class_positive_rate:.6f}"
        )


def evaluate_candidate(
    label: str,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> ThresholdResult:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config path: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint path: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.dataset.pop("target_type", None)
        cfg.dataset.num_workers = 0
        cfg.dataset.pin_memory = False
    datamodule = build_datamodule(cfg.dataset, seed=int(cfg.seed))
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = build_model(cfg.model, datamodule.info)
    lit_module = ResearchLitModule(model, cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lit_module.load_state_dict(checkpoint["state_dict"])
    lit_module.to(device)
    lit_module.eval()

    probs, targets = collect_validation_probs(lit_module, datamodule.val_dataloader(), device)
    targets_bool = targets.bool()

    at_0p5_preds = probs > 0.5
    scalar_threshold, scalar_acc, scalar_positive_rate = best_scalar_threshold(probs, targets_bool)
    class_thresholds, class_acc = best_per_class_thresholds(probs, targets_bool)
    per_class_preds = probs > class_thresholds.view(1, -1)

    return ThresholdResult(
        label=label,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        val_acc_at_0p5=accuracy(at_0p5_preds, targets_bool),
        val_positive_rate_at_0p5=float(at_0p5_preds.float().mean().item()),
        best_scalar_threshold=float(scalar_threshold),
        best_scalar_val_acc=float(scalar_acc),
        best_scalar_positive_rate=float((probs > scalar_threshold).float().mean().item()),
        best_per_class_val_acc=accuracy(per_class_preds, targets_bool),
        best_per_class_positive_rate=float(per_class_preds.float().mean().item()),
        per_class_thresholds=[float(item) for item in class_thresholds.tolist()],
        per_class_acc=[float(item) for item in class_acc.tolist()],
    )


def download_wandb_candidate(
    *,
    label: str,
    run_id: str,
    entity: str,
    project: str,
    artifact_root: Path,
) -> tuple[str, Path, Path]:
    import wandb

    api = wandb.Api(timeout=30)
    run = api.run(f"{entity}/{project}/{run_id}")
    output_artifacts = [
        artifact for artifact in run.logged_artifacts() if artifact.type == "run-output"
    ]
    if len(output_artifacts) != 1:
        raise ValueError(
            f"Expected exactly one run-output artifact for {run_id}, "
            f"found {len(output_artifacts)}."
        )

    artifact_dir = Path(
        output_artifacts[0].download(root=str(artifact_root / f"{label}-{run_id}"))
    )
    config_path = artifact_dir / "config.yaml"
    checkpoint_summary = (
        run.summary.get("evaluation/selected_checkpoint_path")
        or run.summary.get("checkpoint/best_model_path")
    )
    if checkpoint_summary:
        checkpoint_path = artifact_dir / "checkpoints" / Path(str(checkpoint_summary)).name
    else:
        checkpoint_matches = sorted((artifact_dir / "checkpoints").glob("*.ckpt"))
        if len(checkpoint_matches) != 1:
            raise ValueError(
                f"Unable to infer checkpoint for {run_id}; found {len(checkpoint_matches)}."
            )
        checkpoint_path = checkpoint_matches[0]
    return label, config_path, checkpoint_path


@torch.inference_mode()
def collect_validation_probs(
    lit_module: ResearchLitModule,
    dataloader: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    for x, y in dataloader:
        logits = lit_module.model(x.to(device))
        probs_batches.append(logits.sigmoid().cpu())
        target_batches.append(y.cpu())
    if not probs_batches:
        raise ValueError("Validation dataloader produced no batches.")
    return torch.cat(probs_batches, dim=0), torch.cat(target_batches, dim=0)


def best_scalar_threshold(probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    threshold, correct = best_threshold_for_binary_targets(probs.flatten(), targets.flatten())
    preds = probs > threshold
    return threshold, accuracy(preds, targets), float(preds.float().mean().item())


def best_per_class_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    thresholds = []
    accuracies = []
    for class_index in range(probs.shape[1]):
        threshold, _ = best_threshold_for_binary_targets(
            probs[:, class_index],
            targets[:, class_index],
        )
        preds = probs[:, class_index] > threshold
        thresholds.append(threshold)
        accuracies.append(accuracy(preds, targets[:, class_index]))
    return torch.tensor(thresholds), torch.tensor(accuracies)


def best_threshold_for_binary_targets(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, int]:
    order = torch.argsort(probs)
    sorted_probs = probs[order]
    sorted_targets = targets[order].to(torch.int64)
    total_pos = int(sorted_targets.sum().item())
    total_count = int(sorted_targets.numel())
    total_neg = total_count - total_pos

    best_threshold = 0.0
    best_correct = total_pos

    unique_values, counts = torch.unique_consecutive(sorted_probs, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0) - 1
    cumulative_pos = torch.cumsum(sorted_targets, dim=0)
    cumulative_neg = torch.cumsum(1 - sorted_targets, dim=0)
    for value, end_index in zip(unique_values, group_ends, strict=True):
        end = int(end_index.item())
        true_negatives = int(cumulative_neg[end].item())
        true_positives = total_pos - int(cumulative_pos[end].item())
        correct = true_negatives + true_positives
        if correct > best_correct:
            best_correct = correct
            best_threshold = float(value.item())

    if total_neg > best_correct:
        best_correct = total_neg
        best_threshold = 1.0
    return best_threshold, best_correct


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float((preds == targets).float().mean().item())


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, results: list[ThresholdResult]) -> None:
    path.write_text(json.dumps([asdict(item) for item in results], indent=2) + "\n")
    print(f"wrote {path}")


def write_csv(path: Path, results: list[ThresholdResult]) -> None:
    fieldnames = [
        "rank",
        "label",
        "val_acc_at_0p5",
        "val_positive_rate_at_0p5",
        "best_scalar_threshold",
        "best_scalar_val_acc",
        "best_scalar_positive_rate",
        "best_per_class_val_acc",
        "best_per_class_positive_rate",
        "per_class_thresholds",
        "per_class_acc",
        "config_path",
        "checkpoint_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = asdict(result)
            row["rank"] = rank
            row["per_class_thresholds"] = json.dumps(row["per_class_thresholds"])
            row["per_class_acc"] = json.dumps(row["per_class_acc"])
            writer.writerow(row)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
