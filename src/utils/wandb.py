from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


def wandb_tags(cfg: DictConfig) -> list[str]:
    tags = [
        str(cfg.task),
        str(cfg.dataset.name),
        str(cfg.model.name),
        str(cfg.optimizer.name),
        *_as_list(cfg.run.get("tags", [])),
    ]
    for key in ("stage", "study", "sweep_name"):
        value = cfg.run.get(key)
        if value:
            tags.append(str(value))
    return sorted({tag for tag in tags if tag})


def wandb_notes(cfg: DictConfig) -> str | None:
    lines = []
    for label, key in (
        ("Goal", "goal"),
        ("Question", "question"),
        ("Hypothesis", "hypothesis"),
        ("Expected pattern", "expected_pattern"),
    ):
        value = cfg.run.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines) if lines else None


def parameter_count(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"params/trainable": trainable, "params/total": total}


def summarize_training_run(
    trainer: Any,
    model: torch.nn.Module,
    elapsed_seconds: float,
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "runtime/seconds": elapsed_seconds,
        "trainer/global_step": int(getattr(trainer, "global_step", 0)),
        "trainer/current_epoch": int(getattr(trainer, "current_epoch", 0)),
        **parameter_count(model),
    }
    if elapsed_seconds > 0:
        summary["runtime/steps_per_second"] = summary["trainer/global_step"] / elapsed_seconds

    callback_metrics = getattr(trainer, "callback_metrics", {})
    train_loss = _metric_value(callback_metrics.get("train/loss_epoch"))
    val_loss = _metric_value(callback_metrics.get("val/loss"))
    if train_loss is not None and val_loss is not None:
        summary["generalization/loss_gap"] = val_loss - train_loss

    checkpoint = getattr(trainer, "checkpoint_callback", None)
    if checkpoint is not None:
        best_score = _metric_value(getattr(checkpoint, "best_model_score", None))
        if best_score is not None:
            summary["checkpoint/best_score"] = best_score
        best_path = getattr(checkpoint, "best_model_path", "")
        if best_path:
            summary["checkpoint/best_model_path"] = best_path
    return summary


def log_wandb_post_run(
    cfg: DictConfig,
    trainer: Any,
    lit_module: torch.nn.Module,
    datamodule: Any,
    run_dir: str | Path,
    report_path: str | None,
    elapsed_seconds: float,
) -> None:
    if not cfg.wandb.get("enabled", False):
        return
    wandb_run = _active_wandb_run()
    if wandb_run is None:
        return

    run_dir = Path(run_dir)
    summary = summarize_training_run(trainer, lit_module.model, elapsed_seconds)
    for key, value in summary.items():
        wandb_run.summary[key] = value

    if cfg.wandb.get("log_tables", True):
        _log_example_table(cfg, wandb_run, lit_module, datamodule)

    if cfg.wandb.get("log_artifacts", True):
        _log_run_artifact(cfg, wandb_run, run_dir, report_path)


def _log_example_table(
    cfg: DictConfig,
    wandb_run: Any,
    lit_module: torch.nn.Module,
    datamodule: Any,
) -> None:
    try:
        import wandb
    except ImportError:
        return

    max_examples = int(cfg.wandb.get("table_max_examples", 32))
    if max_examples <= 0:
        return

    datamodule.setup("predict")
    dataloader = datamodule.test_dataloader()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return

    x, y = batch
    x = x[:max_examples].to(lit_module.device)
    y = y[:max_examples]
    model = lit_module.model
    model.eval()
    with torch.no_grad():
        output = model(x)

    if cfg.task == "classification":
        logits = output
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu()
        conf = probs.max(dim=1).values.cpu()
        table = wandb.Table(columns=["index", "image", "label", "prediction", "confidence"])
        for index, image in enumerate(_image_batch_for_wandb(cfg, x).cpu()):
            table.add_data(
                index,
                wandb.Image(image),
                int(y[index].item()),
                int(preds[index].item()),
                float(conf[index].item()),
            )
        wandb_run.log({"examples/predictions": table})
        return

    recon = output["recon"].detach().clamp(0, 1)
    originals = _image_batch_for_wandb(cfg, x).cpu()
    table = wandb.Table(columns=["index", "input", "reconstruction", "mse"])
    for index in range(min(len(originals), len(recon))):
        mse = torch.mean((recon[index].cpu() - originals[index]) ** 2)
        table.add_data(
            index,
            wandb.Image(originals[index]),
            wandb.Image(recon[index].cpu()),
            float(mse),
        )
    wandb_run.log({"examples/reconstructions": table})


def _log_run_artifact(
    cfg: DictConfig,
    wandb_run: Any,
    run_dir: Path,
    report_path: str | None,
) -> None:
    try:
        import wandb
    except ImportError:
        return

    artifact = wandb.Artifact(
        name=f"{cfg.experiment_name}-run",
        type="run-output",
        metadata=OmegaConf.to_container(cfg, resolve=True),
    )
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        artifact.add_file(str(config_path), name="config.yaml")
    metrics_matches = sorted(run_dir.glob("**/metrics.csv"))
    for metrics_path in metrics_matches:
        artifact.add_file(str(metrics_path), name=f"metrics/{metrics_path.name}")
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        artifact.add_dir(str(checkpoint_dir), name="checkpoints")
    if report_path and Path(report_path).exists():
        artifact.add_file(report_path, name=Path(report_path).name)
    wandb_run.log_artifact(artifact)


def _active_wandb_run() -> Any | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run


def _image_batch_for_wandb(cfg: DictConfig, x: torch.Tensor) -> torch.Tensor:
    if cfg.dataset.get("normalize", True):
        return _unnormalize_if_needed(x.detach().clamp(-3, 3), cfg.dataset.name)
    return x.detach().clamp(0, 1)


def _unnormalize_if_needed(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if dataset_name == "cifar10":
        mean = torch.tensor((0.4914, 0.4822, 0.4465), device=x.device).view(1, 3, 1, 1)
        std = torch.tensor((0.247, 0.243, 0.261), device=x.device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor((0.1307,), device=x.device).view(1, 1, 1, 1)
        std = torch.tensor((0.3081,), device=x.device).view(1, 1, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _metric_value(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list | tuple | ListConfig):
        return list(value)
    return [value]
