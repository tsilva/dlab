from __future__ import annotations

import hashlib
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
    return sorted({_wandb_tag(tag) for tag in tags if tag})


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


def _wandb_tag(tag: str, max_length: int = 64) -> str:
    if len(tag) <= max_length:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    prefix_length = max_length - len(digest) - 1
    return f"{tag[:prefix_length]}-{digest}"


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
    for key, value in callback_metrics.items():
        metric = _metric_value(value)
        if metric is not None:
            summary[str(key)] = metric

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
    for callback in getattr(trainer, "callbacks", []):
        if callback.__class__.__name__ != "EarlyStopping":
            continue
        stopped_epoch = int(getattr(callback, "stopped_epoch", 0))
        summary["early_stopping/stopped_epoch"] = stopped_epoch
        summary["early_stopping/stopped"] = int(stopped_epoch > 0)
        summary["early_stopping/wait_count"] = int(getattr(callback, "wait_count", 0))
        summary["early_stopping/patience"] = int(getattr(callback, "patience", 0))
        break
    return summary


def log_wandb_post_run(
    cfg: DictConfig,
    trainer: Any,
    lit_module: torch.nn.Module,
    datamodule: Any,
    run_dir: str | Path,
    report_path: str | None,
    elapsed_seconds: float,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    if not cfg.wandb.get("enabled", False):
        return
    wandb_run = _active_wandb_run()
    if wandb_run is None:
        return

    run_dir = Path(run_dir)
    summary = summarize_training_run(trainer, lit_module.model, elapsed_seconds)
    if extra_summary:
        summary.update(extra_summary)
    for key, value in summary.items():
        wandb_run.summary[key] = value

    if cfg.wandb.get("log_tables", True):
        _log_example_table(cfg, wandb_run, lit_module, datamodule)
        if cfg.get("evaluation", {}).get("error_analysis", {}).get("enabled", False):
            _log_error_analysis_table(cfg, wandb_run, lit_module, datamodule)

    if cfg.wandb.get("log_artifacts", True):
        _log_run_artifact(cfg, wandb_run, run_dir, report_path)

    wandb_run.finish()


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

    split = str(cfg.wandb.get("table_split", "val"))
    if split == "val":
        datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError("wandb.table_split must be one of: val, test")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return

    model = lit_module.model
    device = _model_device(model, fallback=lit_module.device)
    x, y = batch
    x = x[:max_examples].to(device)
    y = y[:max_examples]
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


def _log_error_analysis_table(
    cfg: DictConfig,
    wandb_run: Any,
    lit_module: torch.nn.Module,
    datamodule: Any,
) -> None:
    if cfg.task != "classification":
        return
    try:
        import wandb
    except ImportError:
        return

    max_examples = int(cfg.evaluation.error_analysis.get("max_examples", 128))
    if max_examples <= 0:
        return

    split = str(cfg.evaluation.error_analysis.get("split", "val"))
    if split == "val":
        datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError("evaluation.error_analysis.split must be one of: val, test")

    model = lit_module.model
    device = _model_device(model, fallback=lit_module.device)
    model.eval()
    table = wandb.Table(
        columns=[
            "index",
            "image",
            "label",
            "prediction",
            "confidence",
            "label_probability",
            "margin",
            "top2_prediction",
            "top2_confidence",
        ]
    )

    logged = 0
    total_errors = 0
    seen = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    with torch.no_grad():
        for x, y in dataloader:
            x_device = x.to(device)
            logits = model(x_device)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(int(label) for label in y.tolist())
            all_predictions.extend(int(prediction) for prediction in preds.tolist())
            mistakes = preds != y
            if not bool(mistakes.any()):
                seen += len(y)
                continue

            images = _image_batch_for_wandb(cfg, x_device).cpu()
            top2_conf, top2_preds = probs.topk(k=2, dim=1)
            for batch_index in mistakes.nonzero(as_tuple=False).flatten().tolist():
                total_errors += 1
                if logged >= max_examples:
                    continue
                label = int(y[batch_index].item())
                prediction = int(preds[batch_index].item())
                table.add_data(
                    seen + batch_index,
                    wandb.Image(images[batch_index]),
                    label,
                    prediction,
                    float(top2_conf[batch_index, 0].item()),
                    float(probs[batch_index, label].item()),
                    float((top2_conf[batch_index, 0] - top2_conf[batch_index, 1]).item()),
                    int(top2_preds[batch_index, 1].item()),
                    float(top2_conf[batch_index, 1].item()),
                )
                logged += 1
            seen += len(y)

    wandb_run.summary[f"errors/{split}_misclassification_count_logged"] = logged
    wandb_run.summary[f"errors/{split}_misclassification_count_total"] = total_errors
    payload = {f"errors/{split}_misclassifications": table}
    if all_labels:
        class_names = _class_names(datamodule, cfg.dataset.name)
        payload[f"errors/{split}_confusion_matrix"] = wandb.plot.confusion_matrix(
            y_true=all_labels,
            preds=all_predictions,
            class_names=class_names,
        )
        payload[f"errors/{split}_confusion_counts"] = _confusion_count_table(
            wandb,
            all_labels,
            all_predictions,
            class_names,
        )
    wandb_run.log(payload)


def _model_device(model: torch.nn.Module, *, fallback: torch.device | str) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def _confusion_count_table(
    wandb: Any,
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> Any:
    table = wandb.Table(columns=["label", "prediction", "count"])
    counts: dict[tuple[int, int], int] = {}
    for label, prediction in zip(labels, predictions, strict=True):
        counts[(label, prediction)] = counts.get((label, prediction), 0) + 1
    for label, prediction in sorted(counts):
        table.add_data(
            _class_name(class_names, label),
            _class_name(class_names, prediction),
            counts[(label, prediction)],
        )
    return table


def _class_names(datamodule: Any, dataset_name: str) -> list[str]:
    for attr in ("val_data", "test_data", "train_data"):
        dataset = getattr(datamodule, attr, None)
        if dataset is None:
            continue
        classes = getattr(dataset, "classes", None)
        if classes is not None:
            return [str(name) for name in classes]
        nested_dataset = getattr(dataset, "dataset", None)
        classes = getattr(nested_dataset, "classes", None)
        if classes is not None:
            return [str(name) for name in classes]
    if dataset_name == "fashion_mnist":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    return [str(index) for index in range(10)]


def _class_name(class_names: list[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


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
        name=_wandb_artifact_name(f"{cfg.experiment_name}-run"),
        type="run-output",
        metadata=OmegaConf.to_container(cfg, resolve=True),
    )
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        artifact.add_file(str(config_path), name="config.yaml")
    metrics_matches = sorted(run_dir.glob("**/metrics.csv"))
    for metrics_path in metrics_matches:
        metrics_name = metrics_path.relative_to(run_dir).as_posix()
        artifact.add_file(str(metrics_path), name=f"metrics/{metrics_name}")
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        artifact.add_dir(str(checkpoint_dir), name="checkpoints")
    if report_path and Path(report_path).exists():
        artifact.add_file(report_path, name=Path(report_path).name)
    wandb_run.log_artifact(artifact)


def _wandb_artifact_name(name: str, max_length: int = 128) -> str:
    if len(name) <= max_length:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return f"{name[:prefix_length]}-{digest}".rstrip("-")


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
    from src.datasets.vision import normalization_stats

    mean_values, std_values = normalization_stats(dataset_name)
    mean = torch.tensor(mean_values, device=x.device).view(1, len(mean_values), 1, 1)
    std = torch.tensor(std_values, device=x.device).view(1, len(std_values), 1, 1)
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
