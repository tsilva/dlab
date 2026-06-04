from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import WeightAveraging


@dataclass(frozen=True)
class RunResult:
    run_dir: str
    metrics: dict[str, Any]
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"run_dir": self.run_dir, "metrics": self.metrics}
        if self.report_path is not None:
            payload["report_path"] = self.report_path
        return payload


def configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*LeafSpec.*deprecated.*TreeSpec.*is_leaf.*",
        module=r"pytorch_lightning\.utilities\._pytree",
    )


def run_experiment(cfg: DictConfig) -> RunResult:
    configure_runtime_warnings()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        EMAWeightAveraging,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    from src.datasets import build_datamodule
    from src.models import build_model
    from src.trainers import ResearchLitModule
    from src.utils.config import config_to_dict, save_resolved_config
    from src.utils.naming import resolve_run_identity
    from src.utils.reports import write_experiment_report
    from src.utils.run_target import collect_run_target, run_target_summary
    from src.utils.seed import seed_everything
    from src.utils.wandb import log_wandb_post_run, parameter_count, wandb_notes, wandb_tags

    configure_torch_runtime(cfg)
    seed_everything(int(cfg.seed), bool(cfg.trainer.get("deterministic", True)))

    run_identity = resolve_run_identity(cfg)
    cfg.experiment_name = run_identity.name
    cfg.run.group = run_identity.group
    run_target = collect_run_target(cfg)
    with open_dict(cfg):
        cfg.run_target = OmegaConf.create(run_target)

    datamodule = build_datamodule(cfg.dataset, seed=int(cfg.seed))
    model = build_model(cfg.model, datamodule.info)
    lit_module = ResearchLitModule(model, cfg)
    model_summary = parameter_count(model)

    run_dir = Path(cfg.paths.outputs_dir) / cfg.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_checkpoint_path = _materialize_evaluation_checkpoint(cfg, run_dir)
    save_resolved_config(cfg, run_dir)
    resume_ckpt_path = (
        None if _evaluation_only(cfg) else _resolve_resume_checkpoint(cfg, checkpoint_dir)
    )

    callbacks: list[pl.Callback] = [LearningRateMonitor(logging_interval="step")]
    if resume_ckpt_path is not None:
        callbacks.append(_ResumeRestoreEventCallback(resume_ckpt_path))
    weight_averaging = cfg.get("weight_averaging", {})
    weight_averaging_name = str(weight_averaging.get("name", "none"))
    if weight_averaging_name == "swa":
        callbacks.append(WeightAveraging(use_buffers=False))
    elif weight_averaging_name == "ema":
        update_starting_at_epoch = weight_averaging.get("update_starting_at_epoch")
        callbacks.append(
            EMAWeightAveraging(
                decay=float(weight_averaging.get("ema_decay", 0.999)),
                update_every_n_steps=int(weight_averaging.get("update_every_n_steps", 1)),
                update_starting_at_epoch=(
                    None if update_starting_at_epoch is None else int(update_starting_at_epoch)
                ),
            )
        )
    elif weight_averaging_name not in {"none", "null", ""}:
        raise KeyError(f"Unknown weight averaging '{weight_averaging_name}'")
    early_stopping_callback: EarlyStopping | None = None
    if cfg.get("early_stopping", {}).get("enabled", False):
        early_stopping_callback = EarlyStopping(
            monitor=cfg.early_stopping.get("monitor", cfg.checkpoint.monitor),
            mode=cfg.early_stopping.get("mode", cfg.checkpoint.mode),
            patience=int(cfg.early_stopping.patience),
            min_delta=float(cfg.early_stopping.get("min_delta", 0.0)),
            check_finite=bool(cfg.early_stopping.get("check_finite", True)),
            verbose=bool(cfg.early_stopping.get("verbose", False)),
        )
        callbacks.append(early_stopping_callback)
    if cfg.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch:03d}",
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                save_top_k=int(cfg.checkpoint.save_top_k),
                save_last=True,
                auto_insert_metric_name=False,
            )
        )
    if _modal_volume_commit_enabled(cfg):
        callbacks.append(_ModalVolumeCommitCallback(str(cfg.resume.modal_volume.name)))

    loggers: list[pl.loggers.Logger] = [
        CSVLogger(save_dir=str(run_dir), name="csv", version=""),
    ]
    if _wandb_enabled(cfg):
        loggers.append(
            WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.experiment_name,
                id=_wandb_run_id(cfg),
                resume=cfg.wandb.get("resume", "allow"),
                group=cfg.run.group,
                save_dir=str(run_dir),
                log_model=cfg.wandb.log_model,
                config=config_to_dict(cfg),
                mode=cfg.wandb.get("mode", "online"),
                job_type=cfg.wandb.get("job_type", "train"),
                tags=wandb_tags(cfg),
                notes=wandb_notes(cfg),
            )
        )
        if cfg.wandb.get("log_code", False):
            loggers[-1].experiment.log_code(".")
        if cfg.wandb.get("watch", {}).get("enabled", False):
            loggers[-1].watch(
                model,
                log=cfg.wandb.watch.get("log", "gradients"),
                log_freq=int(cfg.wandb.watch.get("log_freq", 100)),
            )
    if cfg.litlogger.enabled:
        from src.utils.loggers import build_litlogger

        loggers.append(
            build_litlogger(
                root_dir=cfg.litlogger.root_dir,
                name=cfg.experiment_name,
                teamspace=cfg.litlogger.get("teamspace"),
                metadata={"dataset": cfg.dataset.name, "model": cfg.model.name, "task": cfg.task},
                log_model=cfg.litlogger.log_model,
                save_logs=cfg.litlogger.save_logs,
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(cfg.trainer.max_epochs),
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        deterministic=bool(cfg.trainer.deterministic),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        enable_checkpointing=bool(cfg.trainer.enable_checkpointing),
        fast_dev_run=bool(cfg.trainer.get("fast_dev_run", False)),
    )
    started_at = time.perf_counter()
    if _evaluation_only(cfg):
        datamodule.prepare_data()
        datamodule.setup("fit")
    else:
        trainer.fit(
            lit_module,
            datamodule=datamodule,
            ckpt_path=resume_ckpt_path,
            weights_only=False if resume_ckpt_path is not None else None,
        )
    elapsed_seconds = time.perf_counter() - started_at
    early_stopping_metrics = _early_stopping_metrics(early_stopping_callback)
    selection_results = _run_selection_validation(cfg, trainer, lit_module, datamodule)
    tta_results = _run_tta_evaluation_if_enabled(cfg, trainer, lit_module, datamodule)
    test_results = _run_test_if_enabled(cfg, trainer, lit_module, datamodule)

    metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in trainer.callback_metrics.items()
    }
    metrics.update(selection_results)
    metrics.update(tta_results)
    if test_results:
        metrics.update(test_results[0])
    metrics.update(model_summary)
    metrics["runtime/seconds"] = elapsed_seconds
    metrics.update(run_target_summary(run_target))
    if evaluation_checkpoint_path is not None:
        metrics["evaluation/checkpoint_path"] = evaluation_checkpoint_path
    metrics.update(_resume_metrics(resume_ckpt_path))
    metrics.update(early_stopping_metrics)
    report_path = None
    if cfg.reports.enabled:
        report_path = str(write_experiment_report(cfg, metrics, run_dir, cfg.paths.reports_dir))
        print(f"Wrote report: {report_path}")

    log_wandb_post_run(
        cfg=cfg,
        trainer=trainer,
        lit_module=lit_module,
        datamodule=datamodule,
        run_dir=run_dir,
        report_path=report_path,
        elapsed_seconds=elapsed_seconds,
        extra_summary={
            **selection_results,
            **tta_results,
            **early_stopping_metrics,
            **_resume_metrics(resume_ckpt_path),
            **(
                {"evaluation/checkpoint_path": evaluation_checkpoint_path}
                if evaluation_checkpoint_path
                else {}
            ),
        },
    )

    result = RunResult(run_dir=str(run_dir), metrics=metrics, report_path=report_path)
    print(OmegaConf.to_yaml(result.to_dict(), resolve=True))
    return result


def _run_test_if_enabled(
    cfg: DictConfig,
    trainer: Any,
    lit_module: Any,
    datamodule: Any,
) -> list[dict[str, Any]]:
    if not cfg.get("evaluation", {}).get("test", {}).get("enabled", False):
        return []

    ckpt_path = cfg.evaluation.test.get("ckpt_path", "best")
    if ckpt_path == "best" and not bool(cfg.trainer.enable_checkpointing):
        ckpt_path = None
    return trainer.test(
        lit_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        weights_only=bool(cfg.evaluation.test.get("weights_only", False)),
    )


def _evaluation_only(cfg: DictConfig) -> bool:
    return bool(cfg.get("evaluation", {}).get("only", False))


def _materialize_evaluation_checkpoint(cfg: DictConfig, run_dir: Path) -> str | None:
    checkpoint_cfg = cfg.get("evaluation", {}).get("checkpoint", {})
    artifact_name = checkpoint_cfg.get("artifact")
    if artifact_name in {None, "", "none", "null"}:
        return None

    artifact_ref = _wandb_artifact_ref(cfg, str(artifact_name))
    artifact_file = checkpoint_cfg.get("file")
    checkpoint_path = _download_wandb_artifact_file(
        artifact_ref,
        None if artifact_file in {None, "", "none", "null"} else str(artifact_file),
        run_dir,
    )
    for section_name in ("selection", "tta", "test"):
        section = cfg.get("evaluation", {}).get(section_name)
        if section is not None and section.get("ckpt_path", "best") == "best":
            section.ckpt_path = checkpoint_path
    return checkpoint_path


def _wandb_artifact_ref(cfg: DictConfig, artifact_name: str) -> str:
    if artifact_name.count("/") >= 2:
        return artifact_name
    wandb_cfg = cfg.get("wandb", {})
    entity = wandb_cfg.get("entity")
    project = wandb_cfg.get("project")
    if entity in {None, ""} or project in {None, ""}:
        raise ValueError(
            "evaluation.checkpoint.artifact must be a full entity/project/artifact:version "
            "reference when wandb.entity or wandb.project is unset."
        )
    return f"{entity}/{project}/{artifact_name}"


def _download_wandb_artifact_file(
    artifact_ref: str,
    artifact_file: str | None,
    run_dir: Path,
) -> str:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    artifact_dir_name = "artifact-" + sha1(artifact_ref.encode("utf-8")).hexdigest()[:12]
    artifact_dir = Path(
        artifact.download(root=str(run_dir / "evaluation_checkpoints" / artifact_dir_name))
    )
    if artifact_file is None:
        matches = sorted(artifact_dir.glob("checkpoints/*.ckpt"))
        if len(matches) != 1:
            raise ValueError(
                "evaluation.checkpoint.file is required when the artifact does not contain "
                f"exactly one checkpoints/*.ckpt file. Found {len(matches)} in {artifact_ref}."
            )
        checkpoint_path = matches[0]
    else:
        checkpoint_path = artifact_dir / artifact_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"W&B artifact file does not exist after download: {checkpoint_path}"
        )
    return str(checkpoint_path)


def _run_tta_evaluation_if_enabled(
    cfg: DictConfig,
    trainer: Any,
    lit_module: Any,
    datamodule: Any,
) -> dict[str, Any]:
    tta_cfg = cfg.get("evaluation", {}).get("tta", {})
    if not bool(tta_cfg.get("enabled", False)):
        return {}
    if cfg.get("task", "classification") != "classification":
        return {}
    if not bool(cfg.trainer.get("enable_checkpointing", True)):
        return {}

    split = str(tta_cfg.get("split", "val"))
    ckpt_path = tta_cfg.get(
        "ckpt_path",
        cfg.get("evaluation", {}).get("selection", {}).get("ckpt_path", "best"),
    )
    selected_path = _checkpoint_path_for_eval(trainer, ckpt_path)
    if selected_path is None:
        return {}
    _load_lit_module_checkpoint(
        lit_module,
        selected_path,
        weights_only=bool(tta_cfg.get("weights_only", False)),
    )

    if split == "val":
        if hasattr(datamodule, "setup"):
            datamodule.setup("fit")
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup("test")
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError("evaluation.tta.split must be 'val' or 'test'.")

    import torch

    from src.datasets.vision import normalization_stats
    from src.evaluation import evaluate_classification_tta

    device = getattr(getattr(trainer, "strategy", None), "root_device", lit_module.device)
    transforms = list(tta_cfg.get("transforms", ["identity", "hflip"]))
    mean, std = (
        normalization_stats(str(cfg.dataset.name))
        if bool(cfg.dataset.get("normalize", True))
        else (None, None)
    )
    results = evaluate_classification_tta(
        model=lit_module.model,
        dataloader=dataloader,
        device=torch.device(device),
        transforms=transforms,
        label_smoothing=float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
        mean=mean,
        std=std,
    )
    metrics = {
        f"evaluation/tta/{split}/loss": results["loss"],
        f"evaluation/tta/{split}/acc": results["acc"],
        f"evaluation/tta/{split}/num_examples": results["num_examples"],
        f"evaluation/tta/{split}/num_views": results["num_views"],
        "evaluation/tta_checkpoint_source": str(ckpt_path),
        "evaluation/tta_checkpoint_path": selected_path,
    }
    numeric_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, int | float)
    }
    for logger in getattr(trainer, "loggers", []) or []:
        logger.log_metrics(numeric_metrics, step=int(getattr(trainer, "global_step", 0)))
    return metrics


def _checkpoint_path_for_eval(trainer: Any, ckpt_path: Any) -> str | None:
    if ckpt_path in {None, "", "none", "null"}:
        return None
    if ckpt_path == "best":
        if not _has_best_checkpoint(trainer):
            return None
        return _selected_checkpoint_path(trainer, ckpt_path)
    return str(ckpt_path)


def _load_lit_module_checkpoint(
    lit_module: Any,
    checkpoint_path: str,
    *,
    weights_only: bool,
) -> None:
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location=lit_module.device,
        weights_only=weights_only,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    lit_module.load_state_dict(state_dict)


def _early_stopping_metrics(callback: Any | None) -> dict[str, Any]:
    if callback is None:
        return {}
    stopped_epoch = int(getattr(callback, "stopped_epoch", 0))
    return {
        "early_stopping/stopped_epoch": stopped_epoch,
        "early_stopping/stopped": bool(stopped_epoch > 0),
        "early_stopping/wait_count": int(getattr(callback, "wait_count", 0)),
        "early_stopping/patience": int(getattr(callback, "patience", 0)),
    }


def _resolve_resume_checkpoint(cfg: DictConfig, checkpoint_dir: Path) -> str | None:
    resume_cfg = cfg.get("resume", {})
    if not bool(resume_cfg.get("enabled", False)):
        return None
    if not bool(cfg.trainer.get("enable_checkpointing", True)):
        return None

    ckpt_path = resume_cfg.get("ckpt_path", "last")
    if ckpt_path in {None, "", "none", "null"}:
        return None
    candidate = checkpoint_dir / "last.ckpt" if ckpt_path == "last" else Path(str(ckpt_path))
    if candidate.exists():
        print(f"Resuming training from checkpoint: {candidate}", flush=True)
        return str(candidate)
    if bool(resume_cfg.get("require_checkpoint", False)):
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
    return None


def _resume_metrics(ckpt_path: str | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {"resume/started_from_checkpoint": bool(ckpt_path)}
    if ckpt_path is not None:
        metrics["resume/checkpoint_path"] = ckpt_path
    return metrics


def _wandb_run_id(cfg: DictConfig) -> str | None:
    wandb_cfg = cfg.get("wandb", {})
    explicit_id = wandb_cfg.get("id")
    if explicit_id not in {None, "", "none", "null"}:
        return str(explicit_id)
    if not bool(wandb_cfg.get("stable_id", True)):
        return None

    entity = wandb_cfg.get("entity") or ""
    source = f"{entity}/{wandb_cfg.get('project')}/{cfg.experiment_name}"
    return "dlab-" + sha1(source.encode("utf-8")).hexdigest()[:24]


def _wandb_enabled(cfg: DictConfig) -> bool:
    return bool(cfg.get("wandb", {}).get("enabled", True))


def _modal_volume_commit_enabled(cfg: DictConfig) -> bool:
    resume_cfg = cfg.get("resume", {})
    volume_cfg = resume_cfg.get("modal_volume", {})
    return bool(volume_cfg.get("enabled", False) and volume_cfg.get("name"))


class _ModalVolumeCommitCallback(Callback):
    def __init__(self, volume_name: str) -> None:
        self.volume_name = volume_name
        self._volume = None

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        self._commit()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        self._commit()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._commit()

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        self._commit()

    def _commit(self) -> None:
        try:
            import modal
        except ImportError:
            return
        try:
            if self._volume is None:
                self._volume = modal.Volume.from_name(self.volume_name)
            self._volume.commit()
        except Exception as exc:
            print(
                f"Warning: failed to commit Modal volume '{self.volume_name}': {exc}",
                flush=True,
            )


class _ResumeRestoreEventCallback(Callback):
    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self._logged = False

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        if self._logged:
            return
        self._logged = True

        global_step = int(getattr(trainer, "global_step", 0))
        metrics = {
            "resume/event": 1.0,
            "resume/restored_from_checkpoint": 1.0,
            "resume/restored_epoch": int(getattr(trainer, "current_epoch", 0)),
            "resume/restored_global_step": global_step,
        }
        for logger in getattr(trainer, "loggers", []) or []:
            logger.log_metrics(metrics, step=global_step)


def _run_selection_validation(
    cfg: DictConfig,
    trainer: Any,
    lit_module: Any,
    datamodule: Any,
) -> dict[str, Any]:
    selection_cfg = cfg.get("evaluation", {}).get("selection", {})
    if not bool(selection_cfg.get("enabled", True)):
        return {}
    if not bool(cfg.trainer.get("enable_checkpointing", True)):
        return {}

    ckpt_path = selection_cfg.get("ckpt_path", "best")
    if ckpt_path in {None, "", "none", "null"}:
        return {}
    if ckpt_path == "best" and not _has_best_checkpoint(trainer):
        return {}

    results = trainer.validate(
        lit_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        weights_only=bool(selection_cfg.get("weights_only", False)),
        verbose=bool(selection_cfg.get("verbose", False)),
    )
    selected_metrics: dict[str, Any] = {
        "evaluation/selected_checkpoint_source": str(ckpt_path),
    }
    selected_path = _selected_checkpoint_path(trainer, ckpt_path)
    if selected_path:
        selected_metrics["evaluation/selected_checkpoint_path"] = selected_path

    if not results:
        return selected_metrics

    for key, value in results[0].items():
        metric = value.item() if hasattr(value, "item") else value
        selected_metrics[f"evaluation/selected/{key}"] = metric
        if str(key).startswith("val/"):
            selected_metrics[str(key)] = metric
    return selected_metrics


def _has_best_checkpoint(trainer: Any) -> bool:
    checkpoint = getattr(trainer, "checkpoint_callback", None)
    return bool(getattr(checkpoint, "best_model_path", ""))


def _selected_checkpoint_path(trainer: Any, ckpt_path: Any) -> str | None:
    if ckpt_path != "best":
        return None if ckpt_path is None else str(ckpt_path)
    checkpoint = getattr(trainer, "checkpoint_callback", None)
    best_path = getattr(checkpoint, "best_model_path", "")
    return str(best_path) if best_path else None


def configure_torch_runtime(cfg: DictConfig) -> None:
    precision = cfg.get("runtime", {}).get("float32_matmul_precision")
    if precision is None:
        return

    import torch

    torch.set_float32_matmul_precision(str(precision))
