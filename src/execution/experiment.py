from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
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
        EMAWeightAveraging,
        EarlyStopping,
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
    from src.utils.seed import seed_everything
    from src.utils.wandb import log_wandb_post_run, parameter_count, wandb_notes, wandb_tags

    configure_torch_runtime(cfg)
    seed_everything(int(cfg.seed), bool(cfg.trainer.get("deterministic", True)))

    run_identity = resolve_run_identity(cfg)
    cfg.experiment_name = run_identity.name
    cfg.run.group = run_identity.group

    datamodule = build_datamodule(cfg.dataset, seed=int(cfg.seed))
    model = build_model(cfg.model, datamodule.info)
    lit_module = ResearchLitModule(model, cfg)
    model_summary = parameter_count(model)

    run_dir = Path(cfg.paths.outputs_dir) / cfg.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    save_resolved_config(cfg, run_dir)

    callbacks: list[pl.Callback] = [LearningRateMonitor(logging_interval="step")]
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

    loggers: list[pl.loggers.Logger] = [
        CSVLogger(save_dir=str(run_dir), name="csv", version=""),
    ]
    if cfg.wandb.enabled:
        loggers.append(
            WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=cfg.experiment_name,
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
    trainer.fit(lit_module, datamodule=datamodule)
    elapsed_seconds = time.perf_counter() - started_at
    test_results = _run_test_if_enabled(cfg, trainer, lit_module, datamodule)

    metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in trainer.callback_metrics.items()
    }
    if test_results:
        metrics.update(test_results[0])
    metrics.update(model_summary)
    metrics["runtime/seconds"] = elapsed_seconds
    if early_stopping_callback is not None:
        stopped_epoch = int(early_stopping_callback.stopped_epoch)
        metrics["early_stopping/stopped_epoch"] = stopped_epoch
        metrics["early_stopping/stopped"] = bool(stopped_epoch > 0)
        metrics["early_stopping/wait_count"] = int(early_stopping_callback.wait_count)
        metrics["early_stopping/patience"] = int(early_stopping_callback.patience)
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


def configure_torch_runtime(cfg: DictConfig) -> None:
    precision = cfg.get("runtime", {}).get("float32_matmul_precision")
    if precision is None:
        return

    import torch

    torch.set_float32_matmul_precision(str(precision))
