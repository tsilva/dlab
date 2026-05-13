from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


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
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    from src.datasets import build_datamodule
    from src.models import build_model
    from src.trainers import ResearchLitModule
    from src.utils.config import config_to_dict, save_resolved_config
    from src.utils.naming import resolve_run_identity
    from src.utils.reports import write_experiment_report
    from src.utils.seed import seed_everything

    configure_torch_runtime(cfg)
    seed_everything(int(cfg.seed), bool(cfg.trainer.get("deterministic", True)))


    run_identity = resolve_run_identity(cfg)
    cfg.experiment_name = run_identity.name
    cfg.run.group = run_identity.group

    datamodule = build_datamodule(cfg.dataset, seed=int(cfg.seed))
    model = build_model(cfg.model, datamodule.info)
    lit_module = ResearchLitModule(model, cfg)

    run_dir = Path(cfg.paths.outputs_dir) / cfg.experiment_name
    checkpoint_dir = run_dir / "checkpoints"
    save_resolved_config(cfg, run_dir)

    callbacks: list[pl.Callback] = [LearningRateMonitor(logging_interval="step")]
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
            )
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
    trainer.fit(lit_module, datamodule=datamodule)

    metrics = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in trainer.callback_metrics.items()
    }
    report_path = None
    if cfg.reports.enabled:
        report_path = str(write_experiment_report(cfg, metrics, run_dir, cfg.paths.reports_dir))
        print(f"Wrote report: {report_path}")

    result = RunResult(run_dir=str(run_dir), metrics=metrics, report_path=report_path)
    print(OmegaConf.to_yaml(result.to_dict(), resolve=True))
    return result


def configure_torch_runtime(cfg: DictConfig) -> None:
    precision = cfg.get("runtime", {}).get("float32_matmul_precision")
    if precision is None:
        return

    import torch

    torch.set_float32_matmul_precision(str(precision))
