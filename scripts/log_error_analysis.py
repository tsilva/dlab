from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import wandb
from omegaconf import OmegaConf

from src.datasets import build_datamodule
from src.models import build_model
from src.utils.wandb import _log_error_analysis_table


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.evaluation.error_analysis.enabled = True
    cfg.evaluation.error_analysis.split = args.split
    cfg.evaluation.error_analysis.max_examples = args.max_examples

    datamodule = build_datamodule(cfg.dataset, seed=int(cfg.seed))
    model = build_model(cfg.model, datamodule.info)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state_dict(checkpoint))
    model.eval()

    device = torch.device(args.device)
    model.to(device)
    lit_module = SimpleNamespace(model=model, device=device)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="error-analysis",
        name=args.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[
            "error-analysis",
            "validation-only" if args.split == "val" else "test-errors",
            str(cfg.dataset.name),
            str(cfg.model.name),
        ],
    )
    run.summary["source_checkpoint"] = str(Path(args.checkpoint))
    run.summary["error_analysis/split"] = args.split
    _log_error_analysis_table(cfg, run, lit_module, datamodule)
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log a W&B misclassification table for a checkpoint.")
    parser.add_argument("--config", required=True, help="Resolved training config.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint to evaluate.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default="error-analysis")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value
    return model_state or state_dict


if __name__ == "__main__":
    main()
