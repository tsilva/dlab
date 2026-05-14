from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import onnx
import torch
from omegaconf import OmegaConf

from src.models import build_model


def main() -> None:
    args = parse_args()
    artifact_dir = download_artifact(args.artifact, args.download_dir)
    config_path = artifact_dir / args.config_name
    checkpoint_path = artifact_dir / args.checkpoint_name
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config in artifact: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint in artifact: {checkpoint_path}")

    cfg = OmegaConf.load(config_path)
    model = build_model(cfg.model, dataset_info_for(cfg))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state_dict(checkpoint))
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(args.batch_size, *input_shape_for(cfg), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if args.upload:
        upload_onnx_artifact(args, output_path, cfg, checkpoint_path)

    print(f"exported: {output_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"config: {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a W&B checkpoint artifact to ONNX.")
    parser.add_argument(
        "--artifact",
        default="tsilva/dlab/mnist-mlp-mlp-best-full-adam-lr0p001-bs512-w512-d1-do0p1-seed1-run:v0",
        help="W&B artifact path containing config.yaml and checkpoints.",
    )
    parser.add_argument("--checkpoint-name", default="checkpoints/011.ckpt")
    parser.add_argument("--config-name", default="config.yaml")
    parser.add_argument("--download-dir", default="outputs/artifacts/mnist_mlp_best_run")
    parser.add_argument(
        "--output",
        default="outputs/exported_models/mnist_mlp_best_seed1.onnx",
        help="ONNX output path.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--upload", action="store_true", help="Upload ONNX file as a W&B artifact.")
    parser.add_argument("--project", default="dlab")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--artifact-name", default="mnist-mlp-best-seed1-onnx")
    return parser.parse_args()


def download_artifact(artifact_name: str, download_dir: str) -> Path:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    return Path(artifact.download(root=download_dir))


def model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value
    return model_state or state_dict


def dataset_info_for(cfg: Any) -> dict[str, Any]:
    return {
        "input_shape": input_shape_for(cfg),
        "num_classes": int(cfg.dataset.get("num_classes", 10)),
    }


def input_shape_for(cfg: Any) -> tuple[int, int, int]:
    dataset_name = str(cfg.dataset.name)
    if dataset_name in {"mnist", "fashion_mnist"}:
        return (1, 28, 28)
    if dataset_name == "cifar10":
        return (3, 32, 32)
    return tuple(cfg.dataset.get("input_shape", (1, 28, 28)))


def upload_onnx_artifact(
    args: argparse.Namespace,
    output_path: Path,
    cfg: Any,
    checkpoint_path: Path,
) -> None:
    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="model-export",
        name=f"{args.artifact_name}-export",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type="model",
        metadata={
            "format": "onnx",
            "source_artifact": args.artifact,
            "source_checkpoint": str(checkpoint_path),
            "opset": args.opset,
            "input_name": "images",
            "output_name": "logits",
            "dynamic_batch": True,
        },
    )
    artifact.add_file(str(output_path), name=output_path.name)
    run.log_artifact(artifact, aliases=["best", "onnx"])
    run.finish()


if __name__ == "__main__":
    main()
