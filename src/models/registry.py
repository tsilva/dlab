from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.models.autoencoders import VAE, VQVAE, Autoencoder
from src.models.classifiers import MLP, ConvNet, ResNetClassifier, image_dim

MODEL_REGISTRY = {
    "mlp": MLP,
    "cnn": ConvNet,
    "resnet18": ResNetClassifier,
    "autoencoder": Autoencoder,
    "vae": VAE,
    "vqvae": VQVAE,
}


def _to_container(cfg: Mapping[str, Any] | DictConfig | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def build_model(model_cfg: DictConfig, dataset_info: Mapping[str, Any] | None = None):
    name = model_cfg.name
    params = _to_container(model_cfg.get("params", {}))
    dataset_info = dataset_info or {}

    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")

    input_shape = tuple(dataset_info.get("input_shape", (1, 28, 28)))
    if name == "mlp":
        params.setdefault("input_dim", image_dim(input_shape))
        params.setdefault("num_classes", dataset_info.get("num_classes", 10))
    elif name in {"cnn", "resnet18"}:
        params.setdefault("in_channels", input_shape[0])
        params.setdefault("num_classes", dataset_info.get("num_classes", 10))
    elif name in {"autoencoder", "vae", "vqvae"}:
        params.setdefault("input_shape", input_shape)

    return MODEL_REGISTRY[name](**params)

