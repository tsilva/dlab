from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from src.utils.wandb import _class_names, wandb_tags


def test_class_names_use_underlying_subset_dataset_classes() -> None:
    datamodule = SimpleNamespace(
        val_data=SimpleNamespace(dataset=SimpleNamespace(classes=["cat", "dog"]))
    )

    assert _class_names(datamodule, "custom") == ["cat", "dog"]


def test_class_names_fallback_for_fashion_mnist() -> None:
    assert _class_names(SimpleNamespace(), "fashion_mnist") == [
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


def test_wandb_tags_are_shortened_to_wandb_limit() -> None:
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "dataset": {"name": "cifar10"},
            "model": {"name": "resnet18"},
            "optimizer": {"name": "adamw"},
            "run": {
                "tags": ["diagnostic"],
                "stage": "03_dataset_difficulty",
                "study": "cifar10_resnet18_cifar_stem_aug_cosine75_ls005_jitter_medium_earlystop",
                "sweep_name": None,
            },
        }
    )

    tags = wandb_tags(cfg)

    assert all(len(tag) <= 64 for tag in tags)
    assert any(tag.startswith("cifar10_resnet18_cifar_stem_aug_cosine75") for tag in tags)
