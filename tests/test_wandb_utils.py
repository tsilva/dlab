from __future__ import annotations

from types import SimpleNamespace

from src.utils.wandb import _class_names


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
