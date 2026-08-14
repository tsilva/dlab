from __future__ import annotations

# ruff: noqa: E501
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import onnx
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PUBLISH_ROOT = ROOT / "outputs" / "hf_publish"
REPO_ROOT = PUBLISH_ROOT / "repos"


@dataclass(frozen=True)
class FamilyPackage:
    family: str
    repo_name: str
    artifact_dir: Path
    onnx_path: Path
    checkpoint_name: str
    selected_seed: int
    train_run_id: str
    train_run_name: str
    test_run_id: str
    val_accs: tuple[float, float, float]
    val_losses: tuple[float, float, float]
    test_accs: tuple[float, float, float]
    test_losses: tuple[float, float, float]
    selected_val_acc: float
    selected_val_loss: float
    selected_test_acc: float
    selected_test_loss: float


PACKAGES = [
    FamilyPackage(
        family="gru",
        repo_name="mnist-classifier-gru",
        artifact_dir=PUBLISH_ROOT / "mnist-rnn-gru-seed9001" / "artifact",
        onnx_path=PUBLISH_ROOT / "mnist-rnn-gru-seed9001" / "model.onnx",
        checkpoint_name="013.ckpt",
        selected_seed=9001,
        train_run_id="fs4fa4vl",
        train_run_name="mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs512-cosine_w384-d1-do0p2_val-acc-confirm-top-i000_seed9001",
        test_run_id="enmuxlt3",
        val_accs=(0.992667, 0.993667, 0.994667),
        val_losses=(0.027881, 0.022512, 0.018941),
        test_accs=(0.992900, 0.993500, 0.991700),
        test_losses=(0.022434, 0.022758, 0.027544),
        selected_val_acc=0.994667,
        selected_val_loss=0.018941,
        selected_test_acc=0.991700,
        selected_test_loss=0.027544,
    ),
    FamilyPackage(
        family="lstm",
        repo_name="mnist-classifier-lstm",
        artifact_dir=PUBLISH_ROOT / "mnist-lstm" / "artifact",
        onnx_path=PUBLISH_ROOT / "mnist-lstm" / "model.onnx",
        checkpoint_name="015.ckpt",
        selected_seed=9001,
        train_run_id="652i33os",
        train_run_name="mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs512-cosine_w384-d2-do0p1-ls0p01_val-acc-confirm-top-i001_seed9001",
        test_run_id="ntzpryf9",
        val_accs=(0.991000, 0.992667, 0.993833),
        val_losses=(0.099892, 0.094958, 0.091574),
        test_accs=(0.992500, 0.991700, 0.993100),
        test_losses=(0.097108, 0.096570, 0.092943),
        selected_val_acc=0.993833,
        selected_val_loss=0.091574,
        selected_test_acc=0.993100,
        selected_test_loss=0.092943,
    ),
    FamilyPackage(
        family="rnn",
        repo_name="mnist-classifier-rnn",
        artifact_dir=PUBLISH_ROOT / "mnist-rnn" / "artifact",
        onnx_path=PUBLISH_ROOT / "mnist-rnn" / "model.onnx",
        checkpoint_name="005.ckpt",
        selected_seed=9001,
        train_run_id="t8u8llqo",
        train_run_name="mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs256-constant_w512-d2-do0_val-acc-confirm-top-i005_seed9001",
        test_run_id="vdt5duxq",
        val_accs=(0.979333, 0.978500, 0.979667),
        val_losses=(0.073548, 0.075157, 0.067957),
        test_accs=(0.980800, 0.980600, 0.977800),
        test_losses=(0.064096, 0.065379, 0.076220),
        selected_val_acc=0.979667,
        selected_val_loss=0.067957,
        selected_test_acc=0.977800,
        selected_test_loss=0.076220,
    ),
]


def mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def pstdev(values: tuple[float, ...]) -> float:
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / len(values))


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def pp(value: float) -> str:
    return f"{value * 100:.4f} pp"


def family_label(package: FamilyPackage) -> str:
    return package.family.upper()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - width) / 2
    y = box[1] + (box[3] - box[1] - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=5, align="center")


def draw_architecture(path: Path, package: FamilyPackage, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = config["model"]["params"]
    family = family_label(package)
    sequence_axis = params["sequence_axis"]
    steps_label = "28 column vectors" if sequence_axis == "columns" else "28 row vectors"
    sequence_title = "Column sequence" if sequence_axis == "columns" else "Row sequence"
    hidden_dim = int(params["hidden_dim"])
    layers = int(params["num_layers"])
    classifier_in = hidden_dim * (2 if params["bidirectional"] else 1)
    dropout = params["dropout"]

    image = Image.new("RGB", (1600, 760), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = load_font(46, bold=True)
    label_font = load_font(27, bold=True)
    small_font = load_font(22)
    tiny_font = load_font(18)

    draw.text((72, 52), f"MNIST {family} Sequence Classifier", font=title_font, fill="#111827")
    draw.text(
        (74, 112),
        f"{steps_label} feed a {layers}-layer {family}; the pooled hidden state is classified into 10 digit logits.",
        font=small_font,
        fill="#475569",
    )

    boxes = [
        ((90, 240, 350, 420), "Input", "[batch, 1, 28, 28]", "#dbeafe", "#1d4ed8"),
        ((430, 240, 690, 420), sequence_title, "28 steps x 28 features", "#dcfce7", "#15803d"),
        (
            (770, 215, 1045, 445),
            family,
            f"hidden_dim={hidden_dim}\nlayers={layers}",
            "#fef3c7",
            "#b45309",
        ),
        ((1125, 240, 1325, 420), "Pooling", str(params["pooling"]), "#fae8ff", "#a21caf"),
        ((1400, 240, 1540, 420), "Linear", f"{classifier_in} -> 10", "#fee2e2", "#b91c1c"),
    ]

    for box, title, subtitle, bg, stroke in boxes:
        draw.rounded_rectangle(box, radius=24, fill=bg, outline=stroke, width=4)
        draw_centered(
            draw, (box[0], box[1] + 18, box[2], box[1] + 80), title, label_font, "#111827"
        )
        draw_centered(
            draw,
            (box[0] + 16, box[1] + 82, box[2] - 16, box[3] - 24),
            subtitle,
            small_font,
            "#334155",
        )

    arrow_y = 330
    for start, end in [(350, 430), (690, 770), (1045, 1125), (1325, 1400)]:
        draw.line((start + 18, arrow_y, end - 18, arrow_y), fill="#334155", width=5)
        draw.polygon(
            [(end - 18, arrow_y), (end - 38, arrow_y - 13), (end - 38, arrow_y + 13)],
            fill="#334155",
        )

    band = (90, 545, 1540, 655)
    draw.rounded_rectangle(band, radius=24, fill="#ffffff", outline="#cbd5e1", width=3)
    details = [
        "Optimizer: AdamW",
        f"LR: {config['optimizer']['lr']} + {config['optimizer']['scheduler']['name']}",
        f"Batch size: {config['dataset']['batch_size']}",
        "Selection: max validation accuracy",
        f"Dropout: {dropout}",
    ]
    x = 126
    for detail in details:
        draw.text((x, 590), detail, font=tiny_font, fill="#0f172a")
        x += draw.textlength(detail, font=tiny_font) + 48

    image.save(path)


def copy_publication_files(package: FamilyPackage) -> Path:
    out_dir = REPO_ROOT / package.repo_name
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    checkpoint_source = package.artifact_dir / "checkpoints" / package.checkpoint_name
    metrics_source = package.artifact_dir / "metrics" / "csv" / "metrics.csv"
    config_source = package.artifact_dir / "config.yaml"

    for source, target in [
        (package.onnx_path, out_dir / "model.onnx"),
        (checkpoint_source, out_dir / "model.ckpt"),
        (metrics_source, out_dir / "metrics.csv"),
        (config_source, out_dir / "config.yaml"),
    ]:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)

    with config_source.open() as f:
        config = yaml.safe_load(f)

    draw_architecture(assets_dir / "architecture.png", package, config)
    onnx.checker.check_model(str(out_dir / "model.onnx"))
    assert has_validation_row(metrics_source)

    (out_dir / "metadata.json").write_text(
        json.dumps(build_metadata(package, config), indent=2) + "\n"
    )
    (out_dir / "README.md").write_text(build_readme(package, config))
    write_gitattributes(out_dir / ".gitattributes")
    return out_dir


def has_validation_row(metrics_path: Path) -> bool:
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return any(row.get("val/acc") for row in reader)


def aggregate(package: FamilyPackage) -> dict[str, float]:
    return {
        "val_acc_mean": mean(package.val_accs),
        "val_acc_std": pstdev(package.val_accs),
        "val_loss_mean": mean(package.val_losses),
        "val_loss_std": pstdev(package.val_losses),
        "test_acc_mean": mean(package.test_accs),
        "test_acc_std": pstdev(package.test_accs),
        "test_loss_mean": mean(package.test_losses),
        "test_loss_std": pstdev(package.test_losses),
    }


def build_metadata(package: FamilyPackage, config: dict) -> dict:
    metrics = aggregate(package)
    return {
        "source_run": f"https://wandb.ai/tsilva/dlab/runs/{package.train_run_id}",
        "test_run": f"https://wandb.ai/tsilva/dlab/runs/{package.test_run_id}",
        "source_artifact": f"https://wandb.ai/tsilva/dlab/artifacts/run-output/{package.train_run_name}-run/latest",
        "checkpoint": "model.ckpt",
        "onnx": "model.onnx",
        "input_shape": [1, 28, 28],
        "input_name": "images",
        "output_name": "logits",
        "classes": [str(i) for i in range(10)],
        "preprocessing": {
            "to_tensor": True,
            "normalize_mean": [0.1307],
            "normalize_std": [0.3081],
        },
        "architecture": {
            "name": config["model"]["name"],
            **config["model"]["params"],
        },
        "metrics": {
            "selected_validation_accuracy": package.selected_val_acc,
            "selected_validation_loss": package.selected_val_loss,
            "test_accuracy": package.selected_test_acc,
            "test_loss": package.selected_test_loss,
            "three_seed_validation_accuracy_mean": metrics["val_acc_mean"],
            "three_seed_validation_accuracy_std": metrics["val_acc_std"],
            "three_seed_test_accuracy_mean": metrics["test_acc_mean"],
            "three_seed_test_accuracy_std": metrics["test_acc_std"],
        },
    }


def build_readme(package: FamilyPackage, config: dict) -> str:
    metrics = aggregate(package)
    params = config["model"]["params"]
    family = family_label(package)
    repo_id = f"tsilva/{package.repo_name}"
    source_run = f"https://wandb.ai/tsilva/dlab/runs/{package.train_run_id}"
    test_run = f"https://wandb.ai/tsilva/dlab/runs/{package.test_run_id}"
    scheduler = config["optimizer"]["scheduler"]["name"]
    augmentation = config["dataset"]["augmentation"]["enabled"]
    tag_family = package.family
    return f"""---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
- image-classification
- mnist
- {tag_family}
- onnx
- onnxruntime
- pytorch
- dlab
datasets:
- mnist
metrics:
- accuracy
---

# MNIST {family} Classifier

This repository contains a validation-selected MNIST {family} digit classifier trained with [dlab](https://github.com/tsilva/dlab).

## Architecture

![MNIST {family} architecture](assets/architecture.png)

## Results

3-seed confirmation and test audit for the selected {family} recipe:

| metric | value |
|---|---:|
| validation accuracy | {pct(metrics["val_acc_mean"])} ± {pp(metrics["val_acc_std"])} |
| validation loss | {metrics["val_loss_mean"]:.5f} ± {metrics["val_loss_std"]:.5f} |
| test accuracy | {pct(metrics["test_acc_mean"])} ± {pp(metrics["test_acc_std"])} |
| test loss | {metrics["test_loss_mean"]:.5f} ± {metrics["test_loss_std"]:.5f} |

Representative checkpoint selected by validation accuracy:

| metric | value |
|---|---:|
| seed | `{package.selected_seed}` |
| selected validation accuracy | {pct(package.selected_val_acc)} |
| selected validation loss | {package.selected_val_loss:.5f} |
| test accuracy | {pct(package.selected_test_acc)} |
| test loss | {package.selected_test_loss:.5f} |

The ONNX model was exported from the validation-selected checkpoint. Test metrics were produced after the recipe was selected and were logged in W&B test-audit run [`{package.test_run_id}`]({test_run}).

## Model Details

- Dataset: MNIST
- Architecture: {family} sequence classifier
- Sequence axis: `{params["sequence_axis"]}`
- Pooling: `{params["pooling"]}`
- Hidden width: `{params["hidden_dim"]}`
- Recurrent layers: `{params["num_layers"]}`
- Bidirectional: `{str(params["bidirectional"]).lower()}`
- Dropout: `{params["dropout"]}`
- Optimizer: AdamW
- Learning rate: `{config["optimizer"]["lr"]}`
- Weight decay: `{config["optimizer"]["weight_decay"]}`
- Scheduler: {scheduler}
- Label smoothing: `{config["loss"]["label_smoothing"]}`
- Batch size: `{config["dataset"]["batch_size"]}`
- Training augmentation: `{str(augmentation).lower()}`
- Checkpoint selection: max validation accuracy
- Source W&B run: [`{package.train_run_id}`]({source_run})

## Input / Output

Use `model.onnx` for code-independent inference.

- Input name: `images`
- Input shape: `[batch, 1, 28, 28]`
- Input dtype: `float32`
- Output name: `logits`
- Output shape: `[batch, 10]`

Preprocessing:

- Convert image to grayscale.
- Resize to `28 x 28`.
- Scale pixel values to `[0, 1]`.
- Normalize with mean `0.1307` and standard deviation `0.3081`.
- Arrange the tensor as channels-first `[batch, 1, 28, 28]`.

## Usage

Install the runtime dependencies:

```bash
pip install huggingface_hub onnxruntime pillow numpy
```

Run inference with the ONNX model:

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

LABELS = {{
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}}

model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="model.onnx",
)

image = Image.open("example.png").convert("L").resize((28, 28))
x = np.asarray(image, dtype=np.float32) / 255.0
x = (x - 0.1307) / 0.3081
x = x[None, None, :, :].astype(np.float32)

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
logits = session.run(["logits"], {{"images": x}})[0]
prediction = int(logits.argmax(axis=1)[0])

print(prediction, LABELS[prediction])
```

## Labels

MNIST labels:

| id | label |
|---:|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4 |
| 5 | 5 |
| 6 | 6 |
| 7 | 7 |
| 8 | 8 |
| 9 | 9 |

## Files

- `model.onnx`: ONNX export of the validation-selected checkpoint. Prefer this file for portable inference.
- `model.ckpt`: PyTorch Lightning checkpoint for the same model. This is code-dependent and mainly useful for PyTorch-based inspection or continued experimentation.
- `config.yaml`: resolved Hydra training config.
- `metrics.csv`: training metrics from the uploaded checkpoint run.
- `metadata.json`: compact metadata for inference and provenance.

## Limitations

This {family} model treats each MNIST image as a short sequence rather than using convolutional inductive bias. It is intended for normalized `28 x 28` grayscale MNIST-style images; remaining errors are expected to concentrate in ambiguous handwritten digits and distribution shifts outside that input format.
"""


def write_gitattributes(path: Path) -> None:
    path.write_text(
        "*.ckpt filter=lfs diff=lfs merge=lfs -text\n*.onnx filter=lfs diff=lfs merge=lfs -text\n"
    )


def main() -> None:
    outputs = [copy_publication_files(package) for package in PACKAGES]
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
