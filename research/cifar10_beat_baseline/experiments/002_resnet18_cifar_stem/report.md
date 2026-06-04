# 002 ResNet-18 CIFAR Stem

## Question

Does replacing the ImageNet-style ResNet-18 input stem with a CIFAR-sized stem
improve validation behavior on CIFAR-10?

## Hypothesis

A 3x3 stride-1 stem without max pooling should preserve spatial detail and
improve validation accuracy/loss relative to the default stem.

## Setup

- Dataset: CIFAR-10 with 10% validation split.
- Model: ResNet-18 with 3x3 stride-1 stem and no max pool.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.01.
- Budget: migrated run reached epoch 3 in the local metrics.
- Selection metric: validation accuracy/loss; test metrics were not used.

## Results

| Run | Seed | Best Val Acc | Best Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337` | 1337 | 0.6948 | 0.8702 | - | `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337-run` |

## Observations

- The CIFAR stem improved the migrated validation accuracy slightly over the
  default-stem gradient-clipping baseline.
- Validation loss improved materially in the migrated local metrics.
- No generated report was present in `reports/` for this run at migration time;
  the record is based on `outputs/<run_name>/config.yaml` and metrics CSV.

## Interpretation

The stem change is directionally useful and should remain part of later CIFAR-10
recipes, but the run is not enough by itself to reach the milestone.

## Decision

Keep the CIFAR stem as a baseline improvement and continue toward stronger
augmentation and higher-capacity candidates.

## Next Experiment

Evaluate stronger CIFAR-10 recipes such as WRN-28-10 with augmentation,
regularization, and seed confirmation.
