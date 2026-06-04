# 001 ResNet-18 Gradient-Clipping Baseline

## Question

How does a standard ResNet-18 baseline behave on CIFAR-10 when gradient clipping
and gradient-flow diagnostics are enabled?

## Hypothesis

The default ResNet-style model should train, but CIFAR-10 validation performance
may remain limited and gradients may show clipping pressure.

## Setup

- Dataset: CIFAR-10 with 10% validation split.
- Model: ResNet-18, default input stem.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.01.
- Budget: 20 epochs.
- Selection metric: validation accuracy/loss; test metrics were not used.

## Results

| Run | Seed | Best Val Acc | Best Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337` | 1337 | 0.6880 | 1.8596 | - | `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337-run` |

## Observations

- The migrated run reached validation accuracy around 0.688.
- The generated report records high clipping frequency and a large train/validation gap.

## Interpretation

This is useful as a diagnostic baseline, not as a competitive CIFAR-10 candidate.
The result supports moving to CIFAR-specific architecture/augmentation changes.

## Decision

Keep this experiment as the baseline reference.

## Next Experiment

Compare against a ResNet-18 with a CIFAR-sized input stem while holding the main
training setup fixed.
