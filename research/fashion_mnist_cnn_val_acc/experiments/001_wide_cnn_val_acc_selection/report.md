# 001 Wide CNN Val-Acc Selection

## Question

Which of the two best wide Fashion-MNIST CNN regularization variants wins when
checkpointing and early stopping optimize validation accuracy directly?

## Setup

- Dataset: Fashion-MNIST with a 10% validation split.
- Model family: `cnn`.
- Architecture: `[64, 128, 256]` channels, two convolutions per stage, batch norm.
- Optimizer: Adam, learning rate `0.001`, cosine schedule.
- Regularization: label smoothing `0.02`, EMA, random affine augmentation.
- Compared variants: dropout `0.1` vs `0.2`, both with weight decay `0.00003`.
- Seeds: `1`, `2`, `3`.
- Selection metric: `val/acc`, mode `max`.
- Test policy: held-out test set is not used.

## Incumbent Bar

Existing W&B runs give this bar before the new selection-policy sweep:

| Variant | Mean Val Acc | Best Val Acc | Mean Val Loss |
|---|---:|---:|---:|
| dropout `0.1`, wd `0.00003` | 0.951500 | 0.953000 | 0.266265 |
| dropout `0.2`, wd `0.00003` | 0.951389 | 0.953500 | 0.265328 |

## Status

Planned. The sweep config has been generated and fast-dev smoke-tested, but the
SkyPilot launch requires explicit approval because it uses the local `.env` to
pass W&B and R2 credentials through SkyPilot.

## Decision Rule

Choose the config with the best mean validation accuracy across seeds. Use mean
validation loss as the first tie-breaker and best single-run validation accuracy
as secondary context only.
