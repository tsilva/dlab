# Checkpoints

Validation-selected checkpoint family chosen on 2026-06-19:

| Seed | Run | Selected checkpoint | Val Acc | Val Loss | Positive Rate |
|---:|---|---|---:|---:|---:|
| 1337 | [3ou9pzdw](https://wandb.ai/tsilva/dlab/runs/3ou9pzdw) | `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-phase1-sweep_seed1337/checkpoints/009.ckpt` | 0.949696 | 0.157640 | 0.003788 |
| 2024 | [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f) | `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-confirm-top-i000_seed2024/checkpoints/005.ckpt` | 0.949601 | 0.158365 | 0.006316 |
| 9001 | [dqlodc51](https://wandb.ai/tsilva/dlab/runs/dqlodc51) | `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-confirm-top-i000_seed9001/checkpoints/001.ckpt` | 0.949524 | 0.164337 | 0.001719 |

Selected config:

- `model.name=cnn`
- `model.params.channels=[64,128,256]`
- `model.params.convs_per_stage=2`
- `model.params.batch_norm=true`
- `model.params.residual=true`
- `model.params.dropout=0`
- `optimizer.name=adamw`
- `optimizer.lr=0.001`
- `optimizer.weight_decay=3e-05`
- `optimizer.scheduler.name=cosine`
- `dataset.batch_size=512`
- `dataset.augmentation.enabled=true`
- `checkpoint.monitor=val/acc`
- `checkpoint.mode=max`

Policy used:

- Selection metric: validation accuracy.
- Tie-breaker: validation loss, then non-degenerate positive prediction rate.
- Test data: not used for selection.
- Durable storage: W&B run-output artifacts, using R2 reference storage when
  `CHECKPOINT_BUCKET_URI` is present.
