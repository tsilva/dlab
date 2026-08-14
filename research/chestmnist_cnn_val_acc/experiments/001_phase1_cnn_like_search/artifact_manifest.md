# Artifact Manifest

ChestMNIST validation-confirmation artifacts were produced on 2026-06-19.

Primary selected config artifacts:

- Seed 1337: [3ou9pzdw](https://wandb.ai/tsilva/dlab/runs/3ou9pzdw)
  - Selected checkpoint:
    `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-phase1-sweep_seed1337/checkpoints/009.ckpt`
- Seed 2024: [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f)
  - Selected checkpoint:
    `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-confirm-top-i000_seed2024/checkpoints/005.ckpt`
- Seed 9001: [dqlodc51](https://wandb.ai/tsilva/dlab/runs/dqlodc51)
  - Selected checkpoint:
    `/home/sky/sky_workdir/outputs/chestmnist-cnn_001-compact-cnn-family-search_adamw-lr0p001-bs512-wd3em05-cosine_do0-bn-res-cps2-ch64x128x256-aug_val-acc-confirm-top-i000_seed9001/checkpoints/001.ckpt`

Artifacts are logged as W&B run-output artifacts in `tsilva/dlab`, with R2
reference storage under the `chestmnist_cnn_val_acc` research track when
`CHECKPOINT_BUCKET_URI` is present.
