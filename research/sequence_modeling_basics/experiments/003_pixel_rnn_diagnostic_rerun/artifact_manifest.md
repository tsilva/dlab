# Artifact Manifest

## Runs

- Config: `configs/experiment/mnist_rnn_pixel_diagnostics.yaml`
- Run name: `mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337`
- W&B run: `https://wandb.ai/tsilva/dlab/runs/7sz83m7v`
- Modal app run: `https://modal.com/apps/eng-tiago-silva/main/ap-6HGYvaLnnyJa7K2by1sdbm`
- Remote output: `/root/outputs/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337`
- Generated report: `reports/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337.md`
- Selected checkpoint: `/root/outputs/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337/checkpoints/002.ckpt`
- W&B artifact: `tsilva/dlab/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest`

This run was executed on Modal T4 with W&B enabled:

```bash
uv run python train.py experiment=mnist_rnn_pixel_diagnostics launcher=modal launcher.gpu=T4 launcher.timeout_seconds=3600 dataset.num_workers=2 wandb.enabled=true wandb.entity=tsilva wandb.project=dlab wandb.stable_id=false wandb.resume=never litlogger.enabled=false
```
