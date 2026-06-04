# Artifact Manifest

## Runs

- Config: `configs/experiment/mnist_rnn_sequence.yaml`
- Run name: `mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337`
- Modal app run: `https://modal.com/apps/eng-tiago-silva/main/ap-xITrwbon2S0eBhxADlBsdA`
- W&B run: `https://wandb.ai/tsilva/dlab/runs/cihe98pv`
- Remote output: `/root/outputs/mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337`
- Remote selected checkpoint: `/root/outputs/mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337/checkpoints/004.ckpt`
- W&B artifact: `tsilva/dlab/mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest`
- W&B artifact contents: resolved config, metrics CSV, checkpoints, and generated markdown report

## Durable Storage

This run was executed on Modal T4 with W&B enabled:

```bash
uv run python train.py experiment=mnist_rnn_sequence launcher=modal launcher.gpu=T4 launcher.timeout_seconds=3600 dataset.num_workers=2 wandb.enabled=true wandb.entity=tsilva wandb.project=dlab wandb.stable_id=false wandb.resume=never litlogger.enabled=false
```
