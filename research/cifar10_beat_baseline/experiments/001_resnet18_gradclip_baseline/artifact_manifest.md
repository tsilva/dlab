# Artifact Manifest

## Run

- Run name: `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337`
- Local output: `outputs/cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337`
- Generated report: `reports/cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337.md`
- Local checkpoint: `outputs/cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337/checkpoints/004.ckpt`
- W&B artifact name: `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337-run`
- W&B artifact version: unknown in local migration

## Stored Remotely

Future runs with `wandb.log_artifacts=true` should upload `config.yaml`,
`metrics/*.csv`, `checkpoints/*`, and the generated report into the W&B run
artifact named `<run_name>-run`.
