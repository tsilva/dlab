# Artifact Manifest

## Run

- Run name: `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337`
- Local output: `outputs/cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337`
- Generated report: not present in `reports/` at migration time
- Local checkpoint: `outputs/cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337/checkpoints/003.ckpt`
- W&B artifact name: `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337-run`
- W&B artifact version: unknown in local migration

## Stored Remotely

Future reruns should preserve the same run-name key pattern and upload the
checkpoint, resolved config, metrics CSV, and generated report as the W&B
artifact `<run_name>-run`.
