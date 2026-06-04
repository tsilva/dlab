# Checkpoints

W&B Artifacts are the durable checkpoint store. Local checkpoint files listed
below are cache/archive references only and are not committed to Git.

| Experiment | Role | Run Name | Local Checkpoint | W&B Artifact |
|---|---|---|---|---|
| 001 | best validation checkpoint | `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337` | `outputs/cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337/checkpoints/004.ckpt` | `cifar10-resnet18_gradclip-baseline_adamw-lr0p0003-bs128_seed1337-run` |
| 002 | best validation checkpoint | `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337` | `outputs/cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337/checkpoints/003.ckpt` | `cifar10-resnet18_cifar-stem_adamw-lr0p0003-bs128_seed1337-run` |
| 003 | selected validation checkpoint | `cifar10-wide-resnet_wrn28-10-cutmix-vs-erasing_adamw-lr0p0003-bs128-cosine_do0-d28-k10-ls0p05-mixup0p2-cutmix1_wrn28-10-cutmix-vs-erasing-seed-sweep_seed9001` | `outputs/artifacts/cifar10_wrn28_10_cutmix_best_run/checkpoints/063.ckpt` | `tsilva/dlab/cifar10-wide-resnet_wrn28-10-cutmix-vs-erasing_adamw-lr0p0003-bs128-cosine_do0-d28-k10-ls0p05-mixup0p2-cutmix1_wrn28-10-cutmix-vs-erasing-seed-sweep_seed9001-run` |

For future runs, the exact artifact reference should include the W&B version or
alias, for example `<entity>/<project>/<run_name>-run:v0` or
`<entity>/<project>/<run_name>-run:best-val`.
