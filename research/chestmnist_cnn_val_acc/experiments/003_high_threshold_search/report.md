# ChestMNIST Phase 3 High-Threshold Search

Status: complete as of 2026-06-19.

This phase tested whether making predictions more conservative could improve
raw validation accuracy. It ran two top ConvNet recipes at thresholds above
`0.5`:

- Dropout `0`, AdamW `weight_decay=0.00003`
- Dropout `0.1`, AdamW `weight_decay=0.0001`
- Shared settings: channels `[64, 128, 256]`, `convs_per_stage=2`, batch norm,
  residual blocks, AdamW `lr=0.001`, cosine schedule, batch size `512`, light
  affine augmentation, seed `1337`, validation only

Final W&B ranking after the sweep:

```text
ranked_runs=35
best single run: cfkcjq98, val_acc=0.949735, val_loss=0.158269, positive_rate=0.004081
best confirmed group: 3ou9pzdw agb59z8f dqlodc51, mean_val_acc=0.949607
best high-threshold run: 057rykaw, threshold=0.52, val_acc=0.949607, val_loss=0.157640, positive_rate=0.003152
```

Threshold results:

| Recipe | Threshold | Run | Val Acc | Val Loss | Positive Rate |
|---|---:|---|---:|---:|---:|
| dropout `0`, wd `3e-05` | 0.52 | [057rykaw](https://wandb.ai/tsilva/dlab/runs/057rykaw) | 0.949607 | 0.157640 | 0.003152 |
| dropout `0`, wd `3e-05` | 0.55 | [uagm87si](https://wandb.ai/tsilva/dlab/runs/uagm87si) | 0.949595 | 0.157640 | 0.002502 |
| dropout `0`, wd `3e-05` | 0.60 | [r1ppx5r1](https://wandb.ai/tsilva/dlab/runs/r1ppx5r1) | 0.949429 | 0.158451 | 0.001394 |
| dropout `0.1`, wd `0.0001` | 0.52 | [9j0akk7r](https://wandb.ai/tsilva/dlab/runs/9j0akk7r) | 0.949556 | 0.159671 | 0.006271 |
| dropout `0.1`, wd `0.0001` | 0.55 | [vbx0j637](https://wandb.ai/tsilva/dlab/runs/vbx0j637) | 0.949435 | 0.161131 | 0.001082 |
| dropout `0.1`, wd `0.0001` | 0.60 | [o30glvrm](https://wandb.ai/tsilva/dlab/runs/o30glvrm) | 0.949359 | 0.161131 | 0.000598 |

Conclusion: threshold `0.52` roughly matched the confirmed mean on the dropout
`0` recipe, but did not beat the best single run or improve the confirmed
three-seed result. Thresholds `0.55` and `0.60` became too conservative and
moved toward the all-negative baseline.

Operational note: the task ran on SkyPilot `ssh/beast2` / RTX 2060 as direct
validation-only W&B runs under `run.sweep_name=chestmnist_cnn_val_acc_high_threshold_phase3`.
Provider compute cost was `$0`; local electricity was not metered.
