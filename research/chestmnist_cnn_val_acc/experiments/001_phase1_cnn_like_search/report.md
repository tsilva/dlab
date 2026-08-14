# ChestMNIST Phase 1 CNN-Like Search

Status: complete as of 2026-06-19.

This phase compares compact residual ConvNets and a 28px-compatible ResNet-18
under the same validation-first policy:

- Select by `val/acc`.
- Use `val/loss` as the main tie-breaker.
- Track `val/pred_positive_rate` to catch degenerate all-negative behavior.
- Keep test evaluation disabled.

Live W&B ranking after confirmation:

```text
ranked_runs=24
best single run: cfkcjq98, val_acc=0.949735, val_loss=0.158269, positive_rate=0.004081
best confirmed group: 3ou9pzdw agb59z8f dqlodc51, mean_val_acc=0.949607
```

Confirmed winner by three-seed mean validation accuracy:

- Compact residual ConvNet, channels `[64, 128, 256]`
- Dropout `0`
- AdamW `lr=0.001`, `weight_decay=0.00003`
- Cosine schedule
- Batch size `512`
- Light affine augmentation enabled
- Checkpoint monitor `val/acc`, mode `max`

| Config | Seeds | Mean Val Acc | Std | Mean Val Loss | Mean Positive Rate |
|---|---|---:|---:|---:|---:|
| dropout `0`, wd `3e-05` | 1337, 2024, 9001 | 0.949607 | 0.000086 | 0.160114 | 0.003941 |
| dropout `0.1`, wd `0.0001` | 1337, 2024, 9001 | 0.949510 | 0.000248 | 0.160038 | 0.002506 |

Key runs:

- [3ou9pzdw](https://wandb.ai/tsilva/dlab/runs/3ou9pzdw): seed 1337,
  `val_acc=0.949696`
- [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f): seed 2024,
  `val_acc=0.949601`
- [dqlodc51](https://wandb.ai/tsilva/dlab/runs/dqlodc51): seed 9001,
  `val_acc=0.949524`

Diagnostic note: this is the requested validation-accuracy winner, but
ChestMNIST is multi-label and the raw accuracy is dominated by negative labels.
The confirmed winner's mean positive prediction rate is only `0.003941`.

Operational note: the SkyPilot API server on beast-2 had to be restarted with
`KUBECONFIG=/home/tsilva/.kube/config` before `sky check ssh` enabled the
`ssh/beast2` target. The confirmation task ran on
`dlab-chestmnist-cnn-val-acc-confirm-2060` and succeeded with provider compute
cost `$0`.
