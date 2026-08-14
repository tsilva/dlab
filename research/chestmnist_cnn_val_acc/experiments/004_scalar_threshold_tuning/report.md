# ChestMNIST Phase 4 Exact Scalar Threshold Tuning

Status: complete as of 2026-06-19.

This phase tuned only the scalar multi-label decision threshold for the strongest
confirmed ConvNet checkpoints. The training recipe stayed fixed:

- Model: compact residual ConvNet, channels `[64, 128, 256]`,
  `convs_per_stage=2`, batch norm, residual blocks, dropout `0`
- Optimizer: AdamW, `lr=0.001`, `weight_decay=0.00003`, cosine schedule
- Dataset: ChestMNIST validation split, batch size `512`, light affine
  augmentation used during training
- Checkpoint source: seed `2024` confirmation run
  [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f),
  `checkpoints/005.ckpt`
- Test evaluation: disabled

Offline exact-threshold scan selected scalar threshold
`0.5358771681785583` for the seed-`2024` checkpoint. Re-running that setting
through the normal validation-only W&B evaluation path produced:

| Run | Threshold | Val Acc | Val Loss | Positive Rate |
|---|---:|---:|---:|---:|
| [ujafqz39](https://wandb.ai/tsilva/dlab/runs/ujafqz39) | 0.5358771681785583 | 0.949894 | 0.158365 | 0.004380 |

Final W&B ranking after this phase:

```text
ranked_runs=36
best=0.949894 loss=0.158365 positive_rate=0.004380 run=ujafqz39
previous best single run=cfkcjq98 val_acc=0.949735
threshold-0.5 confirmed mean=0.949607
```

Conclusion: exact scalar threshold tuning is the best validation-accuracy
configuration found so far. It improves raw validation accuracy without changing
the trained weights. The improvement is still in the low-positive-rate regime,
which is expected for the requested raw multi-label accuracy objective.

Operational note: the W&B verification ran on SkyPilot `ssh/beast2` / RTX 2060
inside the existing `dlab-chestmnist-cnn-val-acc-confirm-2060` cluster as a
validation-only task under
`run.sweep_name=chestmnist_cnn_val_acc_scalar_threshold_phase4`. Provider compute
cost was `$0`; local electricity was not metered.
