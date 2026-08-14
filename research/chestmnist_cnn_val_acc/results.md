# Results

## Current Status

Phase 1, validation-only seed confirmation, and focused threshold follow-ups
are complete as of 2026-06-19. The selected validation-accuracy config is a
compact residual ConvNet with an exact scalar decision threshold tuned on the
validation split:

- Model: `cnn`
- Channels: `[64, 128, 256]`
- Convs per stage: `2`
- Batch norm: `true`
- Residual: `true`
- Dropout: `0`
- Optimizer: AdamW, `lr=0.001`, `weight_decay=0.00003`
- Scheduler: cosine
- Loss: BCE with logits, threshold `0.5358771681785583`
- Batch size: `512`
- Augmentation: enabled, random affine `degrees=5`, `translate=0.03`,
  `scale=[0.95, 1.05]`
- Checkpoint monitor: `val/acc`, mode `max`
- Source checkpoint: seed `2024` confirmation run
  [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f),
  `checkpoints/005.ckpt`

Selected validation-only result:

| Run | Val Acc | Val Loss | Positive Rate |
|---|---:|---:|---:|
| [ujafqz39](https://wandb.ai/tsilva/dlab/runs/ujafqz39) | 0.949894 | 0.158365 | 0.004380 |

This phase-4 result is the current best W&B-ranked validation-accuracy run
after refreshing `36` finished runs. It beats the previous best single run
([cfkcjq98](https://wandb.ai/tsilva/dlab/runs/cfkcjq98), `val_acc=0.949735`)
and the threshold-`0.5` three-seed confirmed mean (`0.949607`).

The underlying training recipe was first confirmed at threshold `0.5`:

Three-seed validation confirmation:

| Seed | Run | Val Acc | Val Loss | Positive Rate |
|---:|---|---:|---:|---:|
| 1337 | [3ou9pzdw](https://wandb.ai/tsilva/dlab/runs/3ou9pzdw) | 0.949696 | 0.157640 | 0.003788 |
| 2024 | [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f) | 0.949601 | 0.158365 | 0.006316 |
| 9001 | [dqlodc51](https://wandb.ai/tsilva/dlab/runs/dqlodc51) | 0.949524 | 0.164337 | 0.001719 |

Aggregate: mean validation accuracy `0.949607`, std `0.000086`, mean
validation loss `0.160114`, mean positive prediction rate `0.003941`.

The strongest single run after threshold-`0.5` confirmation was a different config
([cfkcjq98](https://wandb.ai/tsilva/dlab/runs/cfkcjq98), `val_acc=0.949735`),
but that config's three-seed mean was lower at `0.949510`.

Focused low-threshold search on the confirmed architecture did not improve the
validation-accuracy objective. The best low-threshold follow-up was threshold
`0.45` ([nx7t69cv](https://wandb.ai/tsilva/dlab/runs/nx7t69cv)) with
`val_acc=0.949524`, `val_loss=0.157640`, and positive prediction rate
`0.005997`. Lower thresholds increased the positive prediction rate but reduced
raw validation accuracy.

Focused high-threshold search across the two best ConvNet regularization
recipes also did not improve the objective. The best high-threshold follow-up
was threshold `0.52` on the confirmed dropout-`0`, weight-decay-`3e-05` recipe
([057rykaw](https://wandb.ai/tsilva/dlab/runs/057rykaw)) with
`val_acc=0.949607`, `val_loss=0.157640`, and positive prediction rate
`0.003152`. Higher thresholds reduced the positive rate toward the all-negative
regime and lowered raw validation accuracy.

Offline exact-threshold tuning over the top confirmed checkpoints then found
that the seed-`2024` dropout-`0`, weight-decay-`3e-05` checkpoint improved at
threshold `0.5358771681785583`. Re-running that setting through the normal
validation-only W&B evaluation path on SkyPilot `ssh/beast2` produced
[ujafqz39](https://wandb.ai/tsilva/dlab/runs/ujafqz39), the current best ranked
run at `val_acc=0.949894`.

Important caveat: ChestMNIST is multi-label and raw validation accuracy is
dominated by negative labels. The confirmed winner's positive prediction rate
is still very low, so this result maximizes the requested `val_acc` objective
but should not be treated as a clinically useful disease detector.

Completed local smoke:

- Config: `experiment=chestmnist_cnn_val_acc_search`
- Mode: `trainer.fast_dev_run=true`
- W&B: disabled
- Test evaluation: disabled
- Outcome: one train batch and one validation batch completed with
  BCE-with-logits, thresholded `val/acc`, and `val/pred_positive_rate` logged.

## Baseline Context

The MedMNIST public benchmark reports ChestMNIST ACC around `0.947` for
ResNet-18 at 28px resolution. This is a useful sanity target, not a selection
criterion; this project selects only from local validation evidence.

## Next Decision Gate

Selection by validation accuracy is complete. The next gate is a held-out test
audit for the selected config only, if explicitly approved.
