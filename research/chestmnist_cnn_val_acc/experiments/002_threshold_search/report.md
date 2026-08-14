# ChestMNIST Phase 2 Threshold Search

Status: complete as of 2026-06-19.

This phase tuned only the multilabel decision threshold on the confirmed
compact residual ConvNet recipe:

- Channels `[64, 128, 256]`, `convs_per_stage=2`, batch norm, residual blocks
- Dropout `0`
- AdamW `lr=0.001`, `weight_decay=0.00003`
- Cosine schedule
- Batch size `512`
- Light affine augmentation enabled
- Seed `1337`
- Test evaluation disabled

The hypothesis was that the phase-1 incumbent predicted positives very rarely
at threshold `0.5`, so a modestly lower threshold might recover enough true
positives to improve raw validation accuracy.

Final W&B ranking after the sweep:

```text
ranked_runs=29
best single run: cfkcjq98, val_acc=0.949735, val_loss=0.158269, positive_rate=0.004081
best confirmed group: 3ou9pzdw agb59z8f dqlodc51, mean_val_acc=0.949607
best threshold run: nx7t69cv, threshold=0.45, val_acc=0.949524, val_loss=0.157640, positive_rate=0.005997
```

Threshold results:

| Threshold | Run | Val Acc | Val Loss | Positive Rate |
|---:|---|---:|---:|---:|
| 0.25 | [00ldoldq](https://wandb.ai/tsilva/dlab/runs/00ldoldq) | 0.942998 | 0.167527 | 0.020673 |
| 0.30 | [58efqy14](https://wandb.ai/tsilva/dlab/runs/58efqy14) | 0.946188 | 0.161634 | 0.015853 |
| 0.35 | [ns9es7qm](https://wandb.ai/tsilva/dlab/runs/ns9es7qm) | 0.947946 | 0.161634 | 0.010021 |
| 0.40 | [hvvuzili](https://wandb.ai/tsilva/dlab/runs/hvvuzili) | 0.948684 | 0.161634 | 0.006150 |
| 0.45 | [nx7t69cv](https://wandb.ai/tsilva/dlab/runs/nx7t69cv) | 0.949524 | 0.157640 | 0.005997 |

Conclusion: lowering the threshold increased the positive prediction rate, but
the extra false positives hurt the requested raw validation-accuracy objective.
The selected validation config remains the phase-1 confirmed threshold-`0.5`
ConvNet with mean `val_acc=0.949607`.

Operational note: the clean threshold sweep ran on SkyPilot
`ssh/beast2` / RTX 2060 as sweep
[`wew8i75n`](https://wandb.ai/tsilva/dlab/sweeps/wew8i75n). Provider compute
cost was `$0`; local electricity was not metered.
