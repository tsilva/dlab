# 003 WRN-28-10 CutMix vs Erasing

## Question

Does CutMix or light RandomErasing improve the current best WRN-28-10 CIFAR-10
recipe across seeds?

## Hypothesis

CutMix should help if patch-level label mixing improves spatial robustness;
RandomErasing should help if the model over-relies on small local cues.

## Setup

- Dataset: CIFAR-10 with augmentation, 10% validation split.
- Model: WideResNet-28-10 with CIFAR-style stem.
- Optimizer: AdamW, learning rate 0.0003, cosine schedule.
- Regularization: label smoothing 0.05, MixUp 0.2, CutMix 1.0 with probability 0.5.
- Budget: up to 75 epochs with early stopping.
- Selection metric: validation metrics; test metrics were not used.

## Results

| Run | Seed | Best Val Acc | Selected Val Acc | Selected Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---:|---|
| `cifar10-wide-resnet_wrn28-10-cutmix-vs-erasing_adamw-lr0p0003-bs128-cosine_do0-d28-k10-ls0p05-mixup0p2-cutmix1_wrn28-10-cutmix-vs-erasing-seed-sweep_seed9001` | 9001 | 0.9640 | 0.9624 | 0.4080 | - | `tsilva/dlab/<run_name>-run` |

## Observations

- The downloaded artifact includes `config.yaml`, `metrics/metrics.csv`,
  checkpoints, and the generated markdown report.
- The best validation accuracy observed in the metrics CSV is approximately
  0.9640.
- The generated report records the selected checkpoint validation accuracy as
  0.9624 and selected validation loss as 0.4080.

## Interpretation

This is the strongest migrated CIFAR-10 candidate and should anchor follow-up
confirmation work. Because only one downloaded run is represented here, the
experiment still needs the full seed-level comparison summarized before closing
the project milestone.

## Decision

Keep as the current strongest migrated candidate. Do not report final test
accuracy until the final candidate policy is satisfied.

## Next Experiment

Run or migrate the full seed comparison and then choose a final test-evaluation
candidate by validation metrics only.
