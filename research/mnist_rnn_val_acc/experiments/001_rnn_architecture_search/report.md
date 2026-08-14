# 001 RNN Architecture Search

## Question

Which RNN-family MNIST sequence classifier maximizes selected validation accuracy on a dedicated RTX 2060?

## Hypothesis

GRU or LSTM row/column-sequence models with moderate width, bidirectionality, and light regularization should outperform the prior vanilla row-RNN baseline while avoiding the 784-step pixel-sequence failure mode.

## Setup

- Hardware: SkyPilot `ssh/beast2` target with one NVIDIA RTX 2060.
- Provider compute cost: `$0` on the local RTX 2060 target; electricity was not
  metered.
- Search phases:
  - Exploratory W&B sweeps, cancelling OOM/low-throughput settings.
  - Focused phase-1 sweep over GRU/LSTM widths, depth, learning rate, label
    smoothing, batch size, sequence axis, pooling, augmentation, and clipping.
  - Confirmation of the top seven unique phase-1 configs on seeds 2024 and
    9001, combined with the phase-1 seed 1337 result.
- Selection policy: validation accuracy only. Test metrics were not run.

SkyPilot job durations were approximately 1m53s, 12m25s, and 15m35s for
cancelled exploration attempts, plus 28m28s for the successful confirmation job.

## Results

| Run | Seed | Val Acc | Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| [z8wx1r2x](https://wandb.ai/tsilva/dlab/runs/z8wx1r2x) | 1337 | 0.992667 | 0.027881 | n/a | [run output](https://wandb.ai/tsilva/dlab/artifacts/run-output/mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs512-cosine_w384-d1-do0p2_val-acc-phase1-sweep_seed1337-run/latest) |
| [slgba1s8](https://wandb.ai/tsilva/dlab/runs/slgba1s8) | 2024 | 0.993667 | 0.022512 | n/a | [run output](https://wandb.ai/tsilva/dlab/artifacts/run-output/mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs512-cosine_w384-d1-do0p2_val-acc-confirm-top-i000_seed2024-run/latest) |
| [fs4fa4vl](https://wandb.ai/tsilva/dlab/runs/fs4fa4vl) | 9001 | 0.994667 | 0.018941 | n/a | [run output](https://wandb.ai/tsilva/dlab/artifacts/run-output/mnist-rnn_001-rnn-architecture-search_adamw-lr0p003-bs512-cosine_w384-d1-do0p2_val-acc-confirm-top-i000_seed9001-run/latest) |

Winner summary:

- Mean selected validation accuracy: `0.993667`.
- Population standard deviation across seeds: `0.000816`.
- Best single selected validation accuracy: `0.994667`.
- Mean selected validation loss: `0.023111`.
- Mean test accuracy after explicit test audit: `0.992700`.
- Population standard deviation for test accuracy: `0.000748`.
- Best single test accuracy: `0.993500`.
- Mean test loss: `0.024245`.
- Config: `gru`, hidden size `384`, `1` layer, unidirectional, dropout `0.2`,
  `last` pooling, `columns` sequence axis, AdamW `lr=0.003`,
  `weight_decay=0.001`, cosine scheduler, label smoothing `0`, batch size `512`,
  augmentation enabled, gradient clip value `2`.

Confirmed config ranking:

| Rank | Mean Val Acc | Std | Best | Mean Val Loss | N | Config |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.993667 | 0.000816 | 0.994667 | 0.023111 | 3 | `gru w384 d1 bidir=False pool=last axis=columns lr=0.003 wd=0.001 sched=cosine do=0.2 ls=0 bs=512 aug=True clip=2` |
| 2 | 0.992500 | 0.001163 | 0.993833 | 0.095475 | 3 | `lstm w384 d2 bidir=False pool=last axis=columns lr=0.003 wd=0.001 sched=cosine do=0.1 ls=0.01 bs=512 aug=True clip=1` |
| 3 | 0.991833 | 0.001780 | 0.993333 | 0.037100 | 3 | `lstm w256 d3 bidir=True pool=mean axis=rows lr=0.001 wd=0.001 sched=cosine do=0.2 ls=0 bs=256 aug=False clip=2` |
| 4 | 0.989278 | 0.001220 | 0.991000 | 0.110316 | 3 | `lstm w256 d1 bidir=False pool=last axis=rows lr=0.003 wd=0.001 sched=constant do=0.2 ls=0.01 bs=512 aug=False clip=0.5` |
| 5 | 0.987056 | 0.001624 | 0.988667 | 0.324981 | 3 | `lstm w384 d1 bidir=False pool=mean axis=columns lr=0.0003 wd=0.0001 sched=cosine do=0.2 ls=0.05 bs=256 aug=False clip=0.5` |
| 6 | 0.979167 | 0.000491 | 0.979667 | 0.072221 | 3 | `rnn w512 d2 bidir=False pool=mean axis=rows lr=0.003 wd=0.0001 sched=constant do=0 ls=0 bs=256 aug=False clip=1` |
| 7 | 0.967667 | 0.001472 | 0.969167 | 0.387803 | 3 | `rnn w384 d3 bidir=False pool=last axis=columns lr=0.003 wd=0.0001 sched=constant do=0 ls=0.05 bs=512 aug=True clip=2` |

Test audit ranking, run after the validation-confirmed candidates were already
selected:

| Rank | Mean Test Acc | Std | Best Test Acc | Mean Test Loss | N | Config |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.992700 | 0.000748 | 0.993500 | 0.024245 | 3 | `gru w384 d1 bidir=False pool=last axis=columns lr=0.003 wd=0.001 sched=cosine do=0.2 ls=0 bs=512 aug=True clip=2` |
| 2 | 0.992433 | 0.000573 | 0.993100 | 0.095540 | 3 | `lstm w384 d2 bidir=False pool=last axis=columns lr=0.003 wd=0.001 sched=cosine do=0.1 ls=0.01 bs=512 aug=True clip=1` |
| 3 | 0.990867 | 0.001223 | 0.992100 | 0.039376 | 3 | `lstm w256 d3 bidir=True pool=mean axis=rows lr=0.001 wd=0.001 sched=cosine do=0.2 ls=0 bs=256 aug=False clip=2` |
| 4 | 0.990067 | 0.001389 | 0.992000 | 0.106015 | 3 | `lstm w256 d1 bidir=False pool=last axis=rows lr=0.003 wd=0.001 sched=constant do=0.2 ls=0.01 bs=512 aug=False clip=0.5` |
| 5 | 0.986900 | 0.000356 | 0.987400 | 0.323762 | 3 | `lstm w384 d1 bidir=False pool=mean axis=columns lr=0.0003 wd=0.0001 sched=cosine do=0.2 ls=0.05 bs=256 aug=False clip=0.5` |
| 6 | 0.979733 | 0.001370 | 0.980800 | 0.068565 | 3 | `rnn w512 d2 bidir=False pool=mean axis=rows lr=0.003 wd=0.0001 sched=constant do=0 ls=0 bs=256 aug=False clip=1` |
| 7 | 0.971000 | 0.003590 | 0.975900 | 0.374466 | 3 | `rnn w384 d3 bidir=False pool=last axis=columns lr=0.003 wd=0.0001 sched=constant do=0 ls=0.05 bs=512 aug=True clip=2` |

## Observations

- The winning GRU was small enough to train quickly and consistently. It beat
  the deeper LSTM runner-up by about `0.001167` mean selected validation
  accuracy while using fewer trainable parameters.
- Column-wise sequences outperformed the searched vanilla RNN row/pixel-style
  candidates here. The plain RNN configs were materially worse even when wide.
- Label smoothing helped some LSTM configurations remain stable, but the best
  GRU used no label smoothing and achieved the lowest mean selected validation
  loss among the confirmed configs.
- Large early broad-sweep settings caused OOM or crashed W&B runs on the 6 GB
  RTX 2060. The focused sweep avoided batch size 1024 and kept useful GPU
  utilization without making memory the bottleneck.
- The held-out test audit did not overturn the validation decision. The top GRU
  beat the LSTM runner-up by only `0.000267` mean test accuracy, but it also had
  much lower mean test loss (`0.024245` vs `0.095540`).

## Interpretation

The best current RNN-family MNIST classifier in this track is a moderately wide,
single-layer GRU that treats each image as a sequence of columns. The result is
not a lucky seed: it held the top mean across the three confirmed seeds and the
best single seed. The larger LSTMs can approach it, but their added capacity did
not buy a better validation result in this search region.

## Decision

Select the GRU `w384 d1 columns last` config as the current best. Do not run
additional test-driven hyperparameter search from this test result. The user
explicitly requested this test audit; use it to report the final candidate, not
to keep iterating on the test set.

## Next Experiment

If this track continues, search locally around the GRU winner rather than
broadening immediately: `hidden_dim` 320/384/448, dropout 0.1/0.2/0.3,
`lr` 0.002/0.003/0.004, optional recurrent dropout if supported, and
augmentation strength. Keep batch size 512 unless memory pressure changes.
