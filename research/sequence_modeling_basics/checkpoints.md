# Checkpoints

W&B Artifacts are the durable checkpoint store. Local checkpoint files are cache/archive references only and are not committed to Git.

| Experiment | Role | Run Name | Local Checkpoint | W&B Artifact |
|---|---|---|---|---|
| 001 | selected validation checkpoint | `mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | remote Modal output; use W&B artifact | `tsilva/dlab/mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest` |
| 002 | selected validation checkpoint | `mnist-rnn_002-pixel-rnn-failure-probe_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | remote Modal output; use W&B artifact | `tsilva/dlab/mnist-rnn_002-pixel-rnn-failure-probe_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest` |
| 003 | selected validation checkpoint | `mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | remote Modal output; use W&B artifact | `tsilva/dlab/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest` |
