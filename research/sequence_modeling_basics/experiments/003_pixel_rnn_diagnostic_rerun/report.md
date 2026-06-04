# 003 Pixel Rnn Diagnostic Rerun

## Question

Can timestep hidden-state, entropy, and recurrent-gradient diagnostics identify why the 784-step vanilla RNN fails?

## Hypothesis

The pixel-sequence RNN should remain at chance while W&B diagnostics show high prediction entropy and weak recurrent signal flow compared with the classifier head or input kernel.

## Setup

- Config: `configs/experiment/mnist_rnn_pixel_diagnostics.yaml`
- Dataset: MNIST, interpreted as 784 pixel steps with 1 feature per step.
- Model: vanilla RNN, hidden size 128, one recurrent layer, final hidden-state classifier.
- Optimizer: Adam, learning rate 0.001.
- Budget: 5 epochs.
- Added diagnostics: prediction entropy, hidden-state norms by timestep, recurrent/input/classifier gradient splits.
- Baseline comparison: `002_pixel_rnn_failure_probe`, which reached `val/acc = 0.1090`.

## Failure Criteria

- Validation accuracy remains near chance under the same budget.
- Prediction entropy remains close to `log(10)`.
- Recurrent-gradient split shows weak recurrent signal relative to classifier or input gradients.
- Hidden-state timestep panels show little useful separation across the long sequence.

## Results

| Run | Seed | Val Acc | Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| `mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | 1337 | 0.1090 | 2.3009 | - | `tsilva/dlab/mnist-rnn_003-pixel-rnn-diagnostic-rerun_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest` |

## Observations

- Training reproduced the failure state: selected `val/acc = 0.1090`, selected `val/loss = 2.3009`.
- Prediction entropy stayed essentially maximal: selected `val/pred_entropy = 2.3007`, normalized entropy `0.9992`, and selected `val/pred_max_prob = 0.1121`.
- Hidden-state magnitude decayed across the long sequence: selected validation `hidden_norm_t000 = 0.5628`, `hidden_norm_tlast = 0.3566`, `last_to_first_ratio = 0.6336`.
- The train recurrent gradient was weaker than the input-kernel gradient (`recurrent_to_input_ratio = 0.6861`) while classifier gradients dominated recurrent gradients (`classifier_to_recurrent_ratio = 14.3552`).
- Gradient clipping did not explain the failure: epoch clip coefficient stayed `1.0`, clipped-step fraction stayed `0.0`.
- Runtime was `64.7171` seconds on Modal T4. Estimated GPU cost was `$0.0106`, before small CPU/memory charges.

## Interpretation

- This is a stronger diagnosis than experiment `002`: the model is not merely inaccurate, it remains nearly maximally uncertain.
- The hidden-state timestep probes show late-sequence state magnitude below early-sequence state magnitude, consistent with weak long-range signal persistence across 784 recurrent transitions.
- The gradient split points to the classifier head receiving much larger updates than the recurrent transition. That is consistent with a model that can adjust the final readout but is not learning useful temporal dynamics through the long pixel stream.
- Because clipping is inactive, the immediate failure signature is not exploding gradients. The leading diagnosis is long-sequence credit assignment / temporal representation failure in a vanilla RNN.

## Decision

- Keep this as the diagnostic failure baseline. The next useful experiment is a matched GRU or LSTM on the same 784-step sequence to test whether gating improves entropy, hidden-state persistence, recurrent gradient balance, and validation accuracy.

## Next Experiment

- Run a GRU with the same hidden size, optimizer, batch size, seed, epoch budget, and sequence diagnostics.
