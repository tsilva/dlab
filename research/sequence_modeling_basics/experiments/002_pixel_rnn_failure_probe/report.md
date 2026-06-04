# 002 Pixel Rnn Failure Probe

## Question

Does a vanilla RNN fail or degrade when MNIST is treated as a 784-step pixel sequence instead of 28 row steps?

## Hypothesis

A vanilla RNN should struggle more on 784-step pixel sequences because long-range credit assignment and vanishing gradients make optimization harder than row-wise sequential MNIST.

## Setup

- Config: `configs/experiment/mnist_rnn_pixel_failure.yaml`
- Dataset: MNIST, interpreted as 784 pixel steps with 1 feature per step.
- Model: vanilla RNN, hidden size 128, one recurrent layer, final hidden-state classifier.
- Optimizer: Adam, learning rate 0.001.
- Budget: 5 epochs.
- Gradient clipping: 1.0.
- Baseline comparison: `001_rnn_baseline`, which used 28 row steps and reached `val/acc = 0.9507`.

## Failure Criteria

- Validation accuracy is materially lower than the row-wise RNN after the same epoch budget.
- Validation accuracy improves slowly or plateaus early.
- Validation loss remains high relative to the row-wise RNN.
- Gradient clipping or gradient-flow panels show optimization stress.

## Results

| Run | Seed | Val Acc | Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| `mnist-rnn_002-pixel-rnn-failure-probe_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | 1337 | 0.1090 | 2.3009 | - | `tsilva/dlab/mnist-rnn_002-pixel-rnn-failure-probe_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337-run:latest` |

## Observations

- Training completed normally on Modal T4 with W&B run `w30kx8ft`, but the model stayed at chance-level validation accuracy.
- The selected checkpoint was epoch `002.ckpt` with `evaluation/selected/val/acc = 0.1090` and `evaluation/selected/val/loss = 2.3009`.
- Final epoch training accuracy was also near chance (`train/acc_epoch = 0.112`), so this is not a validation-only generalization failure.
- Gradient clipping did not dominate the run; raw gradient norm decreased across epochs and clip coefficient stayed at 1.0.
- Runtime was `60.7931` seconds on Modal T4. Estimated GPU cost from Modal's T4 rate was about `$0.0100`, before small CPU/memory charges.

## Interpretation

- This is a useful failure state. The same vanilla RNN family reached `val/acc = 0.9507` on the 28-step row sequence, but collapsed to chance when the sequence length was increased to 784 one-pixel steps.
- The result points at model/representation characteristics rather than dataset difficulty: the row sequence preserves short local structure per step, while the pixel sequence asks the vanilla RNN to preserve useful evidence over hundreds of recurrent transitions.
- Because training accuracy also stayed near chance, the first diagnosis is optimization/credit assignment failure, not overfitting or poor validation split behavior.

## Decision

- Keep this as the recurrent failure baseline. Use it to evaluate whether gated recurrence fixes the long-sequence failure.

## Next Experiment

- Compare GRU and LSTM on the same 784-step pixel sequence with the same optimizer, batch size, hidden size, epoch budget, and W&B gradient diagnostics.
