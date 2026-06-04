# 001 Rnn Baseline

## Question

Can a small vanilla RNN classify MNIST when each image is treated as a sequence of rows?

## Hypothesis

A small vanilla RNN should learn sequential MNIST quickly, but it should underperform CNN baselines because it lacks a strong spatial inductive bias.

## Setup

- Config: `configs/experiment/mnist_rnn_sequence.yaml`
- Dataset: MNIST, interpreted as 28 row steps with 28 pixel features per step.
- Model: vanilla RNN, hidden size 128, one recurrent layer, final hidden-state classifier.
- Optimizer: Adam, learning rate 0.001.
- Budget: 5 epochs.
- Gradient clipping: 1.0.
- Selection metric: validation accuracy. Test metrics are not used for selection.

## Results

| Run | Seed | Val Acc | Val Loss | Test Acc | Artifact |
|---|---:|---:|---:|---:|---|
| `mnist-rnn_001-rnn-baseline_adam-lr0p001-bs128-constant_w128-d1-do0_seed1337` | 1337 | 0.9507 | 0.1783 | - | `tsilva/dlab/<run_name>-run:latest` |

## Observations

- A 21.5k-parameter vanilla RNN reached `val/acc = 0.9507` after 5 epochs on Modal T4.
- The Modal run took about 42 seconds of remote runtime after startup/setup.
- The previous local MPS run reached `val/acc = 0.9473`; the Modal/W&B run is now the recorded durable result.
- Gradient clipping was enabled, but the selected summary does not indicate instability.
- Test metrics were not evaluated.

## Interpretation

- Sequential MNIST is learnable with a very small recurrent model.
- This is a strong enough baseline to make GRU/LSTM comparisons meaningful.
- The result should still underperform mature CNN baselines, which is useful for teaching inductive bias.

## Decision

- Keep as the first recurrent baseline.
- Use the W&B artifact as the durable checkpoint/report store.

## Next Experiment

- Compare GRU and LSTM variants under the same low-compute budget.
