# Checkpoints

No final Fashion-MNIST val_acc-selected checkpoint has been chosen yet.

Current policy:

- Selection metric: validation accuracy.
- Tie-breaker: validation loss.
- Test data: not used for selection.
- Durable storage: W&B run-output artifacts, using R2 reference storage when
  `CHECKPOINT_BUCKET_URI` is present.

After the confirmation sweep finishes, record the selected run artifact and the
representative checkpoint here.

Use `scripts/wandb_fashion_mnist_cnn_test_eval.py` to materialize evaluation
commands from W&B run-output artifacts. The script requires three finished
confirmation seeds for a config before it will select that config for test
audit.
