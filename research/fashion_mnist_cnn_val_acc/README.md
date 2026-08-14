# Fashion-MNIST CNN Validation Accuracy

This project tracks CNN-like Fashion-MNIST classifiers selected by validation
accuracy. Generated run directories, reports, checkpoints, and W&B artifacts
remain outside this folder; this folder keeps the decision trail and exact
commands needed to reproduce the current ranking.

Current validation evidence points to the shallow/wide three-stage CNN family as
the best local neighborhood:

- channels `[64, 128, 256]`
- two convolutions per stage with batch norm
- cosine schedule, EMA, label smoothing `0.02`
- random affine augmentation
- dropout `0.1` or `0.2`
- weight decay `0.00003`

The next active experiment changes checkpoint and early-stopping selection from
validation loss to validation accuracy, because the objective is to maximize
`val/acc`.

Primary summarizer:

```bash
.venv/bin/python scripts/wandb_fashion_mnist_cnn_val_acc.py
```

After the validation winner is selected, prepare or run the held-out test audit
with:

```bash
.venv/bin/python scripts/wandb_fashion_mnist_cnn_test_eval.py --evaluate --dry-run
```

Remove `--dry-run` only after confirming the validation-selected candidate.

Prepared SkyPilot sweep:

```bash
(
  set -a
  . ./.env
  set +a

  sky launch -c dlab-fashion-mnist-cnn-val-acc-2060 \
    -y sky_fashion_mnist_cnn_val_acc_confirm_2060.yaml \
    --env AWS_REGION \
    --env AWS_S3_ENDPOINT_URL \
    --env CHECKPOINT_BUCKET_URI \
    --secret AWS_ACCESS_KEY_ID \
    --secret AWS_SECRET_ACCESS_KEY \
    --secret WANDB_API_KEY
)
```
