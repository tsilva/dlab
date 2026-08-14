# ChestMNIST CNN Validation Accuracy

This project tracks CNN-like ChestMNIST classifiers selected by validation
accuracy. ChestMNIST is a MedMNIST multi-label binary-class task with 14 disease
labels, so training uses BCE-with-logits and validation accuracy is computed
from thresholded sigmoid probabilities.

Selection policy:

- Rank by `val/acc`.
- Tie-break with `val/loss` and `val/pred_positive_rate` diagnostics.
- Keep test evaluation disabled until one validation winner is selected and
  confirmed across seeds.

Current selected validation winner:

- Run: [ujafqz39](https://wandb.ai/tsilva/dlab/runs/ujafqz39)
- Config: compact residual ConvNet, channels `[64, 128, 256]`, dropout `0`,
  AdamW `lr=0.001`, `weight_decay=0.00003`, cosine schedule, batch size `512`,
  augmentation enabled
- Source checkpoint: seed `2024` confirmation run
  [agb59z8f](https://wandb.ai/tsilva/dlab/runs/agb59z8f)
- Decision threshold: `0.5358771681785583`
- Validation metrics: `val_acc=0.949894`, `val_loss=0.158365`,
  `val_pred_positive_rate=0.004380`

Prepared phase-1 search:

- `configs/sweep/chestmnist_cnn_val_acc_phase1.yaml`: compact residual ConvNet
  capacity, dropout, weight decay, and light affine augmentation.
- `configs/sweep/chestmnist_resnet18_val_acc_phase1.yaml`: 28px-compatible
  small-stem ResNet-18 optimizer and augmentation search.

Primary summarizer:

```bash
.venv/bin/python scripts/wandb_chestmnist_cnn_val_acc.py
```

After phase 1 finishes, generate validation-only confirmation commands for the
top unique config signatures:

```bash
.venv/bin/python scripts/wandb_chestmnist_cnn_val_acc.py \
  --confirm \
  --dry-run \
  --top-k 2 \
  --seeds 2024 9001
```

Remove `--dry-run` only after confirming the phase-1 W&B ranking. These
commands add extra seeds for the validation-selected configs and keep
`evaluation.test.enabled=false`.

Prepared SkyPilot command:

```bash
(
  set -a
  . ./.env
  set +a

  /Users/tsilva/repos/tsilva/sandbox-skypilot/.venv/bin/sky launch \
    -c dlab-chestmnist-cnn-val-acc-phase1-2060 \
    -y sky_chestmnist_cnn_val_acc_phase1_2060.yaml \
    --env AWS_REGION \
    --env AWS_S3_ENDPOINT_URL \
    --env CHECKPOINT_BUCKET_URI \
    --secret AWS_ACCESS_KEY_ID \
    --secret AWS_SECRET_ACCESS_KEY \
    --secret WANDB_API_KEY
)
```

Before launching from this machine, make sure the SkyPilot client is connected
to a live API server; the current client may need `sky api logout` and
`sky api start --host 127.0.0.1`.
