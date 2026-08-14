# dlab Agent Notes

## W&B Dashboards

This repo uses curated W&B saved views for monitoring instead of relying on the
auto-generated workspace.

Default project:

- Entity: `tsilva`
- Project: `dlab`

Base saved views:

- Training monitor: https://wandb.ai/tsilva/dlab?nw=sv5fl2bc5oj
- Evaluation monitor: https://wandb.ai/tsilva/dlab?nw=90kzwksh85v
- Forensics: https://wandb.ai/tsilva/dlab?nw=k0m3cz11jsf
- Minimal gradient debug: https://wandb.ai/tsilva/dlab?nw=sjf87nvvkem
- Gradient diagnostics: https://wandb.ai/tsilva/dlab?nw=ugdkdmsk99g
- Sweep comparison: https://wandb.ai/tsilva/dlab?nw=uoso6yflo12

When running or helping with W&B-backed training:

- For an active or just-finished single run, point the user to the Training
  monitor first.
- If validation, test, reconstruction, or example-table outputs matter, also
  point to the Evaluation monitor.
- If debugging failures, misclassifications, early stopping, gradient flow, or
  suspicious results, point to the Forensics view.
- If debugging gradient clipping or gradient flow in a single run or small run
  comparison, point to Minimal gradient debug first; use Gradient diagnostics
  only when the minimal view is not detailed enough.

When running or helping with W&B sweeps:

- Point the user to the Sweep comparison view after creating or starting a
  sweep.
- If the sweep belongs to a roadmap stage or has a known `run.stage`, create
  stage-filtered saved views and return those links instead of only the base
  views:

```bash
uv run --with wandb-workspaces python scripts/setup_wandb_workspaces.py --entity tsilva --project dlab --stage <run.stage>
```

Use `--view training`, `--view evaluation`, `--view forensics`,
`--view gradient_debug`, `--view gradients`, or `--view sweeps` to create only
the relevant view. Use `--dry-run` before saving if the intended dashboard scope
is unclear.

After any W&B run or sweep, include the most relevant dashboard link(s) in the
handoff so the user can monitor the correct stage without searching through the
default W&B workspace.

## W&B Artifact Storage: Cloudflare R2

Use the same R2-backed W&B reference-artifact pattern as `../sandbox-sb3` for
large run artifacts. `CHECKPOINT_BUCKET_URI` should be a repo-local base URI,
normally `s3://wandb`. The code expands it with the research-track id from
`run.project`, so artifacts land under `s3://wandb/<research_track_id>/...`.
`wandb.artifact_storage_uri`, `WANDB_ARTIFACT_STORAGE_URI`, and
`CHECKPOINT_BUCKET_URI` also support `{research_track_id}` or
`<research_track_id>` placeholders.

Private local values are declared in `.keyenv.toml` and stored in macOS
Keychain:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
WANDB_API_KEY
```

Keep only non-secret settings such as `AWS_S3_ENDPOINT_URL`, `AWS_REGION`, and
`CHECKPOINT_BUCKET_URI=s3://wandb` in local `.env`. `WANDB_ARTIFACT_STORAGE_URI`
can override `CHECKPOINT_BUCKET_URI` when a run needs a different bucket/base.
The code loads only `WANDB_*`, `AWS_*`, and
`CHECKPOINT_BUCKET_URI` from `.env`, and exported shell variables take
precedence, so credentials injected by `keyenv run -- ...` remain authoritative.
Without either URI, W&B artifact behavior remains the normal direct upload path.

For SkyPilot launches, inject the Keychain credentials only into the local
`sky launch` process, then pass non-secret config with `--env` and secret values
with `--secret`. Do not mount `.env`, copy it into the workdir, source it inside
the task YAML, or bake it into an image/container.

Preferred SkyPilot launch pattern:

```bash
keyenv run -- sky launch -c <cluster-name> -y <task.yaml> \
  --env AWS_REGION \
  --env AWS_S3_ENDPOINT_URL \
  --env CHECKPOINT_BUCKET_URI \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  --secret WANDB_API_KEY
```

Smoke validation:

- 2026-06-18: `sky_dlab_r2_smoke_2060.yaml` succeeded on SkyPilot
  `ssh/beast2` / RTX 2060 with W&B run `afkr0egi`.
- W&B artifact:
  `https://wandb.ai/tsilva/dlab/artifacts/run-output/mnist-mlp_skypilot-r2-reference_adam-lr0p001-bs64-constant_w256-d2-do0p1_seed1337-run/latest`
- Confirmed manifest refs and R2 objects under
  `s3://wandb/skypilot_r2_smoke/mnist-mlp_skypilot-r2-reference_adam-lr0p001-bs64-constant_w256-d2-do0p1_seed1337-run/`.

## Modal Experiment Execution

When running training experiments on Modal:

- Before starting any run, sweep, or experiment, estimate the expected compute
  cost and ask the user for approval. Do not launch the run until the user
  approves the estimate.
- Separate the cost estimate into:
  - expected cost: the best forecast from measured prior epoch/runtime data,
    model size, likely early stopping, hardware rate, setup/download overhead,
    and artifact/evaluation overhead;
  - max approved budget: the conservative spend ceiling the user is approving;
  - assumptions: expected epochs or steps, hardware, provider rate, and whether
    CPU, memory, downloads, artifact upload, and retries/preemptions are included.
  Do not present a padded ceiling as the expected cost.
- At the end of any run, sweep, or experiment, report the actual measured cost
  alongside the result summary. Use provider billing data when available;
  otherwise compute an estimate from wall-clock/runtime, hardware, and the
  current provider price, and say that it is estimated.
- Use preemptible GPU instances by default. Keep `launcher.nonpreemptible:
  false` unless the user explicitly asks to pay the nonpreemptible premium for a
  specific run.
- Rely on durable checkpoint resume support for preemptions instead of switching
  to nonpreemptible execution.

## Current Best Experiments

Best CIFAR-10 validation setup so far, as of 2026-05-30:

- Model: Wide ResNet WRN-28-10.
- Recipe: AdamW, `lr=0.0003`, `batch_size=128`, cosine schedule, label
  smoothing `0.05`, mixup `0.2`, medium color jitter, RandAugment `N=1/M=7`,
  CutMix `alpha=1.0` on `p=0.5` of batches with MixUp fallback on the rest,
  early stopping/checkpoint selection by best validation loss.
- Config basis:
  `configs/experiment/local/cifar10_wrn28_10_aug_cosine75_ls005_jitter_medium_randaug_n1_m7_earlystop_mixup02_cutmix05_errors.yaml`
  with `configs/model/wide_resnet.yaml`.
- W&B sweep comparison: https://wandb.ai/tsilva/dlab/sweeps/oqqnf2am
- Confirmed seeds:
  - Seed `1337`, run `i74h6ia5`: selected val acc `0.9580`, selected val
    loss `0.4114`, validation errors `210`.
  - Seed `2024`, run `gbjs85ip`: selected val acc `0.9486`, selected val
    loss `0.4372`, validation errors `257`.
  - Seed `9001`, run `shhjijpo`: selected val acc `0.9624`, selected val
    loss `0.4080`, validation errors `188`.
- Three-seed mean selected validation accuracy is about `0.9563`, mean selected
  validation loss is about `0.4189`, and mean validation errors are about
  `218.3`.
- This beat the prior WRN-28-10 MixUp-only incumbent at about `0.9485` mean
  selected validation accuracy, `0.4283` mean selected validation loss, and
  `257.7` mean validation errors. The same sweep's light RandomErasing arm
  reached about `0.9513` mean selected validation accuracy, `0.4223` mean
  selected validation loss, and `243.0` mean validation errors, so CutMix is the
  stronger current recipe.

When resuming CIFAR-10 work, treat this WRN-28-10 recipe as the incumbent best
unless newer W&B runs explicitly beat it on selected validation loss/accuracy
across comparable seeds.

## Hugging Face Publishing

When the user asks to upload or publish a model to Hugging Face:

- Write or update the model card using an ONNX-first structure when an ONNX
  export is available or can reasonably be created:
  - Frontmatter: set `library_name: onnx`, `pipeline_tag` when applicable, and
    include relevant tags such as `onnx`, `onnxruntime`, dataset/task tags, and
    `dlab`.
  - Title and one concise description sentence linking `dlab` to
    `https://github.com/tsilva/dlab`.
  - `Architecture`: generated architecture image.
  - `Results`: validation mean ± std when available, final test metrics only
    when the user explicitly ran/approved test evaluation, and a note that test
    metrics were not used for checkpoint selection.
  - `Model Details`: dataset, architecture, key hyperparameters, source W&B run,
    source checkpoint, and final test run when applicable.
  - `Input / Output`: ONNX input/output names, shapes, dtype, and preprocessing.
  - `Usage`: install command and a runnable ONNX Runtime example that downloads
    `model.onnx` from Hugging Face with `hf_hub_download`; do not require
    PyTorch, torchvision, or project-local code for the primary inference path.
  - `Labels`: class id mapping when applicable.
  - `Files`: list `model.onnx` first; describe code-dependent checkpoints such
    as `.ckpt` as secondary artifacts.
  - `Limitations`: concise observed limitations or failure modes.
- Generate a clean model architecture image for the uploaded model.
- Verify the image matches the actual model config and implementation.
- Upload the image as a repo asset, for example `assets/architecture.png`.
- Integrate the image into the model card near the top, before the metrics.
- Prefer including a portable inference artifact such as ONNX when the uploaded
  checkpoint is code-dependent.
