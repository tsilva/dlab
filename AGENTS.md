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

## Modal Experiment Execution

When running training experiments on Modal:

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
