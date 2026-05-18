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
- Sweep comparison: https://wandb.ai/tsilva/dlab?nw=uoso6yflo12

When running or helping with W&B-backed training:

- For an active or just-finished single run, point the user to the Training
  monitor first.
- If validation, test, reconstruction, or example-table outputs matter, also
  point to the Evaluation monitor.
- If debugging failures, misclassifications, early stopping, gradient flow, or
  suspicious results, point to the Forensics view.

When running or helping with W&B sweeps:

- Point the user to the Sweep comparison view after creating or starting a
  sweep.
- If the sweep belongs to a roadmap stage or has a known `run.stage`, create
  stage-filtered saved views and return those links instead of only the base
  views:

```bash
uv run --with wandb-workspaces python scripts/setup_wandb_workspaces.py --entity tsilva --project dlab --stage <run.stage>
```

Use `--view training`, `--view evaluation`, `--view forensics`, or
`--view sweeps` to create only the relevant view. Use `--dry-run` before saving
if the intended dashboard scope is unclear.

After any W&B run or sweep, include the most relevant dashboard link(s) in the
handoff so the user can monitor the correct stage without searching through the
default W&B workspace.

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
