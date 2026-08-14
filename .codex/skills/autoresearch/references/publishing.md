# Model Publishing Reference

Use this reference when an autoresearch task reaches artifact packaging or Hugging Face publication.

## ONNX-First Model Repo Standard

Package:

- `README.md`
- `.gitattributes`
- `assets/architecture.png`
- `config.yaml`
- `metadata.json`
- `metrics.csv`
- `model.ckpt`
- `model.onnx`

README frontmatter:

```yaml
---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
- image-classification
- <dataset-or-domain>
- <architecture-family>
- onnx
- onnxruntime
- pytorch
- dlab
datasets:
- <dataset>
metrics:
- accuracy
---
```

README body:

- title as the public model name, e.g. `MNIST GRU Classifier`;
- one sentence linking the model to `dlab`;
- architecture image near the top;
- aggregate validation and post-selection test audit metrics;
- representative checkpoint metrics;
- model details and source W&B run;
- input/output names, shapes, dtype, and preprocessing;
- ONNX Runtime usage snippet with `hf_hub_download`;
- labels;
- files;
- limitations.

## Publication Discipline

- Select checkpoints by validation metrics.
- Run test only after validation selection unless the user explicitly requests otherwise.
- Do not use test results to keep searching.
- Publish family/product names, not seed names.
- Keep seed and W&B run ids as provenance, not branding.
- Verify uploaded cards by downloading `README.md` from the Hub after upload.
