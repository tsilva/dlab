---
name: autoresearch
description: End-to-end ML research execution workflow for validation-first experiment tracks, GPU sweeps, W&B/SkyPilot-backed training, candidate confirmation, test audit discipline, artifact packaging, and Hugging Face publication. Use when the user asks Codex to start or continue a research track, maximize a metric, run sweeps until a best config is found, evaluate best candidates, publish trained models, or turn experiment results into reusable model artifacts.
---

# Autoresearch

## Core Contract

Run research as a closed loop: define the question, search by validation metrics, confirm promising configs across seeds, run test evaluation only after selection, preserve provenance, and publish artifacts only after local and remote verification.

Default to acting end-to-end. Do not stop at a plan when the user asks to start, maximize, sweep, validate, publish, or push.

## Workflow

1. Establish the research track.
   - Read repo instructions, current research docs, configs, launch scripts, `.env` expectations, W&B conventions, and prior result summaries.
   - State the target metric, selection policy, hardware target, and test policy before launching work.
   - For mutable repo state, re-read local files and W&B outputs instead of relying on memory.

2. Design the sweep.
   - Choose a compact search space that fits the hardware. For small GPUs, prefer fewer high-signal trials over broad expensive grids.
   - Include controlled variables, changed variables, expected pattern, and failure criteria in configs or research docs.
   - Keep test data out of selection. Select by validation metrics and checkpoint monitor only.

3. Launch and monitor.
   - Use the repo's launcher boundary rather than embedding provider logic in training code.
   - For SkyPilot/R2/W&B flows, load repo-local `.env` only in the local shell that invokes `sky launch`; pass non-secrets with `--env` and secrets with `--secret`.
   - Before SkyPilot launches, verify the active API endpoint and enabled infra with `sky check ssh --verbose`; if the client has drifted to a stale or wrong API server, restart the intended remote API server, open a fresh localhost tunnel, log the local client into that tunnel, and re-check that the target node pool is selected.
   - If a SkyPilot launch cannot provision because an existing cluster already holds the GPU allocation, inspect current clusters/jobs before stopping anything; prefer `sky exec` into the existing correct cluster when it is idle and matches the requested hardware.
   - Keep Hydra CLI overrides structured and safely quoted. Avoid comma-containing free-text overrides in shell commands or YAML run blocks unless they are quoted with the repo's override formatter; put long narrative fields in config files or docs instead.
   - Monitor W&B runs for both outcome metrics and failure diagnostics such as OOM, throughput collapse, chance-level accuracy, prediction entropy, gradient clipping, and sequence-specific diagnostics.

4. Confirm candidates.
   - Group runs by full config signature, not by display name alone.
   - Confirm the top unique validation configs across additional seeds.
   - Rank by mean validation metric first, then tie-break with validation loss or best validation metric when justified.

5. Test audit.
   - Run test evaluation only for selected or explicitly approved candidates.
   - Treat test results as an audit, not a new hyperparameter search driver.
   - Report both per-seed checkpoint metrics and aggregate mean/std; say which checkpoint was chosen for export and why.

6. Package artifacts.
   - Prefer ONNX-first publication when feasible.
   - Package the same standard files across models: `README.md`, `.gitattributes`, `assets/architecture.png`, `config.yaml`, `metadata.json`, `metrics.csv`, `model.ckpt`, and `model.onnx`.
   - Name public model repos by architecture family or product identity, not seed ids or internal run slugs.
   - Keep seed/run details in provenance sections or metadata, not in the model name.

7. Verify before handoff.
   - Run structural checks such as `onnx.checker.check_model`.
   - Run runtime smoke tests when possible, e.g. ONNX Runtime with a zero tensor and expected logits shape.
   - Run repo formatting/lint checks for helper scripts.
   - After upload, download a small published file such as `README.md` from Hugging Face and verify the live card title, frontmatter, and metrics.

## Naming Rules

- Use user-facing names like `mnist-classifier-gru`, `mnist-classifier-lstm`, or `mnist-classifier-rnn`.
- Avoid names that combine research-track internals with architecture names, such as `rnn-gru`.
- Do not put seed ids in public model names.
- It is acceptable to mention representative checkpoint seed and W&B run ids inside provenance sections.

## Reporting Pattern

Lead with the result and the decision. Include enough detail for reproducibility:

- winning config and why it won;
- validation aggregate and test audit aggregate;
- representative checkpoint seed/run;
- artifact/repo URLs;
- exact validation commands that passed;
- any blocker, especially auth or provider failures.

For learning-oriented users, add the diagnostic insight: what failures taught, what metrics exposed the failure mode, and what the next search neighborhood should be.

## References

Read `references/publishing.md` when packaging or publishing model artifacts.
