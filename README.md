<div align="center">
  <img src="./logo.png" alt="dlab" width="220" />

  **🧪 Local-first experiments for fast deep learning research 🧪**
</div>

`dlab` is a small research framework for running controlled deep learning experiments from Hydra configs. It is built for local training loops, sequential sweeps, CSV-backed metrics, checkpoints, markdown reports, and notebook-based analysis.

The repo currently covers MNIST, Fashion-MNIST, and CIFAR-10 with classifiers and autoencoders built on PyTorch Lightning.

## Install

```bash
git clone git@github.com:tsilva/dlab.git
cd dlab
uv sync
```

Run a debug training pass:

```bash
uv run python train.py experiment=mnist_mlp trainer=debug dataset.num_workers=0 wandb.enabled=false
```

## Commands

```bash
uv run python train.py experiment=mnist_mlp                         # run a named experiment
uv run python train.py experiment=mnist_mlp optimizer.lr=1e-4       # override Hydra config values
uv run python train.py experiment=mnist_vae wandb.enabled=false     # disable W&B for a local-only run
uv run python train.py experiment=mnist_mlp litlogger.enabled=true  # enable LitLogger for a run
uv run python sweep.py sweep=lr                                     # run a sequential sweep
uv run python scripts/analyze.py summary outputs/mnist-mlp-baseline-adam-lr0p001-bs64-w256-d2-do0p1-seed1337
uv run python scripts/analyze.py compare outputs/a outputs/b        # compare runs by validation loss
uv run jupyter lab                                                  # open analysis notebooks
uv run ruff check .                                                 # lint the project
```

Run names are generated automatically from dataset, model, study, optimizer, learning rate, batch size, key model parameters, sweep metadata, and seed.

## Notes

- Python 3.12 or newer is required.
- `uv.lock` is present, so setup is through `uv sync`.
- Hydra configs in `configs/` are the main execution interface.
- Local datasets are cached under `datasets/`; generated runs, metrics, checkpoints, and resolved configs go under `outputs/`.
- Reports are written to `reports/` when `reports.enabled` is true.
- W&B is enabled by default; LitLogger is optional and disabled by default in `configs/train.yaml`.
- This is a local research workspace, not production ML infrastructure.

## Architecture

![Project architecture diagram](./architecture.png)
