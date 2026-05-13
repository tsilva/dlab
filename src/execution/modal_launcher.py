from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from src.execution.experiment import RunResult
from src.execution.launchers import LocalLauncher


class ModalLauncher:
    def launch_experiment(self, cfg: DictConfig) -> RunResult:
        try:
            import modal
        except ImportError as exc:
            raise RuntimeError(
                "Modal launcher requires the Modal SDK. Install it with `uv add modal`, "
                "then authenticate with `modal setup`."
            ) from exc

        launcher = cfg.launcher
        config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        app = modal.App(str(launcher.get("app_name", "dlab-train")))
        image = _build_modal_image(modal, launcher)
        secrets = [modal.Secret.from_name(name) for name in launcher.get("secrets", [])]

        @app.function(
            image=image,
            gpu=launcher.get("gpu"),
            timeout=int(launcher.get("timeout_seconds", 7200)),
            secrets=secrets,
            serialized=True,
        )
        def remote_train(config_yaml: str) -> dict:
            from omegaconf import OmegaConf

            from src.execution.experiment import run_experiment

            return run_experiment(OmegaConf.create(config_yaml)).to_dict()

        context = (
            modal.enable_output() if bool(launcher.get("show_progress", True)) else _nullcontext()
        )
        with context:
            with app.run():
                result = remote_train.remote(config_yaml)
        return RunResult(**result)

    def launch_train_command(self, command: Sequence[str]) -> int:
        return LocalLauncher().launch_train_command(command)


def _build_modal_image(modal, launcher: DictConfig):
    python_version = str(launcher.get("python_version", "3.12"))
    image = modal.Image.debian_slim(python_version=python_version)

    if bool(launcher.get("use_uv_sync", True)):
        image = image.uv_sync(".")
    else:
        image = image.pip_install_from_pyproject("pyproject.toml")

    image = image.workdir("/root")
    copy_source = bool(launcher.get("copy_source", False))
    return image.add_local_python_source("src", copy=copy_source)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False
