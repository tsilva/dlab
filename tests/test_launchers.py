from __future__ import annotations

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.execution.launchers import (
    LocalLauncher,
    format_override_value,
    get_launcher_name,
    launcher_cli_overrides,
    training_default_cli_overrides,
)
from src.execution.modal_launcher import ModalLauncher
from src.execution.runpod_flash_launcher import RunpodFlashLauncher


def test_get_launcher_name_defaults_to_local() -> None:
    cfg = OmegaConf.create({})

    assert get_launcher_name(cfg) == "local"


def test_launcher_cli_overrides_forward_nested_launcher_config() -> None:
    cfg = OmegaConf.create(
        {
            "launcher": {
                "name": "runpod_flash",
                "endpoint_name": "dlab-train",
                "gpu": "NVIDIA_GEFORCE_RTX_4090",
                "workers": [0, 2],
                "python_version": "3.12",
                "dependencies": ["torch>=2.4.0", "pytorch-lightning>=2.4.0"],
            }
        }
    )

    assert launcher_cli_overrides(cfg) == [
        "launcher=runpod_flash",
        "launcher.dependencies=['torch>=2.4.0','pytorch-lightning>=2.4.0']",
        "launcher.endpoint_name='dlab-train'",
        "launcher.gpu='NVIDIA_GEFORCE_RTX_4090'",
        "launcher.python_version='3.12'",
        "launcher.workers=[0,2]",
    ]


def test_format_override_value_quotes_strings_for_hydra_cli() -> None:
    assert format_override_value(["torch>=2.4.0", "3.12"]) == "['torch>=2.4.0','3.12']"


def test_training_default_cli_overrides_forward_modal_gpu_defaults() -> None:
    cfg = OmegaConf.create(
        {
            "runtime": {"float32_matmul_precision": "medium"},
            "dataset": {"num_workers": 16},
        }
    )

    assert training_default_cli_overrides(cfg) == [
        "runtime.float32_matmul_precision='medium'",
        "dataset.num_workers=16",
    ]


def test_local_launcher_loads_dotenv_before_running_experiment(monkeypatch, tmp_path) -> None:
    captured = {}
    (tmp_path / ".env").write_text(
        "CHECKPOINT_BUCKET_URI=s3://wandb\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHECKPOINT_BUCKET_URI", raising=False)

    def fake_run_experiment(_cfg):
        import os

        captured["checkpoint_bucket_uri"] = os.environ.get("CHECKPOINT_BUCKET_URI")
        return SimpleNamespace(run_dir="outputs/fake", metrics={})

    monkeypatch.setattr("src.execution.launchers.run_experiment", fake_run_experiment)

    result = LocalLauncher().launch_experiment(OmegaConf.create({}))

    assert result.run_dir == "outputs/fake"
    assert captured["checkpoint_bucket_uri"] == "s3://wandb"


def test_modal_launcher_uses_serialized_nested_function(monkeypatch) -> None:
    calls = {}
    build_steps = []

    class FakeRemoteFunction:
        def remote(self, _config_yaml: str) -> dict:
            return {"run_dir": "outputs/fake", "metrics": {}}

        def spawn(self, _config_yaml: str):
            return SimpleNamespace(
                object_id="fc-123",
                get_dashboard_url=lambda: "https://modal.test/call/fc-123",
            )

    class FakeApp:
        def __init__(self, name: str) -> None:
            calls["app_name"] = name

        def function(self, **kwargs):
            calls["function_kwargs"] = kwargs

            def decorator(_func):
                return FakeRemoteFunction()

            return decorator

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class FakeImage:
        @classmethod
        def debian_slim(cls, python_version: str):
            calls["python_version"] = python_version
            return cls()

        def uv_sync(self, path: str):
            calls["uv_sync"] = path
            build_steps.append("uv_sync")
            return self

        def add_local_python_source(self, module: str, copy: bool):
            calls["source"] = (module, copy)
            build_steps.append("add_local_python_source")
            return self

        def workdir(self, path: str):
            calls["workdir"] = path
            build_steps.append("workdir")
            return self

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Secret=SimpleNamespace(
            from_name=lambda name: name,
            from_dict=lambda values: ("env", values),
        ),
        Volume=SimpleNamespace(
            from_name=lambda name, create_if_missing=False: (name, create_if_missing)
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    cfg = OmegaConf.create(
        {
            "launcher": {
                "name": "modal",
                "app_name": "dlab-test",
                "gpu": "L4",
                "timeout_seconds": 60,
                "python_version": "3.12",
                "use_uv_sync": True,
                "copy_source": False,
                "detach": True,
                "wait_for_result": True,
                "nonpreemptible": True,
                "volumes": [
                    {
                        "name": "dlab-training-runs",
                        "mount_path": "/vol/dlab",
                        "create_if_missing": True,
                    }
                ],
                "show_progress": False,
                "secrets": [],
            }
        }
    )

    result = ModalLauncher().launch_experiment(cfg)

    assert result.run_dir == "outputs/fake"
    assert calls["function_kwargs"]["serialized"] is True
    assert calls["function_kwargs"]["nonpreemptible"] is True
    assert calls["function_kwargs"]["volumes"] == {
        "/vol/dlab": ("dlab-training-runs", True),
    }
    assert any(
        secret[0] == "env" and "WANDB_API_KEY" in secret[1]
        for secret in calls["function_kwargs"]["secrets"]
    )
    assert calls["run_kwargs"] == {"detach": True}
    assert calls["source"] == ("src", False)
    assert build_steps == ["uv_sync", "workdir", "add_local_python_source"]


def test_modal_launcher_can_submit_without_waiting(monkeypatch) -> None:
    calls = {}

    class FakeRemoteFunction:
        def remote(self, _config_yaml: str) -> dict:
            raise AssertionError("fire-and-forget launch should not wait for remote result")

        def spawn(self, _config_yaml: str):
            calls["spawned"] = True
            return SimpleNamespace(
                object_id="fc-123",
                get_dashboard_url=lambda: "https://modal.test/call/fc-123",
            )

    class FakeApp:
        def __init__(self, name: str) -> None:
            calls["app_name"] = name

        def function(self, **_kwargs):
            def decorator(_func):
                return FakeRemoteFunction()

            return decorator

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class FakeImage:
        @classmethod
        def debian_slim(cls, python_version: str):
            return cls()

        def uv_sync(self, _path: str):
            return self

        def add_local_python_source(self, _module: str, copy: bool):
            return self

        def workdir(self, _path: str):
            return self

    fake_modal = SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Secret=SimpleNamespace(
            from_name=lambda name: name,
            from_dict=lambda values: ("env", values),
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    cfg = OmegaConf.create(
        {
            "paths": {"outputs_dir": "outputs"},
            "launcher": {
                "name": "modal",
                "app_name": "dlab-test",
                "gpu": "L4",
                "timeout_seconds": 60,
                "python_version": "3.12",
                "use_uv_sync": True,
                "copy_source": False,
                "detach": True,
                "wait_for_result": False,
                "show_progress": False,
                "secrets": [],
            },
        }
    )

    result = ModalLauncher().launch_experiment(cfg)

    assert calls["spawned"] is True
    assert calls["run_kwargs"] == {"detach": True}
    assert result.run_dir == "outputs"
    assert result.metrics["modal/function_call_id"] == "fc-123"
    assert result.metrics["modal/function_call_dashboard_url"] == "https://modal.test/call/fc-123"


def test_runpod_flash_launcher_uses_deployable_endpoint_module(monkeypatch) -> None:
    calls = {}

    class FakeEndpoint:
        def __init__(self, name=None, *, gpu=None, workers=None, dependencies=None) -> None:
            calls["endpoint_kwargs"] = {
                "name": name,
                "gpu": gpu,
                "workers": workers,
                "dependencies": dependencies,
            }

        def __call__(self, _func):
            def remote_train(config_yaml: str) -> dict:
                calls["config_yaml"] = config_yaml
                return {
                    "id": "job-123",
                    "status": "COMPLETED",
                    "output": {"run_dir": "outputs/runpod", "metrics": {}},
                }

            return remote_train

    fake_runpod_flash = SimpleNamespace(
        Endpoint=FakeEndpoint,
        GpuGroup=SimpleNamespace(ANY="any-gpu"),
        GpuType=SimpleNamespace(NVIDIA_GEFORCE_RTX_4090="rtx-4090", NVIDIA_L4="l4"),
    )
    monkeypatch.delitem(sys.modules, "src.execution.runpod_flash_app", raising=False)
    monkeypatch.setitem(sys.modules, "runpod_flash", fake_runpod_flash)

    cfg = OmegaConf.create(
        {
            "launcher": {
                "name": "runpod_flash",
                "endpoint_name": "dlab-train",
                "gpu": "NVIDIA_GEFORCE_RTX_4090",
                "workers": [0, 1],
                "dependencies": ["torch>=2.4.0"],
                "python_version": "3.12",
            }
        }
    )

    result = RunpodFlashLauncher().launch_experiment(cfg)

    assert result.run_dir == "outputs/runpod"
    assert calls["endpoint_kwargs"] == {
        "name": "dlab-train",
        "gpu": "l4",
        "workers": (0, 1),
        "dependencies": [
            "einops>=0.8.0",
            "hydra-core>=1.3.2",
            "litlogger>=2026.5.12",
            "matplotlib>=3.9.0",
            "numpy>=2.0.0",
            "omegaconf>=2.3.0",
            "pandas>=2.2.0",
            "pillow>=10.4.0",
            "pytorch-lightning>=2.4.0",
            "scikit-learn>=1.5.0",
            "timm>=1.0.0",
            "torch>=2.4.0",
            "torchvision>=0.19.0",
            "transformers>=4.44.0",
            "wandb>=0.17.0",
        ],
    }
    assert "python_version: '3.12'" in calls["config_yaml"]
