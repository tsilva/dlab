from __future__ import annotations

from omegaconf import OmegaConf

from src.utils.run_target import collect_run_target, run_target_summary


def test_collect_run_target_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_APP_ID", raising=False)
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("RUNPOD_GPU_COUNT", raising=False)
    cfg = OmegaConf.create({"launcher": {"name": "local"}})

    target = collect_run_target(cfg)

    assert target["provider"] == "local"
    assert target["launcher"] == "local"
    assert target["name"].startswith("local:")
    assert target["cpu"]["count_logical"] is not None
    assert "cuda_available" in target["gpu"]


def test_collect_run_target_records_remote_launcher_gpu() -> None:
    cfg = OmegaConf.create({"launcher": {"name": "modal", "gpu": "L4"}})

    target = collect_run_target(cfg)
    summary = run_target_summary(target)

    assert target["provider"] == "modal"
    assert target["name"] == "modal:L4"
    assert summary["target/provider"] == "modal"
    assert summary["target/launcher"] == "modal"
    assert summary["target/gpu_requested"] == "L4"
