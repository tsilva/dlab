from __future__ import annotations

from omegaconf import OmegaConf

from src.sweeps.wandb import _agent_command, build_wandb_sweep_config


def test_build_wandb_sweep_config_includes_hydra_parameters_and_study_metadata() -> None:
    cfg = OmegaConf.create(
        {
            "name": "mlp_lr_sweep",
            "experiment": "mnist_mlp_research",
            "train_script": "train.py",
            "backend": "wandb",
            "parameters": {"optimizer": {"lr": [0.001, 0.0003]}},
            "run": {
                "stage": "01_mlp_basics",
                "study": "mlp_lr",
                "question": "How does LR affect convergence?",
                "tags": ["roadmap", "optimization"],
            },
            "wandb_sweep": {
                "method": "grid",
                "metric": {"name": "val/loss", "goal": "minimize"},
            },
            "wandb": {"project": "dlab", "entity": None},
        }
    )

    sweep_config = build_wandb_sweep_config(cfg)

    assert sweep_config["program"] == "train.py"
    assert sweep_config["method"] == "grid"
    assert sweep_config["parameters"]["experiment"] == {"value": "'mnist_mlp_research'"}
    assert sweep_config["parameters"]["wandb.enabled"] == {"value": True}
    assert sweep_config["parameters"]["wandb.project"] == {"value": "'dlab'"}
    assert sweep_config["parameters"]["optimizer.lr"] == {"values": [0.001, 0.0003]}
    assert sweep_config["parameters"]["run.stage"] == {"value": "'01_mlp_basics'"}
    assert sweep_config["parameters"]["run.study"] == {"value": "'mlp_lr'"}
    assert sweep_config["parameters"]["run.tags"] == {"value": ["roadmap", "optimization"]}


def test_agent_command_attaches_to_existing_sweep_with_launcher() -> None:
    cfg = OmegaConf.create(
        {
            "name": "mlp_lr_sweep",
            "backend": "wandb",
            "wandb": {"project": "dlab", "entity": None},
            "wandb_sweep": {"id": "abc123", "count": 1},
            "launcher": {"name": "modal_gpu"},
        }
    )

    command = _agent_command(cfg, "abc123")

    assert "sweep=mlp_lr" in command
    assert "launcher=modal_gpu" in command
    assert "wandb_sweep.create=false" in command
    assert "wandb_sweep.start_agent=true" in command
    assert "wandb_sweep.agents=1" in command
    assert "wandb_sweep.id=abc123" in command
    assert "wandb_sweep.count=1" in command
