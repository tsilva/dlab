from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from src.utils.optimizers import build_optimizer


class ResearchLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.task = cfg.get("task", getattr(model, "task", "classification"))
        self.beta = float(cfg.get("loss", {}).get("beta", 1.0))
        self.label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
        self.learning_rate = float(cfg.optimizer.get("lr", 1e-3))
        self.example_batch: torch.Tensor | None = None
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.parameters(), self.cfg.optimizer)
        scheduler_cfg = self.cfg.optimizer.get("scheduler")
        if not scheduler_cfg or scheduler_cfg.get("name", "constant") in {None, "constant", "none"}:
            return optimizer
        if scheduler_cfg.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.cfg.trainer.max_epochs),
                eta_min=float(scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        if scheduler_cfg.name == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(scheduler_cfg.get("max_lr", self.cfg.optimizer.lr)),
                total_steps=int(self.trainer.estimated_stepping_batches),
                pct_start=float(scheduler_cfg.get("pct_start", 0.25)),
                div_factor=float(scheduler_cfg.get("div_factor", 10.0)),
                final_div_factor=float(scheduler_cfg.get("final_div_factor", 100.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        raise KeyError(f"Unknown scheduler '{scheduler_cfg.name}'")

    def training_step(self, batch, batch_idx: int):
        loss, metrics = self._shared_step(batch, "train")
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)
        self._log_lr()
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss, metrics = self._shared_step(batch, "val")
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        if batch_idx == 0 and isinstance(batch, (tuple, list)):
            self.example_batch = batch[0].detach()[:16]
        return loss

    def test_step(self, batch, batch_idx: int):
        _, metrics = self._shared_step(batch, "test")
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def on_before_optimizer_step(self, optimizer) -> None:
        grad_norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(grad_norms or [torch.tensor(0.0, device=self.device)]),
            2,
        )
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)
        self._log_gradient_clip(total_norm)
        self._log_gradient_flow()

    def on_validation_epoch_end(self) -> None:
        if self.example_batch is None or self.logger is None:
            return
        if self.task not in {"reconstruction", "vae", "vqvae"}:
            self._log_conv_filters()
            return

        x = self.example_batch.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        recon = out["recon"].detach().clamp(0, 1)
        if self.cfg.dataset.get("normalize", True):
            originals = _unnormalize_if_needed(x.detach().clamp(-3, 3), self.cfg.dataset.name)
        else:
            originals = x.detach().clamp(0, 1)
        images = torch.cat([originals.cpu(), recon.cpu()], dim=0)
        self._wandb_log_images("reconstructions", images)
        if self.task == "vae":
            self._log_vae_latent_traversal(out["z"])

    def _shared_step(self, batch, prefix: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, y = batch
        if self.task == "classification":
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            return loss, {f"{prefix}/loss": loss, f"{prefix}/acc": acc}

        out = self.model(x)
        recon = out["recon"]
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss
        metrics = {f"{prefix}/loss": loss, f"{prefix}/recon_loss": recon_loss}

        if self.task == "vae":
            mu = out["mu"]
            logvar = out["logvar"]
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon_loss + self.beta * kl
            metrics[f"{prefix}/loss"] = loss
            metrics[f"{prefix}/kl_loss"] = kl
            metrics[f"{prefix}/beta"] = torch.tensor(self.beta, device=self.device)
        elif self.task == "vqvae":
            vq_loss = out["vq_loss"]
            loss = recon_loss + vq_loss
            metrics[f"{prefix}/loss"] = loss
            metrics[f"{prefix}/vq_loss"] = vq_loss
            metrics[f"{prefix}/codebook_perplexity"] = out["perplexity"]
            metrics[f"{prefix}/codebook_utilization"] = out["codebook_utilization"]
        return loss, metrics

    def _log_lr(self) -> None:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=False)

    def _log_gradient_clip(self, raw_norm: torch.Tensor) -> None:
        threshold = float(self.cfg.trainer.get("gradient_clip_val", 0.0) or 0.0)
        threshold_tensor = torch.tensor(threshold, device=self.device)
        if threshold > 0:
            eps = torch.tensor(1e-12, device=self.device)
            clip_coef = torch.minimum(
                torch.tensor(1.0, device=self.device),
                threshold_tensor / (raw_norm + eps),
            )
            was_clipped = (raw_norm > threshold_tensor).to(dtype=torch.float32)
        else:
            clip_coef = torch.tensor(1.0, device=self.device)
            was_clipped = torch.tensor(0.0, device=self.device)

        self.log_dict(
            {
                "train/grad_clip/raw_norm": raw_norm,
                "train/grad_clip/threshold": threshold_tensor,
                "train/grad_clip/was_clipped": was_clipped,
                "train/grad_clip/clip_coef": clip_coef,
                "train/grad_clip/clipped_norm_estimate": raw_norm * clip_coef,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

    def _log_gradient_flow(self) -> None:
        gradient_flow_cfg = self.cfg.get("gradient_flow", {})
        if not gradient_flow_cfg.get("enabled", False):
            return
        log_every = int(gradient_flow_cfg.get("log_every_n_steps", 50))
        if log_every > 1 and self.global_step % log_every != 0:
            return

        layer_norms = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Conv2d | nn.Linear):
                continue
            weight = module.weight
            if weight.grad is None:
                continue
            layer_norms.append((name, weight.grad.detach().norm(2)))
        if not layer_norms:
            return

        norms = torch.stack([norm for _, norm in layer_norms])
        first = norms[0]
        middle = norms[len(norms) // 2]
        last = norms[-1]
        min_norm = norms.min()
        max_norm = norms.max()
        eps = torch.tensor(1e-12, device=self.device)
        threshold = float(gradient_flow_cfg.get("vanishing_threshold", 1e-8))
        dead_layers = (norms < threshold).sum().to(dtype=torch.float32)

        self.log_dict(
            {
                "train/grad_flow/first_layer_norm": first,
                "train/grad_flow/middle_layer_norm": middle,
                "train/grad_flow/last_layer_norm": last,
                "train/grad_flow/min_layer_norm": min_norm,
                "train/grad_flow/max_layer_norm": max_norm,
                "train/grad_flow/first_to_last_ratio": first / (last + eps),
                "train/grad_flow/min_to_max_ratio": min_norm / (max_norm + eps),
                "train/grad_flow/dead_layers": dead_layers,
                "train/grad_flow/layer_count": torch.tensor(
                    float(len(layer_norms)),
                    device=self.device,
                ),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

    def _log_conv_filters(self) -> None:
        if not hasattr(self.model, "first_layer_filters"):
            return
        filters = self.model.first_layer_filters()
        if filters is None:
            return
        self._wandb_log_images("filters", filters[:16])

    def _wandb_log_images(self, key: str, images: torch.Tensor) -> None:
        try:
            import torchvision

            import wandb
        except ImportError:
            return
        grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
        experiment = getattr(self.logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "log"):
            experiment.log({key: wandb.Image(grid)}, step=self.global_step)

    def _log_vae_latent_traversal(self, z: torch.Tensor) -> None:
        if not hasattr(self.model, "decode"):
            return
        latent_dim = z.shape[1]
        dims = min(4, latent_dim)
        values = torch.linspace(-3, 3, steps=8, device=self.device)
        samples = []
        for dim in range(dims):
            traversal = torch.zeros(values.numel(), latent_dim, device=self.device)
            traversal[:, dim] = values
            with torch.no_grad():
                samples.append(self.model.decode(traversal).detach().cpu())
        self._wandb_log_images("latent_traversals", torch.cat(samples, dim=0))


def _unnormalize_if_needed(x: torch.Tensor, dataset_name: str) -> torch.Tensor:
    from src.datasets.vision import normalization_stats

    mean_values, std_values = normalization_stats(dataset_name)
    mean = torch.tensor(mean_values, device=x.device).view(1, len(mean_values), 1, 1)
    std = torch.tensor(std_values, device=x.device).view(1, len(std_values), 1, 1)
    return (x * std + mean).clamp(0, 1)
