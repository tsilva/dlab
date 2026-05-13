from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder(nn.Module):
    task = "reconstruction"

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 32,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        input_dim = math.prod(self.input_shape)

        encoder_layers: list[nn.Module] = [nn.Flatten()]
        prev = input_dim
        for width in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        encoder_layers.append(nn.Linear(prev, latent_dim))

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        decoder_layers.extend([nn.Linear(prev, input_dim), nn.Sigmoid()])

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z).view(x.shape[0], *self.input_shape)
        return {"recon": recon, "z": z}


class VAE(nn.Module):
    task = "vae"

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 16,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        self.input_shape = tuple(input_shape)
        input_dim = math.prod(self.input_shape)

        layers: list[nn.Module] = [nn.Flatten()]
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        decoder_layers.extend([nn.Linear(prev, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(z.shape[0], *self.input_shape)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view_as(z_perm).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.commitment_cost * commitment_loss
        quantized = z + (quantized - z).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        utilization = (avg_probs > 0).float().mean()
        return {
            "quantized": quantized,
            "vq_loss": loss,
            "perplexity": perplexity,
            "codebook_utilization": utilization,
            "encoding_indices": encoding_indices,
        }


class VQVAE(nn.Module):
    task = "vqvae"

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_embeddings: int = 128,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        channels = int(input_shape[0])
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embedding_dim, 1),
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        q = self.quantizer(z)
        recon = self.decoder(q["quantized"])
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"recon": recon, "z": q["quantized"], **q}
