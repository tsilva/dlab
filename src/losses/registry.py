from __future__ import annotations

from torch.nn import functional as F

LOSS_REGISTRY = {
    "cross_entropy": F.cross_entropy,
    "mse": F.mse_loss,
    "bce": F.binary_cross_entropy,
}

