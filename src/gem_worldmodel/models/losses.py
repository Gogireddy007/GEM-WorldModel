"""Latent prediction loss D(s_hat, s), plus a collapse-monitoring metric.

Representation collapse (all latents converging to a constant vector) is
flagged in the plan as the highest risk in the whole project (Week 4). We
track the per-dimension standard deviation of target-encoder outputs across
a batch as the collapse signal, it goes to ~0 exactly when collapse happens.
"""

import torch
import torch.nn.functional as F


def latent_prediction_loss(s_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """D(s_hat, s): MSE in the (stop-gradient) target latent space."""
    return F.mse_loss(s_hat, s.detach())


def embedding_std(latents: torch.Tensor) -> float:
    """Mean per-dimension std across the batch, the collapse-detection signal."""
    return latents.detach().std(dim=0).mean().item()
