"""Latent prediction loss D(s_hat, s), plus a collapse-monitoring metric.

Representation collapse (all latents converging to a constant vector) is
the highest risk in the whole pretraining step. We
track the per-dimension standard deviation of target-encoder outputs across
a batch as the collapse signal, it goes to ~0 exactly when collapse happens.

The opposite direction matters too: plain MSE on unnormalized latents has a
degenerate way to shrink the loss, uniformly scale every embedding up, since
MSE isn't scale-invariant. This is exactly what showed up combining the small
labeled corpus with the much larger GEM corpus: many more optimizer steps per
epoch than the single-corpus runs ever had, and loss climbed steadily over
200 epochs even with gradient clipping on. L2-normalizing both s_hat and s
before the loss (the same trick SimSiam/BYOL use) removes that degenerate
direction entirely, since a unit-norm vector can't drift in scale.
"""

import torch
import torch.nn.functional as F


def latent_prediction_loss(s_hat: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """D(s_hat, s): MSE between L2-normalized vectors, in the (stop-gradient)
    target latent space.
    """
    s_hat_n = F.normalize(s_hat, dim=-1)
    s_n = F.normalize(s.detach(), dim=-1)
    return F.mse_loss(s_hat_n, s_n)


def embedding_std(latents: torch.Tensor) -> float:
    """Mean per-dimension std across the batch, the collapse-detection signal."""
    return latents.detach().std(dim=0).mean().item()
