"""Week 6: activation intervention, perturb the latent dimension identified
by probing.py as most predictive of oligotroph/copiotroph status, and check
whether the fine-tuned growth-rate head's prediction actually shifts.

This is the corroborating half of Week 6: a probe finding correlation isn't
enough (see the circularity caveat in probing.py), intervention tests
whether that dimension is causally load-bearing for the model's own growth-
rate prediction, not just statistically associated with the label.
"""

import numpy as np
import torch

from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def intervene_on_dimension(
    latents: torch.Tensor,
    head: GrowthRateHead,
    dim: int,
    deltas: list[float] | None = None,
) -> dict:
    """Sweep dimension `dim` of the joint latent by `deltas` (in units of that
    dimension's own std across the batch) and record how the head's growth-rate
    prediction shifts.
    """
    deltas = deltas if deltas is not None else [-2.0, -1.0, 0.0, 1.0, 2.0]
    dim_std = latents[:, dim].std().item()
    if dim_std == 0:
        logger.warning(f"latent dimension {dim} has zero variance in this batch; intervention is degenerate")

    baseline_pred = head(latents).detach()
    shifts = {}
    for delta in deltas:
        perturbed = latents.clone()
        perturbed[:, dim] = perturbed[:, dim] + delta * dim_std
        with torch.no_grad():
            pred = head(perturbed)
        shifts[delta] = (pred - baseline_pred).mean().item()

    return {
        "dim": dim,
        "dim_std": dim_std,
        "baseline_pred_mean": baseline_pred.mean().item(),
        "shift_by_delta": shifts,
        "monotonic": _is_monotonic(list(shifts.values())),
    }


def _is_monotonic(values: list[float]) -> bool:
    diffs = np.diff(values)
    return bool(np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6))
