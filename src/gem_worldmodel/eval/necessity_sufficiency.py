"""Necessity/sufficiency masking per branch (CUB/genomic-traits vs.
GTDB-distance vs. 16S baseline), compared separately for <5h and >5h doubling
time, cross-checked against the benchmark run.

Necessity: zero out one branch at inference time (replace with its batch
mean, i.e. "uninformative"), measure the performance drop. A branch whose
removal hurts a lot is necessary.
Sufficiency: zero out every OTHER branch, keep only this one, measure
absolute performance. A branch that alone still predicts well is sufficient.
"""

import numpy as np
import torch

from gem_worldmodel.eval.benchmark import compute_metrics
from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def _neutralize(branch_tensors: dict[str, torch.Tensor], keep: set[str]) -> dict[str, torch.Tensor]:
    """Replace every branch NOT in `keep` with its own batch mean (removes its
    per-sample information while keeping tensor shapes intact for the encoder).
    """
    out = {}
    for name, tensor in branch_tensors.items():
        if name in keep:
            out[name] = tensor
        else:
            mean = tensor.mean(dim=0, keepdim=True)
            out[name] = mean.expand_as(tensor)
    return out


def evaluate(
    jepa: JEPA, head: GrowthRateHead, branch_tensors: dict[str, torch.Tensor], y_true_log: np.ndarray
) -> dict[str, float]:
    with torch.no_grad():
        z = jepa.context_encoder(branch_tensors)
        pred_log = head(z).numpy()
    return compute_metrics(y_true_log, pred_log)


def necessity_sufficiency_report(
    jepa: JEPA,
    head: GrowthRateHead,
    branch_tensors: dict[str, torch.Tensor],
    y_true_log: np.ndarray,
    regime_mask: np.ndarray | None = None,
) -> dict:
    branch_names = list(branch_tensors.keys())
    if regime_mask is not None:
        branch_tensors = {name: tensor[regime_mask] for name, tensor in branch_tensors.items()}
        y_true_log = y_true_log[regime_mask]

    full_metrics = evaluate(jepa, head, branch_tensors, y_true_log)

    necessity, sufficiency = {}, {}
    for branch in branch_names:
        keep_all_but = set(branch_names) - {branch}
        necessity[branch] = {
            "metrics": evaluate(jepa, head, _neutralize(branch_tensors, keep_all_but), y_true_log),
            "r2_drop": full_metrics["r2"] - evaluate(jepa, head, _neutralize(branch_tensors, keep_all_but), y_true_log)["r2"],
        }
        sufficiency[branch] = evaluate(jepa, head, _neutralize(branch_tensors, {branch}), y_true_log)

    return {"full_metrics": full_metrics, "necessity": necessity, "sufficiency": sufficiency, "n": len(y_true_log)}
