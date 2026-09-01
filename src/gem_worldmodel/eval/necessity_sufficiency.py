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


def _neutralize(
    branch_tensors: dict[str, torch.Tensor],
    keep: set[str],
    mean_source: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Replace every branch NOT in `keep` with a fixed per-branch mean (removes
    its per-sample information while keeping tensor shapes intact for the
    encoder). `mean_source`, if given, is where that mean is computed from,
    pass the fold's TRAIN tensors here so the neutralization never uses any
    statistic derived from the held-out test samples themselves.
    """
    mean_source = mean_source or branch_tensors
    out = {}
    for name, tensor in branch_tensors.items():
        if name in keep:
            out[name] = tensor
        else:
            mean = mean_source[name].mean(dim=0, keepdim=True)
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


def necessity_sufficiency_report_cv(
    fold_models: list[dict],
    branch_tensors: dict[str, torch.Tensor],
    y_true_log: np.ndarray,
    regime_mask: np.ndarray | None = None,
) -> dict:
    """Same necessity/sufficiency masking as necessity_sufficiency_report, but
    run per cross-validation fold on that fold's own held-out test samples
    with that fold's own fine-tuned (jepa, head), instead of evaluating a
    single model on data it was fine-tuned on. Predictions are pooled across
    folds before computing metrics once, the same out-of-fold pattern used
    for the benchmark in training/finetune.py:cross_validate, so every
    species contributes a genuinely held-out ablated prediction.

    Branch-neutralization means come from each fold's own TRAIN split, never
    from its test split, so no test-set statistic leaks into the ablation.
    """
    branch_names = list(branch_tensors.keys())
    n = len(y_true_log)
    in_regime = regime_mask if regime_mask is not None else np.ones(n, dtype=bool)

    oof_pred = np.full(n, np.nan)
    oof_necessity = {b: np.full(n, np.nan) for b in branch_names}
    oof_sufficiency = {b: np.full(n, np.nan) for b in branch_names}
    covered = np.zeros(n, dtype=bool)

    for fold in fold_models:
        jepa, head, full_test_idx = fold["jepa"], fold["head"], fold["test_idx"]
        # Neutralization means always come from this fold's train split (never
        # its test split), whether or not we're restricting evaluation to a regime.
        train_idx = np.setdiff1d(np.arange(n), full_test_idx)
        train_tensors = {name: tensor[train_idx] for name, tensor in branch_tensors.items()}

        test_idx = full_test_idx[in_regime[full_test_idx]]
        if len(test_idx) == 0:
            continue
        test_tensors = {name: tensor[test_idx] for name, tensor in branch_tensors.items()}

        with torch.no_grad():
            oof_pred[test_idx] = head(jepa.context_encoder(test_tensors)).numpy()
        covered[test_idx] = True

        for branch in branch_names:
            keep_all_but = set(branch_names) - {branch}
            with torch.no_grad():
                nec_batch = _neutralize(test_tensors, keep_all_but, mean_source=train_tensors)
                oof_necessity[branch][test_idx] = head(jepa.context_encoder(nec_batch)).numpy()
                suf_batch = _neutralize(test_tensors, {branch}, mean_source=train_tensors)
                oof_sufficiency[branch][test_idx] = head(jepa.context_encoder(suf_batch)).numpy()

    assert (covered == in_regime).all(), "every in-regime sample should be held out exactly once, no others"

    y_regime = y_true_log[in_regime]
    full_metrics = compute_metrics(y_regime, oof_pred[in_regime])
    necessity, sufficiency = {}, {}
    for branch in branch_names:
        nec_metrics = compute_metrics(y_regime, oof_necessity[branch][in_regime])
        necessity[branch] = {"metrics": nec_metrics, "r2_drop": full_metrics["r2"] - nec_metrics["r2"]}
        sufficiency[branch] = compute_metrics(y_regime, oof_sufficiency[branch][in_regime])

    return {"full_metrics": full_metrics, "necessity": necessity, "sufficiency": sufficiency, "n": int(in_regime.sum())}
