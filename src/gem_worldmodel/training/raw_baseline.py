"""Sanity check: does going through the pretrained JEPA context encoder
actually add value over just feeding the raw (standardized) branch features
straight into a growth-rate head, with no encoder at all?

This has to be checked directly, not assumed. Every JEPA-based number
reported so far compares favorably against gRodon/Phydon-style baselines in
one specific way (see FINDINGS.md), but none of that says whether the
pretrained encoder itself is earning its place versus a trivial
concatenation of the same standardized inputs. If the raw-feature baseline
does comparably or better, that's a real, important finding about this
architecture at this corpus size, not a reason to hide the comparison.

Uses the exact same k-fold cross-validation, StratifiedKFold with the same
seed and stratify_labels as training/finetune.py:cross_validate, so the fold
membership is IDENTICAL between the two, this is the fair, apples-to-apples
comparison, not two different random splits that happen to look similar.
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger
from gem_worldmodel.utils.torch_utils import eval_mode

logger = get_logger(__name__)


def concat_branch_tensors(branch_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate every branch into one raw feature vector, in a fixed
    (alphabetical) order so this is deterministic regardless of dict
    iteration order. Same standardized inputs the JEPA encoder receives,
    just without going through it.
    """
    names = sorted(branch_tensors.keys())
    return torch.cat([branch_tensors[name] for name in names], dim=-1)


def cross_validate_raw(
    branch_tensors: dict[str, torch.Tensor],
    target_log_doubling_time: torch.Tensor,
    stratify_labels: np.ndarray,
    train_cfg: dict | None = None,
    head_cfg: dict | None = None,
    k: int = 5,
) -> dict:
    """Same k-fold CV pattern as training/finetune.py:cross_validate: every
    sample gets exactly one out-of-fold prediction, from a head trained fresh
    each fold (no leakage). No encoder, no pretraining, no fine-tuning
    schedule to worry about, since there's nothing upstream of the head to
    freeze or unfreeze, just the raw concatenated features straight in.
    """
    train_cfg = train_cfg or load_config("train")
    head_cfg = head_cfg or load_config("model")["growth_rate_head"]
    ft_cfg = train_cfg["finetune"]
    seed = train_cfg["seed"]

    x = concat_branch_tensors(branch_tensors)
    n = x.shape[0]
    in_dim = x.shape[1]

    oof_pred_log = np.full(n, np.nan)
    fold_id = np.full(n, -1)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(n), stratify_labels)):
        torch.manual_seed(seed + fold)
        rng = np.random.default_rng(seed + fold)
        shuffled = rng.permutation(train_val_idx)
        n_val = max(1, int(len(shuffled) * ft_cfg["val_frac"]))
        train_idx = shuffled[n_val:]  # val split isn't used for anything here, no early stopping
        test_idx_t = torch.tensor(test_idx)
        train_idx_t = torch.tensor(train_idx)

        head = GrowthRateHead(in_dim, head_cfg["hidden_dim"], head_cfg["num_layers"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=ft_cfg["lr"], weight_decay=ft_cfg["weight_decay"])

        for _ in range(ft_cfg["epochs"]):
            head.train()
            pred = head(x[train_idx_t])
            loss = torch.nn.functional.mse_loss(pred, target_log_doubling_time[train_idx_t])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad(), eval_mode(head):
            oof_pred_log[test_idx] = head(x[test_idx_t]).numpy()
        fold_id[test_idx] = fold
        logger.info(f"raw-baseline fold {fold}: test n={len(test_idx)}")

    assert (fold_id >= 0).all(), "every sample should be held out exactly once across folds"
    return {"oof_pred_log": oof_pred_log, "fold_id": fold_id, "k": k}
