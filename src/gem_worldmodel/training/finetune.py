"""Attach the growth-rate head on top of the pretrained context
encoder's joint representation (all branches, unmasked) and fine-tune on the
labeled subset.

At 175 species, a single 70/15/15 train/val/test split leaves a 27-sample
test set, too small for the benchmark metrics on it to mean much (a single
outlier can swing RMSE wildly, and R2/Spearman are noisy at that n).
`cross_validate` fixes this by running k-fold CV and returning one
out-of-fold prediction per species, every species gets evaluated on a fold
where it was held out, and the benchmark can then compute its metrics once
over the full n instead of once over a 27-sample slice.
"""

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger
from gem_worldmodel.utils.seed import set_seed

logger = get_logger(__name__)


def split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def finetune(
    jepa: JEPA,
    branch_tensors: dict[str, torch.Tensor],
    target_log_doubling_time: torch.Tensor,
    train_cfg: dict | None = None,
    splits: dict[str, np.ndarray] | None = None,
) -> dict:
    train_cfg = train_cfg or load_config("train")
    ft_cfg = train_cfg["finetune"]
    set_seed(train_cfg["seed"])

    n = target_log_doubling_time.shape[0]
    if splits is None:
        splits = split_indices(n, ft_cfg["train_frac"], ft_cfg["val_frac"], train_cfg["seed"])

    head_cfg = load_config("model")["growth_rate_head"]
    head = GrowthRateHead(jepa.latent_dim, head_cfg["hidden_dim"], head_cfg["num_layers"])

    for p in jepa.context_encoder.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + list(jepa.context_encoder.parameters()),
        lr=ft_cfg["lr"], weight_decay=ft_cfg["weight_decay"],
    )

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(ft_cfg["epochs"]):
        if epoch < ft_cfg["freeze_encoder_epochs"]:
            for p in jepa.context_encoder.parameters():
                p.requires_grad_(False)
        else:
            for p in jepa.context_encoder.parameters():
                p.requires_grad_(True)

        train_idx = torch.tensor(splits["train"])
        train_batch = {name: tensor[train_idx] for name, tensor in branch_tensors.items()}
        z = jepa.context_encoder(train_batch)
        pred = head(z)
        loss = torch.nn.functional.mse_loss(pred, target_log_doubling_time[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["train_loss"].append(loss.item())

        with torch.no_grad():
            val_idx = torch.tensor(splits["val"])
            if len(val_idx) > 0:
                val_batch = {name: tensor[val_idx] for name, tensor in branch_tensors.items()}
                val_pred = head(jepa.context_encoder(val_batch))
                val_loss = torch.nn.functional.mse_loss(val_pred, target_log_doubling_time[val_idx]).item()
            else:
                val_loss = float("nan")
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == ft_cfg["epochs"] - 1:
            logger.info(f"finetune epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    return {"head": head, "jepa": jepa, "history": history, "splits": splits}


def cross_validate(
    model_cfg: dict,
    checkpoint_path,
    branch_tensors: dict[str, torch.Tensor],
    target_log_doubling_time: torch.Tensor,
    stratify_labels: np.ndarray,
    train_cfg: dict | None = None,
    k: int = 5,
) -> dict:
    """K-fold CV: every species is held out exactly once, fine-tuning starts
    fresh from the same pretrained checkpoint each fold (no leakage between
    folds), and the returned predictions are one out-of-fold prediction per
    species covering the whole dataset, not a metric averaged over k small
    per-fold test sets. `stratify_labels` should be the fast/slow class (or
    any other class you want folds balanced on).
    """
    train_cfg = train_cfg or load_config("train")
    seed = train_cfg["seed"]
    n = target_log_doubling_time.shape[0]

    oof_pred_log = np.full(n, np.nan)
    fold_id = np.full(n, -1)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(n), stratify_labels)):
        rng = np.random.default_rng(seed + fold)
        shuffled = rng.permutation(train_val_idx)
        n_val = max(1, int(len(shuffled) * train_cfg["finetune"]["val_frac"]))
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        jepa = load_checkpoint(model_cfg, checkpoint_path)
        result = finetune(jepa, branch_tensors, target_log_doubling_time, train_cfg, splits=splits)

        with torch.no_grad():
            test_idx_t = torch.tensor(test_idx)
            test_batch = {name: tensor[test_idx_t] for name, tensor in branch_tensors.items()}
            z = result["jepa"].context_encoder(test_batch)
            pred = result["head"](z).numpy()

        oof_pred_log[test_idx] = pred
        fold_id[test_idx] = fold
        logger.info(f"fold {fold}: test n={len(test_idx)}")

    assert (fold_id >= 0).all(), "every sample should be held out exactly once across folds"
    return {"oof_pred_log": oof_pred_log, "fold_id": fold_id, "k": k}
