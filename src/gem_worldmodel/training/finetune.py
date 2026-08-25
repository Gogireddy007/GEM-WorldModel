"""Attach the growth-rate head on top of the pretrained context
encoder's joint representation (all branches, unmasked) and fine-tune on the
labeled subset.
"""

import numpy as np
import torch

from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.models.jepa import JEPA
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
) -> dict:
    train_cfg = train_cfg or load_config("train")
    ft_cfg = train_cfg["finetune"]
    set_seed(train_cfg["seed"])

    n = target_log_doubling_time.shape[0]
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
