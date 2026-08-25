"""Week 4: self-supervised masked-branch pretraining, with explicit
collapse-detection since the plan calls this the highest-risk step in the
whole project.

Collapse is flagged when the mean per-dimension std of target-encoder outputs
(embedding_std, see models/losses.py), averaged over a trailing window, drops
below `collapse_std_threshold`. This is a monitoring signal, not an automatic
fix, a collapsing run should be stopped and its hyperparameters (EMA decay,
LR, latent_dim) revisited, per the plan's Week 4 buffer.

The opposite failure mode, representation EXPLOSION rather than collapse, is real too: an early run on the actual 175-species labeled corpus (not the
synthetic data used in unit tests) diverged from loss=0.02 to loss=24 over
200 epochs with no gradient clipping. JEPA/BYOL-style architectures are known
to be prone to this without it. Gradient clipping (`grad_clip_norm` below) is
the fix, and ExplosionMonitor gives the same kind of trailing-window warning
CollapseMonitor gives for the opposite failure, so a diverging run is flagged
during training rather than silently producing a garbage checkpoint.
"""

from pathlib import Path

import torch

from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.models.masking import BranchMasker
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger
from gem_worldmodel.utils.seed import set_seed

logger = get_logger(__name__)


class CollapseMonitor:
    def __init__(self, window: int, threshold: float):
        self.window = window
        self.threshold = threshold
        self.history: list[float] = []

    def update(self, std: float) -> bool:
        """Record `std`, return True if a collapse is currently flagged."""
        self.history.append(std)
        self.history = self.history[-self.window :]
        if len(self.history) < self.window:
            return False
        return (sum(self.history) / len(self.history)) < self.threshold


class ExplosionMonitor:
    """The opposite of CollapseMonitor: flags when loss is growing, not shrinking."""

    def __init__(self, window: int, growth_factor: float = 5.0):
        self.window = window
        self.growth_factor = growth_factor
        self.history: list[float] = []

    def update(self, loss: float) -> bool:
        self.history.append(loss)
        self.history = self.history[-self.window :]
        if len(self.history) < self.window:
            return False
        early = sum(self.history[: self.window // 2]) / (self.window // 2)
        late = sum(self.history[self.window // 2 :]) / (self.window - self.window // 2)
        return early > 0 and (late / early) > self.growth_factor


def pretrain(
    branch_tensors: dict[str, torch.Tensor],
    model_cfg: dict | None = None,
    train_cfg: dict | None = None,
    epochs: int | None = None,
) -> dict:
    model_cfg = model_cfg or load_config("model")
    train_cfg = train_cfg or load_config("train")
    set_seed(train_cfg["seed"])

    model = JEPA(model_cfg)
    # Both context_encoder AND predictor must be optimized, the predictor is
    # not covered by context_encoder.parameters(). Missing it here (an earlier
    # bug) meant the predictor stayed randomly initialized forever, which is
    # what actually caused the loss-explosion failure mode gradient clipping
    # alone didn't fix (see module docstring).
    trainable_params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["pretrain"]["lr"],
        weight_decay=train_cfg["pretrain"]["weight_decay"],
    )

    n_samples = next(iter(branch_tensors.values())).shape[0]
    batch_size = min(train_cfg["pretrain"]["batch_size"], n_samples)
    n_epochs = epochs if epochs is not None else train_cfg["pretrain"]["epochs"]

    grad_clip_norm = train_cfg["pretrain"].get("grad_clip_norm", 1.0)
    monitor = CollapseMonitor(
        train_cfg["pretrain"]["collapse_window"], train_cfg["pretrain"]["collapse_std_threshold"]
    )
    explosion_monitor = ExplosionMonitor(train_cfg["pretrain"]["collapse_window"])
    history = {"loss": [], "target_std": [], "context_std": []}
    collapsed_at: int | None = None
    exploded_at: int | None = None

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        epoch_losses, epoch_stds = [], []

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            batch = {name: tensor[idx] for name, tensor in branch_tensors.items()}

            out = model(batch)
            optimizer.zero_grad()
            out["loss"].backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()
            model.update_target_encoder()

            epoch_losses.append(out["loss"].item())
            epoch_stds.append(out["target_std"])

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_std = sum(epoch_stds) / len(epoch_stds)
        history["loss"].append(mean_loss)
        history["target_std"].append(mean_std)

        is_collapsed = monitor.update(mean_std)
        if is_collapsed and collapsed_at is None:
            collapsed_at = epoch
            logger.warning(
                f"epoch {epoch}: possible representation collapse "
                f"(target_std trailing mean below {monitor.threshold})"
            )
        if explosion_monitor.update(mean_loss) and exploded_at is None:
            exploded_at = epoch
            logger.warning(f"epoch {epoch}: possible loss explosion (loss growing, not shrinking)")

        if epoch % train_cfg["pretrain"]["log_every"] == 0 or epoch == n_epochs - 1:
            logger.info(f"epoch {epoch}: loss={mean_loss:.4f} target_std={mean_std:.4f}")

    return {
        "model": model, "history": history, "collapsed_at": collapsed_at, "exploded_at": exploded_at,
    }


def pretrain_multi_corpus(
    corpora: list[dict],
    model_cfg: dict | None = None,
    train_cfg: dict | None = None,
    epochs: int | None = None,
) -> dict:
    """Pretrain across multiple corpora with different branch availability in
    the same model, e.g. the labeled corpus (3 real branches: genomic_traits,
    gtdb_distance, rrna16s) and the GEM MAG corpus (most rows: 2 real branches
    only, since real 16S extraction at 52k-genome scale is a multi-day job, see scripts/gem_slow_features.py).

    Each corpus dict: {"name": str, "tensors": {branch: tensor}, "branches": [names]}.
    `branches` must be a subset of model_cfg's configured branches and is used
    to build a masker restricted to only ever mask/target branches this
    corpus actually has real data for, a 2-branch corpus never has its
    (absent) third branch selected as context or target.

    One epoch = one pass over every corpus (each corpus contributes its own
    minibatches; the same model/optimizer/target-encoder are shared and
    updated across all of them).
    """
    model_cfg = model_cfg or load_config("model")
    train_cfg = train_cfg or load_config("train")
    set_seed(train_cfg["seed"])

    model = JEPA(model_cfg)
    # Both context_encoder AND predictor must be optimized, the predictor is
    # not covered by context_encoder.parameters(). Missing it here (an earlier
    # bug) meant the predictor stayed randomly initialized forever, which is
    # what actually caused the loss-explosion failure mode gradient clipping
    # alone didn't fix (see module docstring).
    trainable_params = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["pretrain"]["lr"],
        weight_decay=train_cfg["pretrain"]["weight_decay"],
    )
    batch_size = train_cfg["pretrain"]["batch_size"]
    n_epochs = epochs if epochs is not None else train_cfg["pretrain"]["epochs"]

    for c in corpora:
        c["masker"] = BranchMasker(c["branches"])
        c["n"] = next(iter(c["tensors"].values())).shape[0]
        logger.info(f"corpus '{c['name']}': n={c['n']} branches={c['branches']}")

    grad_clip_norm = train_cfg["pretrain"].get("grad_clip_norm", 1.0)
    monitor = CollapseMonitor(
        train_cfg["pretrain"]["collapse_window"], train_cfg["pretrain"]["collapse_std_threshold"]
    )
    explosion_monitor = ExplosionMonitor(train_cfg["pretrain"]["collapse_window"])
    history = {"loss": [], "target_std": [], "per_corpus_loss": {c["name"]: [] for c in corpora}}
    collapsed_at: int | None = None
    exploded_at: int | None = None

    for epoch in range(n_epochs):
        epoch_losses, epoch_stds = [], []
        per_corpus_epoch_loss = {c["name"]: [] for c in corpora}

        for c in corpora:
            perm = torch.randperm(c["n"])
            bs = min(batch_size, c["n"])
            for start in range(0, c["n"], bs):
                idx = perm[start : start + bs]
                batch = {name: tensor[idx] for name, tensor in c["tensors"].items()}
                masked_branch = c["masker"].sample_masked_branch()

                out = model(batch, masked_branch=masked_branch)
                optimizer.zero_grad()
                out["loss"].backward()
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                optimizer.step()
                model.update_target_encoder()

                epoch_losses.append(out["loss"].item())
                epoch_stds.append(out["target_std"])
                per_corpus_epoch_loss[c["name"]].append(out["loss"].item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_std = sum(epoch_stds) / len(epoch_stds)
        history["loss"].append(mean_loss)
        history["target_std"].append(mean_std)
        for name, losses in per_corpus_epoch_loss.items():
            history["per_corpus_loss"][name].append(sum(losses) / len(losses))

        is_collapsed = monitor.update(mean_std)
        if is_collapsed and collapsed_at is None:
            collapsed_at = epoch
            logger.warning(f"epoch {epoch}: possible representation collapse")
        if explosion_monitor.update(mean_loss) and exploded_at is None:
            exploded_at = epoch
            logger.warning(f"epoch {epoch}: possible loss explosion (loss growing, not shrinking)")

        if epoch % train_cfg["pretrain"]["log_every"] == 0 or epoch == n_epochs - 1:
            per_corpus_str = " ".join(f"{n}={v[-1]:.4f}" for n, v in history["per_corpus_loss"].items())
            logger.info(f"epoch {epoch}: loss={mean_loss:.4f} target_std={mean_std:.4f} [{per_corpus_str}]")

    return {
        "model": model, "history": history, "collapsed_at": collapsed_at, "exploded_at": exploded_at,
    }


def save_checkpoint(model: JEPA, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"saved checkpoint to {path}")


def load_checkpoint(model_cfg: dict, path: str | Path) -> JEPA:
    model = JEPA(model_cfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
