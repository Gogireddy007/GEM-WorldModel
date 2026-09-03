"""Top-level JEPA module wiring branch masking + context/target encoders +
predictor + latent loss into one forward pass, matching the architecture
diagram's "Self-supervised world model (JEPA)" box.
"""

import random

import torch
import torch.nn as nn

from gem_worldmodel.models.encoders import ContextEncoder, TargetEncoder
from gem_worldmodel.models.losses import embedding_std, latent_prediction_loss
from gem_worldmodel.models.masking import BranchMasker
from gem_worldmodel.models.predictor import Predictor
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.torch_utils import eval_mode


class JEPA(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or load_config("model")
        self.branches = cfg["branches"]
        branch_names = [b["name"] for b in self.branches]

        enc = cfg["encoder"]
        self.context_encoder = ContextEncoder(
            self.branches, enc["hidden_dim"], enc["latent_dim"], enc["num_layers"], enc["dropout"]
        )
        self.target_encoder = TargetEncoder(self.context_encoder, cfg["target_encoder"]["ema_decay"])

        pred = cfg["predictor"]
        self.predictor = Predictor(branch_names, enc["latent_dim"], pred["hidden_dim"], pred["num_layers"])

        self.masker = BranchMasker(branch_names)
        self.latent_dim = enc["latent_dim"]

    def forward(
        self, batch: dict[str, torch.Tensor], masked_branch: str | None = None, rng: random.Random | None = None
    ) -> dict:
        if masked_branch is None:
            masked_branch = self.masker.sample_masked_branch(rng)
        context_batch, masked_branch = self.masker.apply(batch, masked_branch)

        z_context = self.context_encoder(context_batch)
        s_hat = self.predictor(z_context, masked_branch)
        s = self.target_encoder(masked_branch, batch[masked_branch])

        loss = latent_prediction_loss(s_hat, s)
        return {
            "loss": loss,
            "s_hat": s_hat,
            "s": s,
            "z_context": z_context,
            "masked_branch": masked_branch,
            "target_std": embedding_std(s),
            "context_std": embedding_std(z_context),
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        self.target_encoder.ema_update(self.context_encoder)

    @torch.no_grad()
    def joint_representation(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Joint representation over ALL branches (no masking), used downstream
        for the growth-rate head, probing, and intervention experiments.

        Runs the context encoder in eval mode (dropout off) since this is a
        pure inference call, restoring whatever mode it was in before. Left
        this out originally and it meant identical inputs gave different
        outputs across calls, see utils/torch_utils.py:eval_mode.
        """
        with eval_mode(self.context_encoder):
            return self.context_encoder(batch)
