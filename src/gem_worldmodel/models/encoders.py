"""Context encoder E_theta and Target encoder E_xi (EMA, stop-gradient).

Each branch (genomic traits / GTDB-distance / 16S baseline) gets its own
per-branch MLP encoder into a shared latent space, since branches have
different input dimensionalities and semantics. E_theta pools the *context*
branches (whichever aren't masked this step) into a joint context vector.
E_xi is a structurally-identical, EMA-updated copy used only to encode the
masked branch's own vector into the target latent `s`, it never receives
gradients from the training loss.
"""

import copy

import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(max(num_layers - 1, 0)):
        layers += [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class BranchEncoderBank(nn.Module):
    """One MLP per branch, mapping that branch's raw feature vector to latent_dim."""

    def __init__(self, branches: list[dict], hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.branch_names = [b["name"] for b in branches]
        self.encoders = nn.ModuleDict(
            {
                b["name"]: _mlp(b["input_dim"], hidden_dim, latent_dim, num_layers, dropout)
                for b in branches
            }
        )
        self.latent_dim = latent_dim

    def forward(self, branch_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.encoders[branch_name](x)

    def encode_all(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.encoders[name](x) for name, x in batch.items() if name in self.encoders}


class ContextEncoder(nn.Module):
    """E_theta: encodes whichever branches are present (the unmasked context)
    and mean-pools them into a single joint context vector.
    """

    def __init__(self, branches: list[dict], hidden_dim: int, latent_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.bank = BranchEncoderBank(branches, hidden_dim, latent_dim, num_layers, dropout)
        self.latent_dim = latent_dim

    def forward(self, context_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        per_branch = self.bank.encode_all(context_batch)
        if not per_branch:
            raise ValueError("context encoder received no branches, nothing left after masking")
        stacked = torch.stack(list(per_branch.values()), dim=0)  # (n_branches, B, latent_dim)
        return stacked.mean(dim=0)  # (B, latent_dim)


class TargetEncoder(nn.Module):
    """E_xi: EMA copy of a ContextEncoder's per-branch encoder bank, used only
    to produce the target latent `s` for the masked branch. Never trained by
    backprop, only ever updated via `ema_update`.
    """

    def __init__(self, context_encoder: ContextEncoder, ema_decay: float):
        super().__init__()
        self.bank = copy.deepcopy(context_encoder.bank)
        for p in self.bank.parameters():
            p.requires_grad_(False)
        self.bank.eval()
        self.ema_decay = ema_decay
        self.latent_dim = context_encoder.latent_dim

    def train(self, mode: bool = True) -> "TargetEncoder":
        """Always stays in eval mode regardless of what's requested. This
        encoder is never trained by gradient descent, only EMA-updated, so
        its dropout layers should never be active: they'd just inject random
        noise into the training TARGET `s`, degrading the JEPA objective for
        no benefit, dropout's regularization value only applies to something
        that's actually being fit by gradient descent. Found by direct repro
        on 2026-09-02: two forward passes on identical input gave a max
        absolute difference of 1.13 (latents were meant to be L2-normalized
        to unit scale), because nothing in this codebase ever called
        `.eval()` and a freshly constructed/loaded nn.Module defaults to
        `.training = True`. This bug affected every reported number in the
        project up to that point, see research_log.md.
        """
        return super().train(False)

    @torch.no_grad()
    def forward(self, branch_name: str, x: torch.Tensor) -> torch.Tensor:
        return self.bank(branch_name, x)

    @torch.no_grad()
    def ema_update(self, context_encoder: ContextEncoder) -> None:
        """xi <- ema_decay * xi + (1 - ema_decay) * theta, per-parameter."""
        for p_xi, p_theta in zip(self.bank.parameters(), context_encoder.bank.parameters()):
            p_xi.mul_(self.ema_decay).add_(p_theta.detach(), alpha=1.0 - self.ema_decay)
