"""Predictor P_phi: predicts the masked branch's target latent from the joint
context representation plus a learnable "which branch is masked" token.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, branch_names: list[str], latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.branch_names = list(branch_names)
        self.mask_tokens = nn.Embedding(len(branch_names), latent_dim)
        self._branch_index = {name: i for i, name in enumerate(branch_names)}

        layers: list[nn.Module] = []
        in_dim = latent_dim * 2
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, context_latent: torch.Tensor, masked_branch: str) -> torch.Tensor:
        idx = self._branch_index[masked_branch]
        token = self.mask_tokens(torch.tensor(idx, device=context_latent.device))
        token = token.unsqueeze(0).expand(context_latent.shape[0], -1)
        combined = torch.cat([context_latent, token], dim=-1)
        return self.mlp(combined)
