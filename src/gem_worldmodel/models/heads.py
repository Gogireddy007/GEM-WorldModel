"""Growth-rate prediction head, fine-tuned on top of the pretrained context
encoder's joint latent.
"""

import torch
import torch.nn as nn


class GrowthRateHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(max(num_layers - 1, 0)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, joint_latent: torch.Tensor) -> torch.Tensor:
        return self.mlp(joint_latent).squeeze(-1)
