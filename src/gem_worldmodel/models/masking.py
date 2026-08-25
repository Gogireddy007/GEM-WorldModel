"""Branch masking: one branch withheld per training step, sampled uniformly
across the configured branches (genomic traits / GTDB-distance embedding /
16S baseline).

Masking is per training *step* (i.e. shared across the whole batch for that
step), matching the plan's "random branch withheld per training step", not
per-sample.
"""

import random

import torch


class BranchMasker:
    def __init__(self, branch_names: list[str]):
        if len(branch_names) < 2:
            raise ValueError("branch masking requires at least 2 branches")
        self.branch_names = list(branch_names)

    def sample_masked_branch(self, generator: random.Random | None = None) -> str:
        rng = generator or random
        return rng.choice(self.branch_names)

    def apply(
        self, batch: dict[str, torch.Tensor], masked_branch: str
    ) -> tuple[dict[str, torch.Tensor], str]:
        """Split a batch of {branch_name: (B, d_branch)} into the context (all
        branches except `masked_branch`) and return it alongside the masked
        branch name for the caller to fetch the target from the full batch.
        """
        if masked_branch not in self.branch_names:
            raise ValueError(f"unknown branch: {masked_branch}")
        context = {name: tensor for name, tensor in batch.items() if name != masked_branch}
        return context, masked_branch
