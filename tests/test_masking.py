import random

import torch

from gem_worldmodel.models.masking import BranchMasker


def test_apply_excludes_masked_branch():
    masker = BranchMasker(["a", "b", "c"])
    batch = {name: torch.randn(4, 3) for name in ["a", "b", "c"]}
    context, masked = masker.apply(batch, "b")
    assert masked == "b"
    assert set(context.keys()) == {"a", "c"}


def test_sample_masked_branch_covers_all_branches():
    masker = BranchMasker(["a", "b", "c"])
    rng = random.Random(0)
    seen = {masker.sample_masked_branch(rng) for _ in range(200)}
    assert seen == {"a", "b", "c"}


def test_requires_at_least_two_branches():
    try:
        BranchMasker(["only_one"])
        assert False, "expected ValueError"
    except ValueError:
        pass
