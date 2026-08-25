#!/usr/bin/env python
"""Week 3 sanity script: proves no-collapse behavior on a few dummy batches
before committing to the real Week 4 pretraining run. Complements
tests/test_ema.py (EMA correctness + stop-gradient) with a runnable, visible
check of encoder output variance across steps.
"""

import torch

from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.utils.config import load_config
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    cfg = load_config("model")
    model = JEPA(cfg)
    optimizer = torch.optim.AdamW(model.context_encoder.parameters(), lr=3e-4)

    for step in range(50):
        batch = {b["name"]: torch.randn(16, b["input_dim"]) for b in cfg["branches"]}
        out = model(batch)
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        model.update_target_encoder()

        if step % 10 == 0:
            logger.info(
                f"step {step}: loss={out['loss'].item():.4f} "
                f"target_std={out['target_std']:.4f} context_std={out['context_std']:.4f}"
            )

    assert out["target_std"] > 0.01, "collapse detected: target encoder output variance near zero"
    assert not any(p.grad is not None for p in model.target_encoder.bank.parameters()), (
        "gradients leaked into the target encoder"
    )
    logger.info("Week 3 sanity check passed: no collapse, no gradient leakage into target encoder")


if __name__ == "__main__":
    main()
