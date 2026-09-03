"""Small torch helpers shared across the codebase."""

from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(*modules: nn.Module):
    """Temporarily switch `modules` to eval mode (disabling dropout/batchnorm
    training-time behavior) for the duration of the block, then restore each
    module's PRIOR mode, not unconditionally back to train. `torch.no_grad()`
    disables gradient tracking only, it does NOT disable dropout, that
    requires `.eval()` specifically. Use both together for a genuine
    inference-only forward pass:

        with torch.no_grad(), eval_mode(jepa.context_encoder):
            z = jepa.context_encoder(batch)

    Found and fixed on 2026-09-02 after a direct repro showed two forward
    passes on identical input giving different outputs, every module with a
    real nn.Dropout layer (models/encoders.py's branch encoder MLPs) had been
    running in the PyTorch default training=True mode everywhere in the
    codebase, since nothing ever called .eval(). See research_log.md for the
    full story and models/encoders.py:TargetEncoder for the related fix
    (the target encoder is now always in eval mode, permanently, since it's
    never gradient-trained and dropout there only injects noise into the
    training target).
    """
    was_training = [m.training for m in modules]
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, training in zip(modules, was_training):
            m.train(training)
