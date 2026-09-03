"""Core JEPA checks: EMA update rule correctness, and confirmation that
gradients don't reach the target encoder.
"""

import torch

from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.utils.config import load_config


def _toy_batch(cfg, batch_size=4):
    return {
        b["name"]: torch.randn(batch_size, b["input_dim"]) for b in cfg["branches"]
    }


def test_target_encoder_params_require_no_grad():
    cfg = load_config("model")
    model = JEPA(cfg)
    for p in model.target_encoder.bank.parameters():
        assert p.requires_grad is False


def test_backward_does_not_populate_target_encoder_grad():
    cfg = load_config("model")
    model = JEPA(cfg)
    batch = _toy_batch(cfg)

    out = model(batch, masked_branch=cfg["branches"][0]["name"])
    out["loss"].backward()

    for p in model.target_encoder.bank.parameters():
        assert p.grad is None
    for p in model.context_encoder.parameters():
        pass  # context encoder should receive grads on at least some params
    assert any(p.grad is not None for p in model.context_encoder.parameters())


def test_ema_update_matches_formula():
    cfg = load_config("model")
    decay = cfg["target_encoder"]["ema_decay"]
    model = JEPA(cfg)

    xi_before = [p.detach().clone() for p in model.target_encoder.bank.parameters()]

    # Perturb theta (simulate an optimizer step) so theta != xi before the EMA update.
    with torch.no_grad():
        for p in model.context_encoder.bank.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    theta_after = [p.detach().clone() for p in model.context_encoder.bank.parameters()]

    model.update_target_encoder()

    for xi0, theta1, xi1 in zip(xi_before, theta_after, model.target_encoder.bank.parameters()):
        expected = decay * xi0 + (1 - decay) * theta1
        assert torch.allclose(xi1.detach(), expected, atol=1e-6)


def test_ema_update_does_not_move_theta():
    cfg = load_config("model")
    model = JEPA(cfg)
    theta_before = [p.detach().clone() for p in model.context_encoder.bank.parameters()]
    model.update_target_encoder()
    for before, after in zip(theta_before, model.context_encoder.bank.parameters()):
        assert torch.allclose(before, after.detach())


def test_target_encoder_stays_in_eval_mode_even_after_jepa_train():
    """Regression test for a real bug found on 2026-09-02: nothing in the
    codebase ever called .eval(), so the target encoder's dropout layers were
    active everywhere, including during pretraining itself, injecting random
    noise into the training TARGET. TargetEncoder.train() is now overridden
    to always force eval mode; this checks that override actually holds even
    when the parent JEPA module's own .train() is called, which recursively
    calls .train(True) on every submodule by default in plain PyTorch.
    """
    cfg = load_config("model")
    model = JEPA(cfg)
    assert model.target_encoder.bank.training is False  # eval from construction

    model.train()  # PyTorch's default: recursively sets training=True on ALL submodules
    assert model.context_encoder.training is True  # this one SHOULD flip
    assert model.target_encoder.bank.training is False  # this one must not


def test_joint_representation_is_deterministic():
    """Regression test for the same bug: with the encoder's real dropout
    (configs/model.yaml sets dropout > 0) left in training mode, two forward
    passes on the identical input gave different outputs, measured max
    absolute difference of 1.13 on a JEPA checkpoint before this was fixed.
    joint_representation now puts the context encoder in eval mode for the
    duration of the call (see utils/torch_utils.py:eval_mode).
    """
    cfg = load_config("model")
    model = JEPA(cfg)
    batch = _toy_batch(cfg, batch_size=8)

    out1 = model.joint_representation(batch)
    out2 = model.joint_representation(batch)
    assert torch.equal(out1, out2)


def test_joint_representation_restores_prior_training_mode():
    """eval_mode must restore the PRIOR mode after the call, not unconditionally
    flip back to train, so a call to joint_representation in the middle of an
    actual training loop doesn't accidentally leave the encoder in eval mode
    for the next training step.
    """
    cfg = load_config("model")
    model = JEPA(cfg)

    model.context_encoder.eval()
    model.joint_representation(_toy_batch(cfg))
    assert model.context_encoder.training is False  # was eval before, should still be eval after

    model.context_encoder.train()
    model.joint_representation(_toy_batch(cfg))
    assert model.context_encoder.training is True  # was train before, should still be train after
