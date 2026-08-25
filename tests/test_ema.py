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
