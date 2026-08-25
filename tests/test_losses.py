import torch

from gem_worldmodel.models.losses import embedding_std, latent_prediction_loss


def test_latent_prediction_loss_is_zero_for_identical_inputs():
    s = torch.randn(8, 16)
    loss = latent_prediction_loss(s.clone(), s.clone())
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_latent_prediction_loss_detaches_target():
    s_hat = torch.randn(8, 16, requires_grad=True)
    s = torch.randn(8, 16, requires_grad=True)
    loss = latent_prediction_loss(s_hat, s)
    loss.backward()
    assert s_hat.grad is not None
    assert s.grad is None  # target must be detached, no gradient flows into it


def test_embedding_std_near_zero_for_constant_vector():
    latents = torch.ones(32, 8) * 3.0
    assert embedding_std(latents) < 1e-6


def test_embedding_std_positive_for_varied_vectors():
    latents = torch.randn(32, 8)
    assert embedding_std(latents) > 0.01
