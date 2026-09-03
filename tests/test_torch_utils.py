import torch.nn as nn

from gem_worldmodel.utils.torch_utils import eval_mode


def test_eval_mode_switches_to_eval_inside_the_block():
    m = nn.Linear(2, 2)
    m.train()
    with eval_mode(m):
        assert m.training is False


def test_eval_mode_restores_train_mode_after():
    m = nn.Linear(2, 2)
    m.train()
    with eval_mode(m):
        pass
    assert m.training is True


def test_eval_mode_restores_eval_mode_after_if_it_was_already_eval():
    m = nn.Linear(2, 2)
    m.eval()
    with eval_mode(m):
        assert m.training is False
    assert m.training is False


def test_eval_mode_handles_multiple_modules_with_different_prior_states():
    a = nn.Linear(2, 2)
    b = nn.Linear(2, 2)
    a.train()
    b.eval()

    with eval_mode(a, b):
        assert a.training is False
        assert b.training is False

    assert a.training is True
    assert b.training is False


def test_eval_mode_restores_even_if_the_block_raises():
    m = nn.Linear(2, 2)
    m.train()
    try:
        with eval_mode(m):
            raise ValueError("boom")
    except ValueError:
        pass
    assert m.training is True
