import numpy as np

from gem_worldmodel.eval.confidence import confidence_flags, nearest_training_distance


def test_nearest_training_distance_finds_the_closest_point():
    train_x = np.array([[0.0, 0.0], [10.0, 10.0]])
    new_x = np.array([[0.1, 0.1], [9.9, 9.9]])
    dist = nearest_training_distance(new_x, train_x)
    assert dist[0] < 1.0
    assert dist[1] < 1.0


def test_confidence_flags_marks_a_genuinely_far_point():
    rng = np.random.default_rng(0)
    train_x = rng.normal(loc=0.0, scale=1.0, size=(50, 4))
    new_x = np.vstack([rng.normal(loc=0.0, scale=1.0, size=(1, 4)), np.full((1, 4), 50.0)])
    result = confidence_flags(new_x, train_x, percentile=90.0)
    assert not result["far_from_training"][0]
    assert result["far_from_training"][1]


def test_confidence_flags_threshold_comes_from_training_self_distance():
    rng = np.random.default_rng(1)
    train_x = rng.normal(size=(30, 3))
    new_x = rng.normal(size=(5, 3))
    result = confidence_flags(new_x, train_x, percentile=50.0)
    assert result["threshold"] > 0
