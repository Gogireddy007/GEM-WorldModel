"""How far does a new genome sit from anything the model was trained on?

A simple nearest-neighbor distance in the same standardized raw feature
space the deployed model (raw features, gradient-boosted trees) actually
uses. Not a probability, not a calibrated confidence, just a plain distance
so a prediction can be flagged when nothing resembling it was in the
training set.

The threshold for "far" is set from the training corpus's own internal
spread, not a number picked by eye: for every training species, its
distance to its own nearest OTHER training species is computed, and a new
genome is flagged whenever it sits further from the training set than most
training species sit from each other.
"""

import numpy as np


def nearest_training_distance(new_x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of new_x to its nearest row in train_x."""
    dists = np.sqrt(((new_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=-1))
    return dists.min(axis=1)


def training_self_distance(train_x: np.ndarray) -> np.ndarray:
    """Same thing, but each training species against every OTHER training
    species (excluding itself), used to set a self-consistent threshold.
    """
    dists = np.sqrt(((train_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


def confidence_flags(new_x: np.ndarray, train_x: np.ndarray, percentile: float = 90.0) -> dict:
    """Returns the distance for each new genome, the threshold it's judged
    against, and a boolean flag for whether it's farther from the training
    set than `percentile` percent of training species are from each other.
    """
    self_dist = training_self_distance(train_x)
    threshold = np.percentile(self_dist, percentile)
    new_dist = nearest_training_distance(new_x, train_x)
    return {
        "distance": new_dist,
        "threshold": threshold,
        "far_from_training": new_dist > threshold,
    }
