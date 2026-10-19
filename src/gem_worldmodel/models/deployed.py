"""The deployed regressor: the average of a histogram gradient boosting model (median loss) and a
random forest, with settings per target from configs/deployed_model.yaml.

Chosen on 2026-10-18 over the earlier gradient-boosted trees (200 trees, depth 3, median loss): with
nested tuning (scripts/accuracy/evaluate_tuning.py) the average lowered typical error and raised
ranking at every level of lineage novelty (scripts/accuracy/evaluate_learners.py has the untuned
comparison). Settings are chosen by scripts/accuracy/tune_deployed_model.py.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from gem_worldmodel.utils.config import load_config

HIST_GRID = {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400, 800], "max_leaf_nodes": [7, 15, 31],
             "min_samples_leaf": [5, 10, 20], "l2_regularization": [0.0, 1.0, 5.0], "max_features": [0.5, 1.0]}
FOREST_GRID = {"min_samples_leaf": [1, 3, 5, 10], "max_features": [0.2, 0.33, 0.5], "max_samples": [0.7, None]}


def make_member(kind, params, seed=0):
    if kind == "hist":
        return HistGradientBoostingRegressor(random_state=seed, loss="quantile", quantile=0.5, **params)
    return RandomForestRegressor(random_state=seed, n_estimators=300, n_jobs=8, **params)


class DeployedRegressor:
    """Average of the two members' predictions; `target` is a key of configs/deployed_model.yaml."""

    def __init__(self, target="own temperature", seed=0, settings=None):
        self.target = target
        self.seed = seed
        self.settings = settings if settings is not None else load_config("deployed_model")[target]

    def fit(self, X, y):
        self.members_ = [make_member(kind, self.settings[kind], self.seed).fit(X, y) for kind in ("hist", "forest")]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.members_], axis=0)
