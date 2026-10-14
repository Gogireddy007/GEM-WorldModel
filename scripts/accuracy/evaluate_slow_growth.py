#!/usr/bin/env python
"""This tests ways to stop the model from squeezing very slow growth toward the
middle, on the doubling time at the organism's own temperature, with whole
species or whole orders held out.

Candidates:
  median          the current model (median loss)
  mean            the same trees with squared-error loss
  spread-matched  the current model, then a monotone map that makes the spread
                  of the predictions match the spread of measured values,
                  learned from out-of-fold predictions inside the training data
                  only (quantile mapping). Ranking is unchanged by design
  weighted        the current model trained with rare doubling times weighted
                  up (inverse density of log doubling time, capped)
  slow classifier a separate model for the probability of a doubling time above
                  24 hours, compared with ranking by the current prediction

For each: typical error, Spearman, log R2, the slope of prediction on
measurement (1 means no squeezing), the share predicted above 24 h against the
share measured, how many of the truly slow species are called slow, and how
many faster ones are wrongly called slow.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7]
SLOW = np.log(24.0)
PREFIX = {"order": "o__"}


def token(t, p):
    m = re.search(rf"{p}[^;]*", str(t))
    return m.group(0) if m and m.group(0) != p else "none"


def gbr(seed, loss="quantile"):
    kw = {"loss": "quantile", "alpha": 0.5} if loss == "quantile" else {}
    return GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **kw)


def folds(groups, seed, n=5):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    return list(GroupKFold(n).split(groups, groups=np.array([perm[v] for v in groups])))


def inner_oof(X, y, groups, seed):
    pred = np.full(len(y), np.nan)
    for tr, te in folds(groups, seed + 100):
        pred[te] = gbr(seed).fit(X[tr], y[tr]).predict(X[te])
    return pred


def quantile_map(train_pred, train_y, new_pred):
    """Monotone map sending the training predictions' distribution onto the training targets'."""
    qs = np.linspace(0, 1, 101)
    src, dst = np.quantile(train_pred, qs), np.quantile(train_y, qs)
    src = np.maximum.accumulate(src + np.arange(len(src)) * 1e-9)
    return np.interp(new_pred, src, dst, left=dst[0] + (new_pred.min() - src[0]), right=dst[-1] + (new_pred.max() - src[-1]))


def density_weights(y):
    d = gaussian_kde(y)(y)
    w = 1.0 / d
    return np.minimum(w / np.median(w), 5.0)


def metrics(y, p):
    slow, called = y > SLOW, p > SLOW
    return {
        "typical error": np.exp(np.median(np.abs(y - p))),
        "Spearman": spearmanr(y, p).correlation,
        "R2 log": r2_score(y, p),
        "slope": np.polyfit(y, p, 1)[0],
        "share >24h predicted": 100 * called.mean(),
        "slow found": 100 * called[slow].mean(),
        "fast called slow": 100 * called[~slow].mean(),
        "error, slow species": np.exp(np.median(np.abs(y[slow] - p[slow]))),
    }


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, X, _ = deploy_features.load_training_table(processed, load_config("features")["rrna16s"]["kmer_k"], with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    y = np.log(df["doubling_time_hours"].to_numpy(float))
    sp = df["species_group"].to_numpy()
    print(f"{len(y)} rows, {100 * (y > SLOW).mean():.1f}% measured above 24 h ({(y > SLOW).sum()} rows)\n")

    for level in ("species", "order"):
        groups = sp if level == "species" else df["gtdb_taxonomy"].map(lambda t: token(t, "o__")).to_numpy()
        runs = {k: [] for k in ("median", "mean", "spread-matched", "weighted")}
        cls = {"slow classifier": [], "rank by current prediction": []}
        for seed in SEEDS:
            pred = {k: np.full(len(y), np.nan) for k in runs}
            prob = np.full(len(y), np.nan)
            for tr, te in folds(groups, seed):
                pred["median"][te] = gbr(seed).fit(X[tr], y[tr]).predict(X[te])
                pred["mean"][te] = gbr(seed, "squared").fit(X[tr], y[tr]).predict(X[te])
                inner = inner_oof(X[tr], y[tr], sp[tr] if level == "species" else groups[tr], seed)
                pred["spread-matched"][te] = quantile_map(inner, y[tr], pred["median"][te])
                pred["weighted"][te] = gbr(seed).fit(X[tr], y[tr], sample_weight=density_weights(y[tr])).predict(X[te])
                prob[te] = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3).fit(X[tr], y[tr] > SLOW).predict_proba(X[te])[:, 1]
            for k in runs:
                runs[k].append(metrics(y, pred[k]))
            slow = y > SLOW
            cls["slow classifier"].append({"AUC": roc_auc_score(slow, prob), "Brier": brier_score_loss(slow, prob), "mean prob, slow": prob[slow].mean(), "mean prob, others": prob[~slow].mean()})
            cls["rank by current prediction"].append({"AUC": roc_auc_score(slow, pred["median"]), "Brier": np.nan, "mean prob, slow": np.nan, "mean prob, others": np.nan})
        print(f"===== {level.upper()} HELD OUT, mean of {len(SEEDS)} seeds")
        table = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in runs.items()}).T
        print(table.round(3).to_string())
        print()
        print(pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in cls.items()}).T.round(3).to_string())
        print()


if __name__ == "__main__":
    main()
