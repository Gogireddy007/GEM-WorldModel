"""GrowthBench: load the labeled genomes, make the official lineage-held-out splits, and score predictions.

    from gem_worldmodel import benchmark as gb
    data = gb.load()
    results = gb.evaluate(my_fit_predict)          # my_fit_predict(train, test) -> predicted log doubling times
    print(results.groupby("held_out")[["typical error", "Spearman"]].mean())

The splits are the ones of the paper (evaluate_paper_headline.py): for each held-out rank the taxa are shuffled with the seed
and divided into five folds with GroupKFold, so no species group, genus, family or order is on both sides. The target is
the natural log of the maximal doubling time in hours at the organism's own growth temperature; ``y_ref`` (at 20 C) and the
growth temperature are also provided. ``fit_predict`` receives the training and test rows as data frames with the inputs
and the taxonomy; it must not use the measured growth temperature unless the result is reported as such.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

LEVELS = ("species", "genus", "family", "order")
PREFIX = {"genus": "g__", "family": "f__", "order": "o__"}
RANKS = ("g__", "f__", "o__", "c__", "p__", "d__")
SEEDS = (0, 1, 2)
N_FOLDS = 5
SLOW_HOURS = 24.0


@dataclass
class Benchmark:
    table: pd.DataFrame   # one row per genome: species, species_group, gtdb_taxonomy, accession, targets
    X: np.ndarray         # genome inputs (deploy_features.load_training_table, full feature set)
    names: list           # input names


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else "none"


def load(processed=None) -> Benchmark:
    """The labeled genomes with gene features, as in the paper (labels after the audit)."""
    from gem_worldmodel.features import deploy_features
    from gem_worldmodel.utils.config import load_config, resolve_path

    processed = processed or resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    df = df.assign(y=np.log(df["doubling_time_hours"].astype(float)), y_ref=np.log(df["doubling_time_hours_ref"].astype(float)))
    return Benchmark(df, X, list(names))


def groups(table: pd.DataFrame, level: str) -> np.ndarray:
    """The unit held out at a level: the species group, or the GTDB genus, family or order."""
    if level == "species":
        return table["species_group"].to_numpy()
    return table["gtdb_taxonomy"].map(lambda t: token(t, PREFIX[level])).to_numpy()


def splits(table: pd.DataFrame, level: str, seed: int, n_folds: int = N_FOLDS):
    """Official folds: list of (train indices, test indices)."""
    g = groups(table, level)
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    shuffled = np.array([perm[v] for v in g])
    return list(GroupKFold(n_folds).split(np.zeros(len(g)), groups=shuffled))


def score(y, pred) -> dict:
    """Typical fold error, Spearman correlation and AUC for a doubling time above a day (log-hour scale)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    out = {"typical error": float(np.exp(np.median(np.abs(y - pred))))}
    varies = np.ptp(pred) > 0
    out["Spearman"] = float(spearmanr(y, pred).correlation) if varies else float("nan")
    slow = y > np.log(SLOW_HOURS)
    out["AUC slow"] = float(roc_auc_score(slow, pred)) if varies and 0 < slow.sum() < len(slow) else float("nan")
    return out


def relatives_median(train: pd.DataFrame, test: pd.DataFrame, target: str = "y") -> np.ndarray:
    """Required baseline: the median label of the training genomes at the closest rank shared with each test genome."""
    tr_tok = {r: train["gtdb_taxonomy"].map(lambda t, r=r: token(t, r)).to_numpy() for r in RANKS}
    y = train[target].to_numpy()
    out = np.full(len(test), np.median(y))
    for i, t in enumerate(test["gtdb_taxonomy"]):
        for r in RANKS:
            tok = token(t, r)
            if tok == "none":
                continue
            same = tr_tok[r] == tok
            if same.any():
                out[i] = np.median(y[same])
                break
    return out


def evaluate(fit_predict, data: Benchmark = None, levels=LEVELS, seeds=SEEDS, target: str = "y") -> pd.DataFrame:
    """Scores of ``fit_predict(train, test, X_train, X_test)`` at every level and seed, with the relatives' median beside it.

    ``fit_predict`` gets the training and test rows of the table and of the input matrix and returns predictions of the
    target for the test rows. Each held-out genome is predicted once per seed."""
    data = data or load()
    t, X = data.table, data.X
    y = t[target].to_numpy()
    rows = []
    for level in levels:
        for seed in seeds:
            pred, base = np.full(len(t), np.nan), np.full(len(t), np.nan)
            for tr, te in splits(t, level, seed):
                pred[te] = fit_predict(t.iloc[tr], t.iloc[te], X[tr], X[te])
                base[te] = relatives_median(t.iloc[tr], t.iloc[te], target)
            rows.append({"held_out": level, "seed": seed, "method": "submission", **score(y, pred)})
            rows.append({"held_out": level, "seed": seed, "method": "relatives' median", **score(y, base)})
    return pd.DataFrame(rows)
