#!/usr/bin/env python
"""This picks the settings of the deployed model (the average of a histogram gradient boosting
model with median loss and a random forest, see gem_worldmodel.models.deployed) for each target,
with the search evaluated in evaluate_tuning.py: random settings scored by a three-fold split
grouped by GTDB family, here on all labeled rows and averaged over three different splits so the
choice is less noisy. The nested evaluation (evaluate_tuning.py) is the unbiased estimate of what
this procedure gains; accuracy tables computed with these fixed settings are slightly optimistic.

Writes configs/deployed_model.yaml.
"""

import re

import numpy as np
import yaml
from sklearn.model_selection import GroupKFold, ParameterSampler

from gem_worldmodel.features import deploy_features
from gem_worldmodel.models import deployed
from gem_worldmodel.utils.config import load_config, resolve_path

N_CANDIDATES = 20
SPLITS = [0, 1, 2]
TARGETS = {"own temperature": "doubling_time_hours", "20 C": "doubling_time_hours_ref", "growth temperature": "growth_temp_c"}


def family_of(t):
    m = re.search(r"f__[^;]*", str(t))
    return m.group(0) if m else "none"


def score(kind, params, X, y, family):
    errs = []
    for s in SPLITS:
        rng = np.random.default_rng(500 + s)
        ug = np.unique(family)
        perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(3).split(X, y, np.array([perm[v] for v in family])):
            pred[te] = deployed.make_member(kind, params, seed=0).fit(X[tr], y[tr]).predict(X[te])
        errs.append(np.median(np.abs(y - pred)))
    return float(np.mean(errs))


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    df, X = df[use].reset_index(drop=True), X[use]
    family = df["gtdb_taxonomy"].map(family_of).to_numpy()
    config = {}
    for tname, col in TARGETS.items():
        ok = df[col].notna().to_numpy()
        y = df.loc[ok, col].to_numpy(float)
        y = y if tname == "growth temperature" else np.log(y)
        config[tname] = {}
        for kind, grid in (("hist", deployed.HIST_GRID), ("forest", deployed.FOREST_GRID)):
            best, best_err = None, np.inf
            for params in ParameterSampler(grid, N_CANDIDATES, random_state=0):
                err = score(kind, params, X[ok], y, family[ok])
                if err < best_err:
                    best, best_err = params, err
            config[tname][kind] = {key: (float(v) if isinstance(v, float) else v) for key, v in best.items()}
            print(f"{tname:20s} {kind:7s} inner error {best_err:.4f}  {best}", flush=True)
    path = resolve_path("configs/deployed_model.yaml")
    header = "# Settings of the deployed model, chosen by scripts/accuracy/tune_deployed_model.py. Do not edit by hand.\n"
    path.write_text(header + yaml.safe_dump(config, sort_keys=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
