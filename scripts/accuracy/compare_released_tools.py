#!/usr/bin/env python
"""This compares the deployed model with the released gRodon and Phydon R packages.

  --prepare  writes released_tools_folds.csv: for each seed, the test fold of every labeled genome,
             with whole species groups held out exactly as for our own model (5 folds, groups
             shuffled with the seed). scripts/r/run_released_phydon.R then runs Phydon fold by fold.
  (default)  compares the three on the genomes all of them predicted, on identical folds.

Fairness. Our model is refit on the training folds only. Phydon's phylogenetic part leaves out every
genome of the fold being predicted (the whole fold is submitted at once), but it is otherwise the
released fit: its training set (Madin et al.'s growth rates on the GTDB r220 tree) contains species
that are not in ours, which gives it more relatives, and its weighting model and gRodon's regression
were fitted on data that include our species. Both points favor the released tools. The target is
the doubling time at the organism's own growth temperature, and both released tools get that
temperature as input. Only species held out is compared: Phydon's built-in training set cannot be
made to leave out whole orders without editing it.

Asymmetry checked as well: the released tools get the measured growth temperature, our model does
not (it predicts it). So our model is also scored with the measured growth temperature as an input,
and released gRodon is also scored without the temperature. (Before 2026-10-18 the script also
scored our model plus released gRodon's codon statistics; those statistics are now part of the
deployed features, see evaluate_grodon_and_temperature.py.) Missing values in the extra
inputs are filled with the median, as for the other inputs.

Writes released_tools_comparison.csv.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [0, 1, 2]
SLOW = 24.0


def folds_for(groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    fold = np.full(len(groups), -1)
    for i, (_, te) in enumerate(GroupKFold(5).split(groups, groups=np.array([perm[v] for v in groups]))):
        fold[te] = i
    return fold


def load(processed):
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True)
    use = df["has_cub"].to_numpy()
    return df[use].reset_index(drop=True), X[use]


def scores(y, p):
    ly, lp = np.log(y), np.log(p)
    return {"typical error": float(np.exp(np.median(np.abs(ly - lp)))), "Spearman": spearmanr(ly, lp).correlation,
            "AUC slow": roc_auc_score(y > SLOW, lp) if 0 < (y > SLOW).sum() < len(y) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, X = load(processed)
    groups = df["species_group"].to_numpy()

    if args.prepare:
        inp = pd.read_csv(processed / "released_tools_input.csv").drop_duplicates("accession").set_index("accession")
        out = df[["species", "species_group", "accession", "growth_temp_c", "doubling_time_hours", "archaeon"]].copy()
        for s in SEEDS:
            out[f"fold_seed{s}"] = folds_for(groups, s)
        out["gene_location"] = out["accession"].map(inp["gene_location"])
        out["genome_name"] = out["accession"].map(inp["genome_name"])
        out["domain"] = np.where(out["archaeon"] == 1, "archaea", "bacteria")
        out = out[out["growth_temp_c"].notna() & out["gene_location"].notna()].drop_duplicates("accession")
        out.to_csv(processed / "released_tools_folds.csv", index=False)
        print(f"wrote released_tools_folds.csv: {len(out)} genomes with genes and a growth temperature "
              f"({(out['domain'] == 'archaea').sum()} archaeal), seeds {SEEDS}")
        return

    rel = pd.read_csv(processed / "released_tools_predictions.csv")
    y_all = np.log(df["doubling_time_hours"].to_numpy(float))
    names = df["accession"].map(lambda a: ("RS_" if str(a).startswith("GCF_") else "GB_") + str(a))
    grodon = pd.read_csv(processed / "released_grodon.csv").drop_duplicates("genome_name").set_index("genome_name")

    def filled(values):
        values = np.asarray(values, float)
        return np.where(np.isnan(values), np.nanmedian(values), values)[:, None]

    temp_col = filled(df["growth_temp_c"])
    # gRodon's statistics are part of the deployed features since 2026-10-18, so the only variant
    # adds the measured temperature
    variants = {"this model": X, "this model + measured temperature": np.hstack([X, temp_col])}
    rows = []
    for s in sorted(rel["seed"].unique()):
        fold = folds_for(groups, s)
        preds = {}
        for vname, Xv in variants.items():
            p = np.full(len(y_all), np.nan)
            for f in range(5):
                tr, te = fold != f, fold == f
                m = DeployedRegressor("own temperature", s)
                p[te] = m.fit(Xv[tr], y_all[tr]).predict(Xv[te])
            preds[vname] = np.exp(p)
        mine = pd.DataFrame({"genome_name": names, **preds, "measured": df["doubling_time_hours"].to_numpy(float), "fold_ours": fold})
        mine["gRodon without temperature"] = names.map(grodon["d_without_temperature"]).to_numpy()
        r = rel[rel["seed"] == s].rename(columns={"genome": "genome_name"})
        merged = r.merge(mine.drop_duplicates("genome_name"), on="genome_name")
        assert (merged["fold"] == merged["fold_ours"]).all(), "fold assignment differs between R and Python"
        for subset, sel in (("all genomes", np.ones(len(merged), bool)),
                            ("Phydon found the species on its tree", merged["phylopred"].notna().to_numpy())):
            m = merged[sel & merged["gRodonpred"].notna().to_numpy()]
            methods = [(v, v) for v in variants] + [("gRodon (released)", "gRodonpred"), ("gRodon (released), no temperature", "gRodon without temperature"),
                                                     ("Phydon (released)", "combopred")]
            for method, col in methods:
                mm = m[m[col].notna() & (m[col] > 0)]
                if len(mm) >= 10:
                    rows.append({"seed": s, "subset": subset, "method": method, "n": len(mm), **scores(mm["measured"].to_numpy(), mm[col].to_numpy())})
    out = pd.DataFrame(rows)
    out.to_csv(processed / "released_tools_comparison.csv", index=False)
    for subset in out["subset"].unique():
        d = out[out["subset"] == subset]
        print(f"\n===== {subset} (species held out, mean of {d['seed'].nunique()} seeds)")
        print(d.groupby("method")[["n", "typical error", "Spearman", "AUC slow"]].mean().round(3).to_string())
        piv = d.pivot_table(index="seed", columns="method", values="typical error")
        for mine_name in variants:
            for other in ("gRodon (released)", "Phydon (released)"):
                if other in piv and mine_name in piv:
                    diff = piv[mine_name] - piv[other]
                    print(f"  {mine_name} vs {other}: typical error lower in {int((diff < 0).sum())}/{len(diff)} seeds")


if __name__ == "__main__":
    main()
