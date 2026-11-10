#!/usr/bin/env python
"""This compares the deployed model with the released gRodon and Phydon R packages.

  --prepare  writes released_tools_folds.csv: for each seed, the test fold of every labeled genome,
             with whole species groups held out exactly as for our own model (5 folds, groups
             shuffled with the seed). scripts/r/run_released_phydon.R then runs Phydon fold by fold.
  (default)  compares the three on the genomes all of them predicted, on identical folds.
  --nested   the same, with our model built inside every fold exactly as it is deployed: settings
             chosen by the deployed search on the training rows only, then trained once per placement
             rank (evaluate_paper_headline.py). Writes released_tools_comparison_nested.csv.

Fairness. Our model is refit on the training folds only. Phydon's phylogenetic part leaves out every
genome of the fold being predicted (the whole fold is submitted at once), but it is otherwise the
released fit: its training set (Madin et al.'s growth rates on the GTDB r220 tree) contains species
that are not in ours, which gives it more relatives, and its weighting model and gRodon's regression
were fitted on data that include our species. Both points favor the released tools. The target is
the doubling time at the organism's own growth temperature, and both released tools get that
temperature as input. With --level order, whole GTDB orders are held out instead of species groups: the
R script then removes from Phydon's training table every species whose GTDB order (looked up for its
representative genome, or for its genus when the genome is not in GTDB R232) is held out in the fold;
Phydon's code is not changed. Output files carry the suffix _order.

Asymmetry checked as well: the released tools get the measured growth temperature, our model does
not (it predicts it). So our model is also scored with the measured growth temperature as an input,
and released gRodon is also scored without the temperature. (Before 2026-10-18 the script also
scored our model plus released gRodon's codon statistics; those statistics are now part of the
deployed features, see evaluate_grodon_and_temperature.py.) Missing values in the extra
inputs are filled with the median, as for the other inputs.

Writes released_tools_comparison.csv.
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from gem_worldmodel.features import deploy_features
from gem_worldmodel.models.deployed import DeployedRegressor
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [0, 1, 2]
SLOW = 24.0
NESTED = {}


def init_nested(state):
    threadpool_limits(1)  # one thread per worker
    NESTED.update(state)


def nested_fold(job):
    """Settings chosen on the training rows, then trained on the four placement copies; predictions for the test rows."""
    import tune_deployed_model as tuning
    vname, s, f = job
    Xv, copies, y, family, fold = (NESTED[k] for k in ("X", "copies", "y", "family", "fold"))
    Xv, copies = Xv[vname], copies[vname]
    tr, te = np.where(fold[s] != f)[0], np.where(fold[s] == f)[0]
    settings, _ = tuning.choose_settings(Xv[tr], y[tr], family[tr], n_jobs=1)
    m = DeployedRegressor("own temperature", s, settings=settings, n_jobs=1)
    m.fit(np.vstack([c[tr] for c in copies]), np.tile(y[tr], len(copies)))
    return vname, s, te, m.predict(Xv[te])


def folds_for(groups, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug)), strict=True))
    fold = np.full(len(groups), -1)
    for i, (_, te) in enumerate(GroupKFold(5).split(groups, groups=np.array([perm[v] for v in groups]))):
        fold[te] = i
    return fold


def write_phydon_rep_orders(processed):
    """GTDB R232 order of every species in Phydon's own training tables (by representative genome, else by genus)."""
    import subprocess
    raw = resolve_path("data/raw")
    tax = pd.concat([pd.read_csv(raw / f, sep="\t", header=None, names=["acc", "tax"]) for f in ("bac120_taxonomy.tsv", "ar53_taxonomy.tsv")])
    tax = tax.drop_duplicates("acc").set_index("acc")["tax"]
    order_of = tax.str.extract(r"(o__[^;]+)")[0]
    genus_order = pd.Series(order_of.values, index=tax.str.extract(r"g__([^;]+)")[0].values)
    genus_order = genus_order[~genus_order.index.duplicated()]
    tmp = processed / "phydon_training_species.csv"
    subprocess.run(["Rscript", "-e", '.libPaths(c(normalizePath(".Rlib"), .libPaths())); suppressPackageStartupMessages(library(Phydon)); '
                    'ns <- asNamespace("Phydon"); t <- rbind(get("GTDB_tax_trait_repGenome_in_tree_expanded", envir = ns)[, c("species", "Representative.genome")], '
                    'get("GTDB_tax_trait_repGenome_in_tree_expanded_archaea", envir = ns)[, c("species", "Representative.genome")]); '
                    f'write.csv(unique(t), "{tmp}", row.names = FALSE)'], check=True, cwd=resolve_path("."))
    t = pd.read_csv(tmp)
    t["order"] = t["Representative.genome"].map(order_of)
    by_genus = t["species"].str.split().str[0].map(genus_order)
    t["order"] = t["order"].fillna(by_genus)
    t.to_csv(processed / "phydon_rep_orders.csv", index=False)
    print(f"Phydon training species with a GTDB order: {t['order'].notna().mean():.1%} of {len(t)}")


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
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--level", choices=["species", "order"], default="species")
    args = ap.parse_args()
    sfx = "" if args.level == "species" else "_order"
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    df, X = load(processed)
    if args.level == "species":
        groups = df["species_group"].to_numpy()
    else:
        groups = df["gtdb_taxonomy"].str.extract(r"(o__[^;]+)")[0].fillna("o__none").to_numpy()

    if args.prepare:
        inp = pd.read_csv(processed / "released_tools_input.csv").drop_duplicates("accession").set_index("accession")
        out = df[["species", "species_group", "accession", "growth_temp_c", "doubling_time_hours", "archaeon"]].copy()
        for s in SEEDS:
            out[f"fold_seed{s}"] = folds_for(groups, s)
        out["gene_location"] = out["accession"].map(inp["gene_location"])
        out["genome_name"] = out["accession"].map(inp["genome_name"])
        out["domain"] = np.where(out["archaeon"] == 1, "archaea", "bacteria")
        out = out[out["growth_temp_c"].notna() & out["gene_location"].notna()].drop_duplicates("accession")
        if args.level == "order":
            out["order"] = df.drop_duplicates("accession").set_index("accession").loc[out["accession"], "gtdb_taxonomy"].str.extract(r"(o__[^;]+)")[0].to_numpy()
            write_phydon_rep_orders(processed)
        out.to_csv(processed / f"released_tools_folds{sfx}.csv", index=False)
        print(f"wrote released_tools_folds{sfx}.csv: {len(out)} genomes with genes and a growth temperature "
              f"({(out['domain'] == 'archaea').sum()} archaeal), seeds {SEEDS}")
        return

    rel = pd.read_csv(processed / f"released_tools_predictions{sfx}.csv")
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
    nested = {}
    if args.nested:
        import evaluate_paper_headline as headline
        import tune_deployed_model as tuning
        _, df_c, X_c, copies = headline.load()
        assert len(df_c) == len(df) and np.allclose(X_c, X), "row order differs from the headline loader"
        copies = {"this model": copies, "this model + measured temperature": [np.hstack([c, temp_col]) for c in copies]}
        family = df["gtdb_taxonomy"].map(tuning.family_of).to_numpy()
        fold = {s: folds_for(groups, s) for s in sorted(rel["seed"].unique())}
        state = {"X": variants, "copies": copies, "y": y_all, "family": family, "fold": fold}
        jobs = [(v, s, f) for v in variants for s in fold for f in range(5)]
        with Pool(args.workers, initializer=init_nested, initargs=(state,)) as pool:
            for i, (vname, s, te, p) in enumerate(pool.imap_unordered(nested_fold, jobs), 1):
                nested.setdefault((vname, s), np.full(len(y_all), np.nan))[te] = p
                print(f"{i}/{len(jobs)} nested folds done", flush=True)
    rows, per_genome = [], []
    for s in sorted(rel["seed"].unique()):
        fold = folds_for(groups, s)
        preds = {}
        for vname, Xv in variants.items():
            if args.nested:
                preds[vname] = np.exp(nested[(vname, s)])
                continue
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
        per_genome.append(merged[["genome_name", "measured", "gRodonpred", "gRodon without temperature", "combopred", "phylopred"]
                                  + list(variants)].assign(seed=s))
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
    out.to_csv(processed / ((f"released_tools_comparison_nested{sfx}.csv") if args.nested else f"released_tools_comparison{sfx}.csv"), index=False)
    # per-genome predictions, for paired tests (pressure_tests.py)
    pd.concat(per_genome).to_csv(processed / (f"released_tools_genomes_nested{sfx}.csv" if args.nested else f"released_tools_genomes{sfx}.csv"), index=False)
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
