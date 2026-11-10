#!/usr/bin/env python
"""External test: species whose doubling time comes from their species description and that are in neither the Madin
compilation nor our labels (data/curated/external_growth_rates.csv, inputs from build_external_features.py).

The released model is trained exactly as deployed (deployed settings, each labeled genome once per tree placement) on the
training table of the paper, loaded the standard way, and predicts the external genomes at their own GTDB placement,
averaged over five seeds. Reported for the doubling time at the growth temperature:

  - typical fold error and Spearman correlation, overall and by the closest labeled relative of each genome, with 95%
    bootstrap intervals over genera
  - the relatives' median (median label of the training genomes at the closest shared rank) and the released gRodon
    without a temperature (its own estimate, an input of the model) on the same genomes, with paired Wilcoxon tests
  - coverage of the released 80% ranges (range_calibration.csv, by novelty level and domain)
  - the slow-growth probability: AUC for a doubling time above a day

Writes external_test.csv (one row per genome) and external_summary.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_paper_headline as headline  # noqa: E402

from gem_worldmodel import benchmark as gb  # noqa: E402
from gem_worldmodel.features import deploy_features  # noqa: E402
from gem_worldmodel.models.deployed import DeployedRegressor  # noqa: E402
from gem_worldmodel.utils.config import load_config, resolve_path  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
LEVEL = {"same genus": "species", "same family, new genus": "genus", "same order, new family": "family", "new order": "order"}
DRAWS = 2000


def tier(taxonomy, train_tax):
    for name, p in (("same genus", "g__"), ("same family, new genus", "f__"), ("same order, new family", "o__")):
        tok = gb.token(taxonomy, p)
        if tok != "none" and tok in train_tax[p]:
            return name
    return "new order"


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    _, train, X_train, copies = headline.load()               # the paper's training table, unchanged
    y_train = np.log(train["doubling_time_hours"].to_numpy(float))
    wanted = set(deploy_features.GENOME_ROWS.values()) | {"external"}
    full, X_full, names = deploy_features.load_training_table(processed, k, with_cub=True, with_lifestyle=True, genome_rows=wanted)
    ext_mask = (full["origin"] == "external").to_numpy() & full["has_cub"].to_numpy()
    ext, X_ext = full[ext_mask].reset_index(drop=True), X_full[ext_mask]
    assert not set(ext["accession"]) & set(train["accession"]), "an external genome is in the training table"
    y = np.log(ext["doubling_time_hours"].to_numpy(float))

    coarse = np.vstack(copies)
    coarse_y = np.tile(y_train, len(copies))
    pred = np.mean([DeployedRegressor("own temperature", s).fit(coarse, coarse_y).predict(X_ext) for s in SEEDS], axis=0)
    base = gb.relatives_median(train.assign(y=y_train), ext)
    grodon = X_ext[:, names.index("log_grodon_d")]
    has_grodon = ext["has_grodon"].to_numpy() if "has_grodon" in ext else np.ones(len(ext), bool)

    train_tax = {p: set(train["gtdb_taxonomy"].map(lambda t, p=p: gb.token(t, p))) for p in ("g__", "f__", "o__")}
    ext["tier"] = ext["gtdb_taxonomy"].map(lambda t: tier(t, train_tax))
    ext["domain"] = np.where(ext["archaeon"] == 1, "archaea", "bacteria")
    cal = pd.read_csv(processed / "range_calibration.csv")
    rng_ = cal[(cal["kind"] == "range") & (cal["target"] == "own temperature")].set_index(["held_out", "domain"])
    q = rng_.loc[list(zip(ext["tier"].map(LEVEL), ext["domain"]))]
    resid = y - pred
    inside = (resid >= q["q10"].to_numpy()) & (resid <= q["q90"].to_numpy())
    lg = cal[cal["kind"] == "slow logistic"].set_index("held_out").loc[ext["tier"].map(LEVEL)]
    prob = 1 / (1 + np.exp(-(lg["intercept"].to_numpy() + lg["coef_pred"].to_numpy() * pred + lg["coef_archaeon"].to_numpy() * ext["archaeon"].to_numpy())))

    out = ext[["species", "accession", "gtdb_taxonomy", "tier", "domain", "doubling_time_hours", "source"]].copy() \
        if "source" in ext else ext[["species", "accession", "gtdb_taxonomy", "tier", "domain", "doubling_time_hours"]].copy()
    out = out.assign(predicted_hours=np.exp(pred), relatives_median_hours=np.exp(base), grodon_hours=np.exp(grodon),
                     inside_range=inside, prob_slow=prob)
    out.to_csv(processed / "external_test.csv", index=False)

    genus = ext["gtdb_taxonomy"].map(lambda t: gb.token(t, "g__")).to_numpy()
    rs = np.random.default_rng(0)
    ug = np.unique(genus)
    idx = {g: np.flatnonzero(genus == g) for g in ug}
    err = lambda p, r: np.exp(np.median(np.abs(y[r] - p[r])))  # noqa: E731

    def ci(fn):
        vals = [fn(np.concatenate([idx[g] for g in rs.choice(ug, len(ug))])) for _ in range(DRAWS)]
        return np.percentile(vals, [2.5, 97.5])

    rows = []
    allr = np.arange(len(y))
    for name, p, mask in (("model", pred, np.ones(len(y), bool)), ("relatives' median", base, np.ones(len(y), bool)),
                          ("gRodon, no temperature", grodon, has_grodon)):
        r = allr[mask]
        lo, hi = ci(lambda rr, p=p: err(p, rr))
        rows.append({"group": "all", "method": name, "n": len(r), "typical error": err(p, r), "low": lo, "high": hi,
                     "Spearman": spearmanr(y[r], p[r]).correlation})
    for t in LEVEL:
        r = allr[(ext["tier"] == t).to_numpy()]
        if len(r) >= 3:
            for name, p in (("model", pred), ("relatives' median", base)):
                rows.append({"group": t, "method": name, "n": len(r), "typical error": err(p, r),
                             "Spearman": spearmanr(y[r], p[r]).correlation if len(r) > 3 else np.nan})
    for other, p in (("relatives' median", base), ("gRodon, no temperature", grodon)):
        m = has_grodon if other.startswith("gRodon") else np.ones(len(y), bool)
        d = np.abs(y - pred)[m] - np.abs(y - p)[m]
        rows.append({"group": "paired", "method": f"model vs {other}", "n": int(m.sum()), "better": int((d < 0).sum()), "p": wilcoxon(d).pvalue})
    slow = y > np.log(24)
    rows.append({"group": "ranges", "method": "coverage", "n": len(y), "value": inside.mean()})
    rows.append({"group": "ranges", "method": "coverage, slow in truth", "n": int(slow.sum()), "value": inside[slow].mean()})
    if 0 < slow.sum() < len(y):
        rows.append({"group": "slow", "method": "AUC of the slow-growth probability", "n": int(slow.sum()), "value": roc_auc_score(slow, prob)})
        rows.append({"group": "slow", "method": "predicted above a day", "n": int(slow.sum()), "value": int((pred > np.log(24)).sum()),
                     "better": int(((pred > np.log(24)) & slow).sum())})
    s = pd.DataFrame(rows)
    s.to_csv(processed / "external_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(y)} external genomes (of {int((full['origin'] == 'external').sum())}); tiers: {ext['tier'].value_counts().to_dict()}")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
