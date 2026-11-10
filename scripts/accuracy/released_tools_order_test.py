#!/usr/bin/env python
"""Per-genome test of our model against the released gRodon and Phydon with whole orders held out
(released_tools_genomes_nested_order.csv from compare_released_tools.py --nested --level order, Phydon's own training table
cut by order in scripts/r/run_released_phydon.R). Same test as the species-level one in pressure_tests.py: Wilcoxon
signed-rank on the absolute log errors averaged over seeds, on the genomes both methods predicted, with a bootstrap interval
for the difference in typical error.

Writes released_tools_order_paired.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gem_worldmodel.utils.config import load_config, resolve_path

DRAWS = 2000


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    g = pd.read_csv(processed / "released_tools_genomes_nested_order.csv")
    g = g[g["gRodonpred"].notna() & (g["gRodonpred"] > 0)]
    cols = {"this model": "this model", "this model + measured temperature": "this model + measured temperature",
            "gRodon (released)": "gRodonpred", "gRodon (released), no temperature": "gRodon without temperature",
            "Phydon (released)": "combopred"}
    ae = pd.DataFrame({k: np.abs(np.log(g[c]) - np.log(g["measured"])) for k, c in cols.items()})
    ae["genome_name"] = g["genome_name"].to_numpy()
    per_all = ae.groupby("genome_name").mean()
    rng = np.random.default_rng(0)
    rows = []
    for ours in ("this model", "this model + measured temperature"):
        for other in ("gRodon (released)", "Phydon (released)", "gRodon (released), no temperature"):
            per = per_all[[ours, other]].dropna()
            d = per[ours] - per[other]
            diff = lambda r, o=ours, t=other, per=per: np.exp(np.median(per[o].to_numpy()[r])) - np.exp(np.median(per[t].to_numpy()[r]))  # noqa: E731
            lo, hi = np.percentile([diff(rng.integers(0, len(per), len(per))) for _ in range(DRAWS)], [2.5, 97.5])
            rows.append({"held_out": "order", "method": f"{ours} vs {other}", "value": diff(np.arange(len(per))), "low": lo, "high": hi,
                         "n": len(per), "better": int((d < 0).sum()), "p": wilcoxon(d).pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(Path(processed) / "released_tools_order_paired.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
