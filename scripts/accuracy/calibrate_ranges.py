#!/usr/bin/env python
"""Prediction ranges by novelty level and domain, and a calibrated probability of growing slower than a day, from the
nested held-out predictions of evaluate_paper_headline.py.

The ranges first released were the 10th and 90th percentiles of the held-out residuals at each novelty level. They hold
overall but not for archaea or for slow growers. Here:

  ranges        residual percentiles for each novelty level (the held-out rank) and domain, for the three targets
  coverage      checked by cross-fitting: the percentiles are estimated on four fifths of the genera and applied to the
                rest, and the share of measured values inside the range is reported for all genomes, slow and fast growers,
                bacteria and archaea, for the novelty-only ranges and for the ranges by novelty and domain
  slow          a logistic model of a doubling time above 24 h on the predicted log doubling time and the domain, per
                novelty level, cross-fitted the same way; reported with AUC, Brier score and the calibration by fifths of
                predicted probability

Coverage for genomes that are slow in truth is conditional on the outcome, which no range built from the inputs can
target; the probability is the tool for that question.

Writes range_calibration.csv (the ranges and the logistic coefficients fitted on all rows, for export_hq_table.py) and
range_evaluation.csv (the cross-fitted checks).
"""

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.utils.config import load_config, resolve_path

SLOW = np.log(24.0)
LEVELS = ("species", "genus", "family", "order")


def genus(t):
    m = re.search(r"g__[^;]*", str(t))
    return m.group(0) if m and m.group(0) != "g__" else str(t)


def slow_model():
    return LogisticRegression(C=1.0)


def slow_inputs(d):
    return np.column_stack([d["predicted"], d["archaeon"].astype(float)])


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    hp = pd.read_csv(processed / "paper_headline_predictions.csv")
    hp = hp[hp["scheme"] == "nested"].copy()
    hp["archaeon"] = hp["gtdb_taxonomy"].str.startswith("d__Archaea")
    hp["domain"] = np.where(hp["archaeon"], "archaea", "bacteria")
    hp["resid"] = hp["measured"] - hp["predicted"]
    hp["cluster"] = hp["gtdb_taxonomy"].map(genus)
    cal, ev = [], []

    for target in hp["target"].unique():
        for level in LEVELS:
            d = hp[(hp["target"] == target) & (hp["held_out"] == level)].reset_index(drop=True)
            # final ranges on all rows (seeds pooled)
            for dom, g in [("all", d)] + list(d.groupby("domain")):
                cal.append({"target": target, "held_out": level, "domain": dom, "kind": "range",
                            "q10": g["resid"].quantile(0.1), "q90": g["resid"].quantile(0.9), "n": len(g)})
            # cross-fitted coverage
            inside = {"novelty": np.zeros(len(d), bool), "novelty and domain": np.zeros(len(d), bool)}
            for tr, te in GroupKFold(5).split(d, groups=d["cluster"]):
                a, b = d.iloc[tr], d.iloc[te]
                lo, hi = a["resid"].quantile([0.1, 0.9])
                inside["novelty"][te] = (b["resid"] >= lo) & (b["resid"] <= hi)
                q = a.groupby("domain")["resid"].quantile([0.1, 0.9]).unstack()
                inside["novelty and domain"][te] = (b["resid"] >= b["domain"].map(q[0.1])) & (b["resid"] <= b["domain"].map(q[0.9]))
            groups = {"all": np.ones(len(d), bool), "bacteria": ~d["archaeon"].to_numpy(), "archaea": d["archaeon"].to_numpy()}
            if target != "growth temperature":
                groups["slow in truth"] = (d["measured"] > SLOW).to_numpy()
                groups["fast in truth"] = (d["measured"] <= SLOW).to_numpy()
            for scheme, ins in inside.items():
                for gname, gm in groups.items():
                    ev.append({"test": "coverage", "target": target, "held_out": level, "ranges": scheme, "group": gname,
                               "n": int(gm.sum()), "value": float(ins[gm].mean())})

            if target != "own temperature":
                continue
            y = (d["measured"] > SLOW).to_numpy()
            p = np.zeros(len(d))
            for tr, te in GroupKFold(5).split(d, groups=d["cluster"]):
                p[te] = slow_model().fit(slow_inputs(d.iloc[tr]), y[tr]).predict_proba(slow_inputs(d.iloc[te]))[:, 1]
            bins = pd.qcut(p, 5, duplicates="drop")
            calib = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
            ece = float(np.average(np.abs(calib["p"] - calib["y"]), weights=pd.Series(bins).value_counts(sort=False).to_numpy()))
            for stat, v in (("AUC", roc_auc_score(y, p)), ("Brier", brier_score_loss(y, p)), ("Brier of base rate", brier_score_loss(y, np.full(len(y), y.mean()))),
                            ("calibration error", ece), ("base rate", y.mean()), ("mean probability, slow", p[y].mean()),
                            ("mean probability, fast", p[~y].mean()), ("top fifth, share slow", float(calib["y"].iloc[-1])),
                            ("top fifth, mean probability", float(calib["p"].iloc[-1]))):
                ev.append({"test": "slow probability", "target": target, "held_out": level, "ranges": "", "group": stat, "n": len(d), "value": float(v)})
            m = slow_model().fit(slow_inputs(d), y)
            cal.append({"target": target, "held_out": level, "domain": "all", "kind": "slow logistic",
                        "intercept": float(m.intercept_[0]), "coef_pred": float(m.coef_[0][0]), "coef_archaeon": float(m.coef_[0][1]), "n": len(d)})

    pd.DataFrame(cal).to_csv(processed / "range_calibration.csv", index=False)
    out = pd.DataFrame(ev)
    out.to_csv(processed / "range_evaluation.csv", index=False)
    cov = out[(out["test"] == "coverage") & (out["target"] == "own temperature")]
    print(cov.pivot_table(index=["held_out", "ranges"], columns="group", values="value").round(2).to_string())
    print(out[out["test"] == "slow probability"].pivot_table(index="group", columns="held_out", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
