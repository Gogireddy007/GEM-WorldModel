#!/usr/bin/env python
"""This checks how the reference-temperature correction shapes what the model
learns, with the median-loss deployment model, whole species held out, seven
seeds.

Doubling times are scaled to a reference temperature (20 C) with a universal
activation energy, which multiplies a hot organism's doubling time by a large
factor. This script measures:

  A. how well growth temperature itself can be predicted from the features
  B. accuracy for three targets: the corrected doubling time (current), the raw
     doubling time at the organism's own growth temperature, and corrected
     variants with other activation energies or a 37 C reference
  C. ranking ability within temperature bands, which shows whether the model
     ranks growth rates beyond just detecting hot organisms
  D. whether predicting growth temperature and the raw doubling time separately
     reproduces the corrected prediction
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from gem_worldmodel.features import deploy_features as df_
from gem_worldmodel.features import temperature
from gem_worldmodel.utils.config import load_config, resolve_path

SEEDS = [42, 1, 7, 13, 2024, 123, 99]
BINS = [(-1, 30, "below 30 C"), (30, 45, "30 to 45 C"), (45, 200, "45 C and above")]


def oof(X, y, groups, seed, loss="quantile"):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    perm = dict(zip(ug, rng.permutation(len(ug))))
    g = np.array([perm[v] for v in groups])
    pred = np.full(len(y), np.nan)
    kw = {"loss": "quantile", "alpha": 0.5} if loss == "quantile" else {}
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, **kw).fit(X[tr], y[tr]).predict(X[te])
    return pred


def corrected(raw_h, temp_c, ea, ref_k):
    cfg = load_config("features")
    cfg = {"temperature": {**cfg["temperature"], "activation_energy_j_per_mol": ea, "reference_temp_k": ref_k}}
    return temperature.correct_doubling_time(raw_h, temp_c, cfg)


def stats(truth, pred):
    return r2_score(truth, pred), spearmanr(truth, pred).correlation, np.exp(np.median(np.abs(truth - pred)))


def main():
    processed = resolve_path(load_config("data")["paths"]["processed_dir"])
    k = load_config("features")["rrna16s"]["kmer_k"]
    df, X, _ = df_.load_training_table(processed, k)
    has_t = df["growth_temp_c"].notna().to_numpy()
    df, X = df[has_t].reset_index(drop=True), X[has_t]
    sp = df["species"].to_numpy()
    temp_c = df["growth_temp_c"].to_numpy(float)
    raw = df["doubling_time_hours"].to_numpy(float)
    print(f"{len(df)} rows with a growth temperature ({df['archaeon'].sum():.0f} archaeal)\n")

    # ---- A. predicting growth temperature
    res = []
    for s in SEEDS:
        p = oof(X, temp_c, sp, s)
        res.append((r2_score(temp_c, p), np.mean(np.abs(temp_c - p)), spearmanr(temp_c, p).correlation))
    r2, mae, rho = np.mean(res, axis=0)
    print(f"A. GROWTH TEMPERATURE predicted from the features (species held out): R2 {r2:.3f}, mean error {mae:.1f} C, Spearman {rho:.3f}")
    hot = temp_c >= 45
    p = np.mean([oof(X, temp_c, sp, s) for s in SEEDS], axis=0)
    from sklearn.metrics import roc_auc_score
    print(f"   telling organisms that grow at 45 C or above from the rest: AUC {roc_auc_score(hot, p):.3f} (n hot = {int(hot.sum())})\n")

    # ---- B. alternative targets
    targets = {
        "corrected to 20 C, Ea 65 kJ/mol (current)": corrected(raw, temp_c, 65000.0, 293.15),
        "corrected to 20 C, Ea 50 kJ/mol": corrected(raw, temp_c, 50000.0, 293.15),
        "corrected to 20 C, Ea 80 kJ/mol": corrected(raw, temp_c, 80000.0, 293.15),
        "corrected to 37 C, Ea 65 kJ/mol": corrected(raw, temp_c, 65000.0, 310.15),
        "raw, at the organism's own temperature": raw,
    }
    print("B. ACCURACY BY TARGET (species held out, mean of 7 seeds; each against its own target)")
    print(f"{'target':46s}{'R2 log':>9s}{'Spearman':>10s}{'fold error':>12s}")
    preds = {}
    for name, t in targets.items():
        y = np.log(t)
        runs = [oof(X, y, sp, s) for s in SEEDS]
        preds[name] = np.mean(runs, axis=0)
        r = np.mean([stats(y, p) for p in runs], axis=0)
        print(f"{name:46s}{r[0]:9.3f}{r[1]:10.3f}{r[2]:11.2f}x")

    # ---- C. ranking within temperature bands
    print("\nC. RANKING WITHIN TEMPERATURE BANDS (Spearman of the held-out prediction with the truth)")
    print(f"{'target':46s}" + "".join(f"{b[2]:>18s}" for b in BINS))
    print(f"{'(rows in band)':46s}" + "".join(f"{int(((temp_c >= lo) & (temp_c < hi)).sum()):>18d}" for lo, hi, _ in BINS))
    for name in ("corrected to 20 C, Ea 65 kJ/mol (current)", "raw, at the organism's own temperature"):
        y = np.log(targets[name])
        cells = []
        for lo, hi, _ in BINS:
            m = (temp_c >= lo) & (temp_c < hi)
            cells.append(f"{spearmanr(y[m], preds[name][m]).correlation:.3f}")
        print(f"{name:46s}" + "".join(f"{c:>18s}" for c in cells))

    # ---- D. two-step prediction
    y_raw = np.log(raw)
    runs_t = [oof(X, temp_c, sp, s) for s in SEEDS]
    runs_r = [oof(X, y_raw, sp, s) for s in SEEDS]
    direct = np.log(targets["corrected to 20 C, Ea 65 kJ/mol (current)"])
    two = [np.log(corrected(np.exp(r), t, 65000.0, 293.15)) for r, t in zip(runs_r, runs_t)]
    r_two = np.mean([stats(direct, p) for p in two], axis=0)
    r_dir = np.mean([stats(direct, oof(X, direct, sp, s)) for s in SEEDS], axis=0)
    print("\nD. SAME 20 C TARGET, TWO WAYS (species held out)")
    print(f"   directly                                     R2 log {r_dir[0]:.3f}  Spearman {r_dir[1]:.3f}  fold error {r_dir[2]:.2f}x")
    print(f"   predict temperature and raw time, then correct R2 log {r_two[0]:.3f}  Spearman {r_two[1]:.3f}  fold error {r_two[2]:.2f}x")


if __name__ == "__main__":
    main()
