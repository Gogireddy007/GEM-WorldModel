"""gRodon and Phydon baseline reproductions, refit on the same train/test
split as our model, for a fair comparison.

These are re-implementations of each method's *feature set and model class*,
refit on our split, not the original papers' fixed published coefficients,
which would (a) not be available to us outside their R packages and (b) not
be a fair comparison anyway since their coefficients were fit on a different,
overlapping sample. This is the standard way to benchmark against a prior
method when you don't have its exact trained artifact.

  - gRodon: log(doubling time) ~ CUB (their core single-feature predictor).
  - PhydonBaseline: gradient-boosted trees over CUB and genome traits. This
    is a stand-in used in earlier benchmarks, NOT Phydon's method, which
    blends a codon-based estimate with the growth rates of the nearest
    relatives on the tree. Kept so earlier results stay reproducible.
  - PhydonStyleBaseline: Phydon's actual design (Xu, Zakem and Weissman 2025),
    see its docstring.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold

from gem_worldmodel.training.dataset import GENOMIC_TRAIT_COLUMNS


class GRodonBaseline:
    """gRodon genuinely cannot predict without a computable CUB, rows missing
    it (e.g. failed NCBI CDS download) are dropped at fit time and predicted
    as NaN, matching the real method's actual limitation rather than
    fabricating a value for it.
    """

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, df, target_col: str = "doubling_time_hours_ref"):
        valid = df["cub"].notna()
        x = df.loc[valid, ["cub"]].to_numpy(dtype=float)
        y = np.log(df.loc[valid, target_col].to_numpy(dtype=float))
        self.model.fit(x, y)
        return self

    def predict(self, df) -> np.ndarray:
        preds = np.full(len(df), np.nan)
        valid = df["cub"].notna().to_numpy()
        if valid.any():
            x = df.loc[valid, ["cub"]].to_numpy(dtype=float)
            preds[valid] = np.exp(self.model.predict(x))
        return preds


class PhydonBaseline:
    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
        self.feature_cols = ["cub"] + GENOMIC_TRAIT_COLUMNS

    def fit(self, df, target_col: str = "doubling_time_hours_ref"):
        self.fill_values_ = df[self.feature_cols].median().fillna(0.0)
        x = df[self.feature_cols].fillna(self.fill_values_).to_numpy(dtype=float)
        y = np.log(df[target_col].to_numpy(dtype=float))
        self.model.fit(x, y)
        return self

    def predict(self, df) -> np.ndarray:
        x = df[self.feature_cols].fillna(self.fill_values_).to_numpy(dtype=float)
        return np.exp(self.model.predict(x))


class PhydonStyleBaseline:
    """Phydon's design, refit on our data: log prediction =
    P * gRodon + (1 - P) * Phylopred (Phydon's default geometric-mean mode),
    where gRodon is a linear regression of log doubling time on codon usage
    bias, Phylopred is the distance-weighted mean of the log doubling times of
    the k nearest training species on the tree, and P, the probability that the
    gRodon estimate beats the phylogenetic one, is a logistic function of the
    distance to the nearest training relative and of the gRodon estimate, learned
    from out-of-fold predictions inside the training data (whole species held
    out), as in the paper.

    Differences from the published package, so the result is read correctly:
    tree distance is approximated by the distance between GTDB landmark vectors
    (bacteria and archaea are only compared within their own tree), codon usage
    bias is our single MILC statistic (Phydon's metagenome mode uses one codon
    statistic too, its full mode uses three), and everything is refit on our
    training folds instead of using Phydon's published fit.

    Inputs are arrays: cub (n,), phylo (n, d) landmark vectors, domain (n,) 0/1,
    species (n,) labels used to keep a species from being its own relative.
    """

    def __init__(self, k: int = 3):
        self.k = k

    def _phylopred(self, phylo, domain, exclude_species, ref_phylo, ref_domain, ref_species, ref_y):
        est = np.full(len(phylo), np.nan)
        near = np.full(len(phylo), np.nan)
        for i in range(len(phylo)):
            ok = (ref_domain == domain[i]) & (ref_species != exclude_species[i])
            if not ok.any():
                continue
            d = np.linalg.norm(ref_phylo[ok] - phylo[i], axis=1)
            order = np.argsort(d)[: self.k]
            w = 1.0 / (d[order] + 1e-6)
            est[i] = np.sum(w * ref_y[ok][order]) / np.sum(w)
            near[i] = d[order[0]]
        return est, near

    def fit(self, cub, phylo, domain, species, y_log):
        self.ref_ = (np.asarray(phylo, float), np.asarray(domain), np.asarray(species), np.asarray(y_log, float))
        self.grodon_ = LinearRegression().fit(np.asarray(cub, float)[:, None], y_log)
        # out-of-fold estimates inside the training data to learn when each estimate wins
        g_oof = np.full(len(y_log), np.nan)
        species = np.asarray(species)
        _, codes = np.unique(species, return_inverse=True)
        for tr, te in GroupKFold(5).split(cub, y_log, codes):
            g_oof[te] = LinearRegression().fit(np.asarray(cub, float)[tr, None], y_log[tr]).predict(np.asarray(cub, float)[te, None])
        p_oof, near = self._phylopred(self.ref_[0], self.ref_[1], species, *self.ref_)
        wins = (np.abs(g_oof - y_log) < np.abs(p_oof - y_log)).astype(int)
        X = np.column_stack([np.log1p(near), g_oof])
        ok = ~np.isnan(X).any(axis=1)
        self.p_model_ = LogisticRegression().fit(X[ok], wins[ok]) if len(set(wins[ok])) > 1 else None
        self.p_const_ = float(wins[ok].mean())
        return self

    def predict_log(self, cub, phylo, domain, species):
        g = self.grodon_.predict(np.asarray(cub, float)[:, None])
        p_est, near = self._phylopred(np.asarray(phylo, float), np.asarray(domain), np.asarray(species), *self.ref_)
        X = np.column_stack([np.log1p(near), g])
        P = np.full(len(g), self.p_const_)
        ok = ~np.isnan(X).any(axis=1)
        if self.p_model_ is not None and ok.any():
            P[ok] = self.p_model_.predict_proba(X[ok])[:, 1]
        out = P * g + (1 - P) * p_est
        return np.where(np.isnan(p_est), g, out)
