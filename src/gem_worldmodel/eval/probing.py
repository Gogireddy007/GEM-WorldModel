"""Linear/nonlinear probing of the joint latent for oligotroph vs.
copiotroph status, plus activation intervention.

As of 2026-08-31 there IS a real, genome-independent label available:
features/ecological_traits.py:real_trophic_label, built from Madin et al.
2020's curated isolation_source data, not from anything computed off the
genome. scripts/probe_intervene.py uses it by default now.

`heuristic_trophic_label` below is kept only as a documented fallback
(--use-heuristic), and its old circularity caveat still applies to it
specifically: it's a proxy based on rRNA operon copy number and genome size
(per Klappenbach et al. 2000 and Lauro et al. 2009's copiotroph/oligotroph
genomic correlates), and oligotroph status is partly defined by the absence
of a CUB signal in the first place, so a probe that "discovers"
CUB-correlated latent structure predicting the heuristic label risks being
circular rather than a real finding. Checked empirically: the heuristic and
real labels only agree 58.9% of the time on the 124 species where both
exist, barely above chance, so they aren't measuring the same thing and the
heuristic shouldn't be treated as a stand-in for the real label.

`linear_probe`/`nonlinear_probe` fit on a single random 70/30 split, at 124
real-labeled species that leaves ~37 test samples, the same fragility that
was already found and fixed in the benchmark and necessity/sufficiency
evaluations. `linear_probe_cv`/`nonlinear_probe_cv` fix it the same way:
k-fold, every sample gets exactly one out-of-fold prediction, accuracy/AUC
computed once over the full pooled set. scripts/probe_intervene.py uses the
CV versions; the single-split versions are kept only because they're useful,
fast primitives for ad-hoc/exploratory checks and are still covered by
their own tests, they should not be used to report a headline number.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def heuristic_trophic_label(genome_size_bp: np.ndarray, rrna_16s_count: np.ndarray) -> np.ndarray:
    """1 = copiotroph-leaning, 0 = oligotroph-leaning (heuristic proxy, see module docstring)."""
    size_median = np.nanmedian(genome_size_bp)
    rrna_median = np.nanmedian(rrna_16s_count)
    return ((genome_size_bp >= size_median) & (rrna_16s_count >= rrna_median)).astype(int)


def linear_probe(latents: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        latents, labels, test_size=0.3, random_state=seed, stratify=labels if len(set(labels)) > 1 else None
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    probs = clf.predict_proba(x_test)[:, 1]
    result = {"accuracy": accuracy_score(y_test, preds)}
    if len(set(y_test)) > 1:
        result["auc"] = roc_auc_score(y_test, probs)
    result["coef"] = clf.coef_[0]
    return result


def linear_probe_cv(latents: np.ndarray, labels: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """k-fold cross-validated linear probe: every sample gets exactly one
    out-of-fold prediction (via sklearn's cross_val_predict, so the same
    machinery is used for both the class predictions and the probabilities,
    guaranteeing they come from identical fold splits), and accuracy/AUC are
    computed once over the full pooled set rather than one random 30% slice.
    """
    n_per_class = min(np.bincount(labels))
    k = min(k, int(n_per_class))
    if k < 2:
        raise ValueError(f"need at least 2 samples in the smallest class for CV, got {n_per_class}")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000)
    oof_preds = cross_val_predict(clf, latents, labels, cv=cv, method="predict")
    oof_probs = cross_val_predict(clf, latents, labels, cv=cv, method="predict_proba")[:, 1]

    result = {"accuracy": accuracy_score(labels, oof_preds), "k": k, "n": len(labels)}
    if len(set(labels)) > 1:
        result["auc"] = roc_auc_score(labels, oof_probs)
    # Informational only, not used for evaluation: coefficients from one model
    # fit on all the data, for interpreting which latent dims matter, not a
    # claim about out-of-sample performance the way accuracy/auc above are.
    result["coef_full_fit"] = LogisticRegression(max_iter=1000).fit(latents, labels).coef_[0]
    return result


class _SmallMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def nonlinear_probe(latents: np.ndarray, labels: np.ndarray, seed: int = 0, epochs: int = 200) -> dict:
    torch.manual_seed(seed)
    x_train, x_test, y_train, y_test = train_test_split(
        latents, labels, test_size=0.3, random_state=seed, stratify=labels if len(set(labels)) > 1 else None
    )
    model = _SmallMLP(latents.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(x_train_t)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_logits = model(torch.tensor(x_test, dtype=torch.float32))
        probs = torch.sigmoid(test_logits).numpy()
        preds = (probs >= 0.5).astype(int)

    result = {"accuracy": accuracy_score(y_test, preds), "model": model}
    if len(set(y_test)) > 1:
        result["auc"] = roc_auc_score(y_test, probs)
    return result


def nonlinear_probe_cv(
    latents: np.ndarray, labels: np.ndarray, k: int = 5, seed: int = 0, epochs: int = 200
) -> dict:
    """k-fold cross-validated version of nonlinear_probe: a fresh MLP is
    trained per fold (no leakage between folds, same principle as
    training/finetune.py:cross_validate), out-of-fold predictions are pooled
    across all folds, and accuracy/AUC are computed once over the full set.
    """
    n_per_class = min(np.bincount(labels))
    k = min(k, int(n_per_class))
    if k < 2:
        raise ValueError(f"need at least 2 samples in the smallest class for CV, got {n_per_class}")

    n = len(labels)
    oof_probs = np.full(n, np.nan)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(latents, labels)):
        torch.manual_seed(seed + fold)
        model = _SmallMLP(latents.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
        x_train_t = torch.tensor(latents[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(labels[train_idx], dtype=torch.float32)

        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x_train_t), y_train_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            test_logits = model(torch.tensor(latents[test_idx], dtype=torch.float32))
            oof_probs[test_idx] = torch.sigmoid(test_logits).numpy()

    assert not np.isnan(oof_probs).any(), "every sample should get exactly one out-of-fold prediction"
    oof_preds = (oof_probs >= 0.5).astype(int)

    result = {"accuracy": accuracy_score(labels, oof_preds), "k": k, "n": n}
    if len(set(labels)) > 1:
        result["auc"] = roc_auc_score(labels, oof_probs)
    return result


def most_predictive_latent_dim(latents: np.ndarray, labels: np.ndarray, k: int = 5, seed: int = 0) -> int:
    """Which single latent dimension best separates the two classes (for
    intervention.py), scored by cross-validated accuracy rather than in-sample
    fit, picking a dimension based on how well it happens to memorize the
    exact 124 labeled points is a real overfitting risk this avoids.
    """
    from sklearn.model_selection import cross_val_score

    n_per_class = min(np.bincount(labels))
    cv_k = min(k, int(n_per_class))
    scores = []
    for d in range(latents.shape[1]):
        if cv_k < 2:
            clf = LogisticRegression(max_iter=1000).fit(latents[:, [d]], labels)
            scores.append(clf.score(latents[:, [d]], labels))
        else:
            cv = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)
            clf = LogisticRegression(max_iter=1000)
            scores.append(cross_val_score(clf, latents[:, [d]], labels, cv=cv).mean())
    return int(np.argmax(scores))
