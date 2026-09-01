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
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

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


def most_predictive_latent_dim(latents: np.ndarray, labels: np.ndarray) -> int:
    """Which single latent dimension best separates the two classes (for intervention.py)."""
    scores = []
    for d in range(latents.shape[1]):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(latents[:, [d]], labels)
        scores.append(clf.score(latents[:, [d]], labels))
    return int(np.argmax(scores))
