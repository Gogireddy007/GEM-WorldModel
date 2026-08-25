"""Linear/nonlinear probing of the joint latent for oligotroph vs.
copiotroph status, plus activation intervention.

One thing to watch out for: oligotroph status is partly defined by the
absence of a CUB signal in the first place (oligotrophs show weak codon
usage bias because they aren't under strong translational-efficiency
selection). So a probe that "discovers" CUB-correlated latent structure
predicting oligotroph status could just be circular rather than a real
finding. A positive probing result here is a necessary but not sufficient
check, it needs corroboration from the intervention test in
`intervention.py`, and ideally from an ecological label that wasn't itself
derived from CUB.

There's no literature-curated oligotroph/copiotroph label bundled with this
repo. `heuristic_trophic_label` is a proxy based on rRNA operon copy number
and genome size (per Klappenbach et al. 2000 and Lauro et al. 2009's
copiotroph/oligotroph genomic correlates), good enough for testing the
pipeline end to end but not for drawing real conclusions. Swap in an actual
ecological label before doing that.
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
