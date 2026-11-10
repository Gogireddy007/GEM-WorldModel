"""Tests of the GrowthBench interface (splits, scores, the relatives' median baseline) on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from gem_worldmodel import benchmark as gb


def synthetic(n_orders=6, fams=2, gens=2, spp=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for o in range(n_orders):
        for f in range(fams):
            for g in range(gens):
                for s in range(spp):
                    tax = f"d__Bacteria;p__P{o % 2};c__C{o % 3};o__O{o};f__F{o}_{f};g__G{o}_{f}_{g};s__S{o}_{f}_{g}_{s}"
                    rows.append({"species": f"S{o}_{f}_{g}_{s}", "species_group": f"S{o}_{f}_{g}_{s}", "gtdb_taxonomy": tax,
                                 "y": o + 0.1 * f + 0.01 * g + rng.normal(0, 0.01)})
    t = pd.DataFrame(rows)
    return gb.Benchmark(t, rng.normal(size=(len(t), 3)), ["a", "b", "c"])


@pytest.mark.parametrize("level", gb.LEVELS)
def test_splits_hold_out_whole_taxa(level):
    data = synthetic()
    g = gb.groups(data.table, level)
    seen = np.zeros(len(g), int)
    for tr, te in gb.splits(data.table, level, seed=1):
        assert not set(g[tr]) & set(g[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_splits_depend_on_seed_only():
    t = synthetic().table
    a, b, c = gb.splits(t, "family", 0), gb.splits(t, "family", 0), gb.splits(t, "family", 1)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c, strict=True))


def test_score_typical_error_and_spearman():
    y = np.log([1.0, 2.0, 4.0, 48.0])
    s = gb.score(y, y + np.log(2))
    assert s["typical error"] == pytest.approx(2.0)
    assert s["Spearman"] == pytest.approx(1.0)
    assert s["AUC slow"] == pytest.approx(1.0)
    assert np.isnan(gb.score(y, np.zeros(4))["Spearman"])


def test_relatives_median_uses_closest_shared_rank():
    t = synthetic().table
    test = t[t["gtdb_taxonomy"].str.contains("g__G0_0_0;")]
    train = t.drop(test.index)
    pred = gb.relatives_median(train, test)
    family = train[train["gtdb_taxonomy"].str.contains("f__F0_0;")]
    assert np.allclose(pred, family["y"].median())  # the genus is held out, so the family is the closest rank


def test_evaluate_reports_submission_and_baseline():
    data = synthetic()

    def train_mean(train, test, X_train, X_test):
        return np.full(len(test), train["y"].mean())

    res = gb.evaluate(train_mean, data, levels=("genus", "order"), seeds=(0,))
    assert set(res["method"]) == {"submission", "relatives' median"}
    assert len(res) == 4
    genus = res[(res["held_out"] == "genus")].set_index("method")
    assert genus.loc["relatives' median", "typical error"] < genus.loc["submission", "typical error"]
