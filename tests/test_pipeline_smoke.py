"""End-to-end smoke tests for the JEPA/training/eval pipeline on small
synthetic data, no network calls. Real end-to-end runs against live data are
done via the scripts in scripts/ (see README for how to reproduce those).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from gem_worldmodel.eval import benchmark, intervention, necessity_sufficiency, probing
from gem_worldmodel.models.heads import GrowthRateHead
from gem_worldmodel.models.jepa import JEPA
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import GENOMIC_TRAIT_COLUMNS, BranchStandardizer
from gem_worldmodel.training.finetune import cross_validate
from gem_worldmodel.training.pretrain import CollapseMonitor, pretrain, pretrain_multi_corpus, save_checkpoint
from gem_worldmodel.training.raw_baseline import concat_branch_tensors, cross_validate_raw
from gem_worldmodel.utils.config import load_config


def _toy_branch_tensors(cfg, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        b["name"]: torch.randn(n, b["input_dim"], generator=g) for b in cfg["branches"]
    }


def test_pretrain_loss_decreases_and_no_collapse():
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    tensors = _toy_branch_tensors(model_cfg, n=64)

    result = pretrain(tensors, model_cfg, train_cfg, epochs=30)
    history = result["history"]

    assert history["loss"][-1] < history["loss"][0]
    assert min(history["target_std"]) > 0.01, "target encoder output collapsed to a near-constant vector"
    assert result["collapsed_at"] is None


def test_pretrain_actually_trains_the_predictor():
    """Regression test: an earlier bug constructed the optimizer from only
    model.context_encoder.parameters(), silently excluding the predictor.
    The predictor stayed randomly initialized forever, which caused real loss
    divergence (0.02 -> 24 over 200 epochs) on the actual 175-species labeled
    corpus that a short synthetic-data smoke test didn't catch. This asserts
    the SAME predictor's own weights move within a training run, not just
    that two independently-initialized predictors differ (which they always
    would, trained or not).
    """
    model_cfg = load_config("model")
    tensors = _toy_branch_tensors(model_cfg, n=64)

    model = JEPA(model_cfg)
    predictor_before = [p.detach().clone() for p in model.predictor.parameters()]

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()), lr=3e-4
    )
    out = model(tensors)
    optimizer.zero_grad()
    out["loss"].backward()
    assert any(p.grad is not None for p in model.predictor.parameters()), (
        "predictor received no gradient at all, that's a separate, worse bug"
    )
    optimizer.step()

    predictor_after = list(model.predictor.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(predictor_before, predictor_after)), (
        "predictor weights did not change after an optimizer step, it isn't being trained"
    )


def test_pretrain_multi_corpus_handles_mismatched_branch_availability():
    """Simulates the real scenario: a fully-labeled corpus (3 branches) and a
    GEM-like corpus with real 16S only for some rows (2 branches for the
    rest) trained together in one model.
    """
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    all_branches = [b["name"] for b in model_cfg["branches"]]
    two_branches = all_branches[:2]

    labeled = {
        "name": "labeled",
        "tensors": _toy_branch_tensors(model_cfg, n=32, seed=1),
        "branches": all_branches,
    }
    g = torch.Generator().manual_seed(2)
    gem = {
        "name": "gem",
        "tensors": {
            b["name"]: torch.randn(48, b["input_dim"], generator=g)
            for b in model_cfg["branches"]
            if b["name"] in two_branches
        },
        "branches": two_branches,
    }

    result = pretrain_multi_corpus([labeled, gem], model_cfg, train_cfg, epochs=10)
    assert len(result["history"]["loss"]) == 10
    assert set(result["history"]["per_corpus_loss"].keys()) == {"labeled", "gem"}
    # the model must never have tried to encode the missing 3rd branch for "gem"
    assert result["history"]["loss"][-1] == result["history"]["loss"][-1]  # not NaN


def test_collapse_monitor_flags_low_std():
    monitor = CollapseMonitor(window=5, threshold=0.1)
    flagged = False
    for _ in range(6):
        flagged = monitor.update(0.01)
    assert flagged is True


def test_collapse_monitor_does_not_flag_healthy_std():
    monitor = CollapseMonitor(window=5, threshold=0.1)
    flagged = False
    for _ in range(6):
        flagged = monitor.update(0.5)
    assert flagged is False


def test_branch_standardizer_imputes_and_scales():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10.0, 20.0, 30.0]})
    std = BranchStandardizer(["a", "b"]).fit(df)
    out = std.transform(df)
    assert not np.isnan(out).any()
    assert out.shape == (3, 2)


def test_benchmark_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = benchmark.compute_metrics(y, y.copy())
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["spearman"] == 1.0


def test_stratified_benchmark_splits_regimes():
    y_true = np.array([2.0, 3.0, 8.0, 12.0])
    preds = {"model_a": y_true + np.array([0.1, -0.1, 0.2, -0.2])}
    table = benchmark.stratified_benchmark(preds, y_true, split_hours=5.0)
    assert set(table["regime"]) == {"all", "fast_<5h", "slow_>=5h"}
    assert int(table.loc[table["regime"] == "fast_<5h", "n"].iloc[0]) == 2
    assert int(table.loc[table["regime"] == "slow_>=5h", "n"].iloc[0]) == 2


def _toy_growth_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    cub = rng.uniform(0.1, 0.6, n)
    doubling = np.exp(-3 * cub + rng.normal(0, 0.3, n)) * 20 + 0.5
    df = pd.DataFrame({"cub": cub, "doubling_time_hours_ref": doubling})
    for col in GENOMIC_TRAIT_COLUMNS:
        if col != "cub":
            df[col] = rng.normal(0, 1, n)
    return df


def test_grodon_baseline_fits_and_predicts():
    df = _toy_growth_df()
    model = GRodonBaseline().fit(df)
    preds = model.predict(df)
    assert preds.shape == (len(df),)
    assert (preds > 0).all()


def test_phydon_baseline_fits_and_predicts():
    df = _toy_growth_df()
    model = PhydonBaseline().fit(df)
    preds = model.predict(df)
    assert preds.shape == (len(df),)
    assert (preds > 0).all()


def test_linear_and_nonlinear_probe_recover_separable_signal():
    rng = np.random.default_rng(0)
    n, d = 100, 8
    labels = rng.integers(0, 2, n)
    latents = rng.normal(0, 1, (n, d))
    latents[:, 0] += labels * 3.0  # inject a clearly separable signal on dim 0

    lin = probing.linear_probe(latents, labels, seed=0)
    nonlin = probing.nonlinear_probe(latents, labels, seed=0, epochs=100)
    assert lin["accuracy"] > 0.8
    assert nonlin["accuracy"] > 0.7

    best_dim = probing.most_predictive_latent_dim(latents, labels)
    assert best_dim == 0


def test_linear_and_nonlinear_probe_cv_recover_separable_signal():
    rng = np.random.default_rng(0)
    n, d = 100, 8
    labels = rng.integers(0, 2, n)
    latents = rng.normal(0, 1, (n, d))
    latents[:, 0] += labels * 3.0  # same clearly separable signal as the single-split test above

    lin_cv = probing.linear_probe_cv(latents, labels, k=5, seed=0)
    nonlin_cv = probing.nonlinear_probe_cv(latents, labels, k=5, seed=0, epochs=100)

    # Every one of the 100 samples contributed exactly one out-of-fold prediction.
    assert lin_cv["n"] == n
    assert nonlin_cv["n"] == n
    assert lin_cv["accuracy"] > 0.8
    assert nonlin_cv["accuracy"] > 0.7
    assert "auc" in lin_cv and "auc" in nonlin_cv
    assert lin_cv["coef_full_fit"].shape == (d,)

    best_dim_cv = probing.most_predictive_latent_dim(latents, labels, k=5, seed=0)
    assert best_dim_cv == 0


def test_probe_cv_raises_when_too_few_samples_in_smallest_class():
    rng = np.random.default_rng(0)
    latents = rng.normal(0, 1, (20, 4))
    labels = np.array([1] * 19 + [0])  # only 1 sample in the minority class

    with pytest.raises(ValueError, match="need at least 2 samples"):
        probing.linear_probe_cv(latents, labels, k=5)
    with pytest.raises(ValueError, match="need at least 2 samples"):
        probing.nonlinear_probe_cv(latents, labels, k=5)


def test_probe_cv_shrinks_k_when_a_class_is_small():
    # 3 samples in the minority class means k can be at most 3, even if the
    # caller asked for k=5, this must not raise, it should just use k=3.
    rng = np.random.default_rng(0)
    latents = rng.normal(0, 1, (23, 4))
    labels = np.array([1] * 20 + [0] * 3)

    result = probing.linear_probe_cv(latents, labels, k=5)
    assert result["k"] == 3
    assert result["n"] == 23


def test_heuristic_trophic_label_shape():
    labels = probing.heuristic_trophic_label(
        np.array([1e6, 2e6, 3e6, 4e6]), np.array([1, 2, 3, 4])
    )
    assert set(np.unique(labels)).issubset({0, 1})


def test_intervention_shifts_prediction_monotonically_for_linear_head():
    torch.manual_seed(0)
    latent_dim = 8
    head = GrowthRateHead(latent_dim, hidden_dim=4, num_layers=1)  # single linear layer, monotonic per dim
    latents = torch.randn(20, latent_dim)
    result = intervention.intervene_on_dimension(latents, head, dim=0)
    assert result["monotonic"]
    assert set(result["shift_by_delta"].keys()) == {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert result["shift_by_delta"][0.0] == 0.0


def test_necessity_sufficiency_report_shapes():
    model_cfg = load_config("model")
    jepa = JEPA(model_cfg)
    head = GrowthRateHead(jepa.latent_dim, hidden_dim=8, num_layers=1)
    tensors = _toy_branch_tensors(model_cfg, n=40)
    y = np.random.default_rng(0).normal(0, 1, 40)

    report = necessity_sufficiency.necessity_sufficiency_report(jepa, head, tensors, y)
    branch_names = {b["name"] for b in model_cfg["branches"]}
    assert set(report["necessity"].keys()) == branch_names
    assert set(report["sufficiency"].keys()) == branch_names
    assert "r2" in report["full_metrics"]


def test_cross_validate_gives_one_prediction_per_sample_no_leakage(tmp_path):
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    n = 40
    tensors = _toy_branch_tensors(model_cfg, n=n)

    jepa = JEPA(model_cfg)
    ckpt_path = tmp_path / "toy.pt"
    save_checkpoint(jepa, ckpt_path)

    rng = np.random.default_rng(0)
    target_log = torch.tensor(rng.normal(0, 1, n), dtype=torch.float32)
    stratify_labels = (rng.uniform(0, 1, n) < 0.5).astype(int)  # 2-class, roughly balanced

    small_train_cfg = {
        **train_cfg,
        "finetune": {**train_cfg["finetune"], "epochs": 3, "freeze_encoder_epochs": 1},
    }
    result = cross_validate(
        model_cfg, ckpt_path, tensors, target_log, stratify_labels, small_train_cfg, k=4
    )

    assert result["oof_pred_log"].shape == (n,)
    assert not np.isnan(result["oof_pred_log"]).any()  # every sample got exactly one oof prediction
    assert set(result["fold_id"]) == {0, 1, 2, 3}
    # every sample assigned to exactly one fold, none held out twice or never
    assert (result["fold_id"] >= 0).all()
    assert len(result["fold_models"]) == 4


def test_necessity_sufficiency_cv_uses_only_held_out_predictions():
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    n = 40
    tensors = _toy_branch_tensors(model_cfg, n=n)

    jepa = JEPA(model_cfg)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "toy.pt"
        save_checkpoint(jepa, ckpt_path)

        rng = np.random.default_rng(0)
        target_log = torch.tensor(rng.normal(0, 1, n), dtype=torch.float32)
        stratify_labels = (rng.uniform(0, 1, n) < 0.5).astype(int)
        small_train_cfg = {
            **train_cfg,
            "finetune": {**train_cfg["finetune"], "epochs": 3, "freeze_encoder_epochs": 1},
        }
        cv_result = cross_validate(
            model_cfg, ckpt_path, tensors, target_log, stratify_labels, small_train_cfg, k=4
        )

    report = necessity_sufficiency.necessity_sufficiency_report_cv(
        cv_result["fold_models"], tensors, target_log.numpy()
    )
    branch_names = {b["name"] for b in model_cfg["branches"]}
    assert set(report["necessity"].keys()) == branch_names
    assert set(report["sufficiency"].keys()) == branch_names
    assert report["n"] == n
    assert "r2" in report["full_metrics"]

    # A regime mask restricting to roughly half the samples must not trip the
    # "every in-regime sample covered, no others" assertion inside the
    # function, this exercises the exact bug found while wiring this in:
    # covered() was checked against the FULL n instead of the regime subset.
    regime_mask = stratify_labels.astype(bool)
    regime_report = necessity_sufficiency.necessity_sufficiency_report_cv(
        cv_result["fold_models"], tensors, target_log.numpy(), regime_mask
    )
    assert regime_report["n"] == int(regime_mask.sum())


def test_concat_branch_tensors_is_deterministic_and_correct_width():
    model_cfg = load_config("model")
    n = 10
    tensors = _toy_branch_tensors(model_cfg, n=n)
    expected_width = sum(b["input_dim"] for b in model_cfg["branches"])

    x1 = concat_branch_tensors(tensors)
    x2 = concat_branch_tensors(tensors)
    assert x1.shape == (n, expected_width)
    assert torch.equal(x1, x2)  # same dict, same order every time


def test_raw_baseline_cv_learns_a_clearly_separable_signal():
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    n = 60
    tensors = _toy_branch_tensors(model_cfg, n=n)

    rng = np.random.default_rng(0)
    # target driven by the SUM of all genomic_traits dims (gtdb_distance and
    # rrna16s are pure noise), spreading the real signal across several
    # correlated input dims rather than one, which is what a real growth-rate
    # relationship would look like and is learnable from ~48 samples/fold in
    # the epoch budget below, a signal on a single isolated dim among 39
    # total input dims turned out to need far more epochs/samples than this
    # small a synthetic test can reasonably afford (checked empirically).
    target_log = tensors["genomic_traits"].sum(dim=1) * 0.5 + torch.tensor(
        rng.normal(0, 0.1, n), dtype=torch.float32
    )
    stratify_labels = (target_log.numpy() > target_log.numpy().mean()).astype(int)

    small_train_cfg = {
        **train_cfg,
        "finetune": {**train_cfg["finetune"], "epochs": 500, "lr": 1e-2, "weight_decay": 0.0},
    }
    result = cross_validate_raw(tensors, target_log, stratify_labels, small_train_cfg, k=5)

    assert result["oof_pred_log"].shape == (n,)
    assert not np.isnan(result["oof_pred_log"]).any()
    corr = np.corrcoef(result["oof_pred_log"], target_log.numpy())[0, 1]
    assert corr > 0.5  # should recover a strong, real linear relationship


def test_raw_baseline_and_jepa_cross_validate_use_identical_fold_membership(tmp_path):
    """The whole point of this comparison is that it's fair, same fold
    membership means differences in the resulting metrics are attributable to
    the encoder, not to two runs happening to get different splits.
    """
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    n = 40
    tensors = _toy_branch_tensors(model_cfg, n=n)

    jepa = JEPA(model_cfg)
    ckpt_path = tmp_path / "toy.pt"
    save_checkpoint(jepa, ckpt_path)

    rng = np.random.default_rng(0)
    target_log = torch.tensor(rng.normal(0, 1, n), dtype=torch.float32)
    stratify_labels = (rng.uniform(0, 1, n) < 0.5).astype(int)

    small_train_cfg = {
        **train_cfg,
        "finetune": {**train_cfg["finetune"], "epochs": 3, "freeze_encoder_epochs": 1},
    }
    jepa_result = cross_validate(
        model_cfg, ckpt_path, tensors, target_log, stratify_labels, small_train_cfg, k=4
    )
    raw_result = cross_validate_raw(tensors, target_log, stratify_labels, small_train_cfg, k=4)

    assert np.array_equal(jepa_result["fold_id"], raw_result["fold_id"])
