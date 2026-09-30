#!/usr/bin/env python
"""This tests a two-stage model for the fast-growth compression problem:
classify fast versus slow first, then use a regime-specific regressor for
each, instead of one regressor trying to fit both regimes at once. Huber
loss (found to help on 2026-09-30, see research_log.md) is used throughout
since it already beat plain squared error on its own.

Both the classifier and the two regime-specific regressors are trained only
on the fold's training split, exactly like every other benchmark here, so a
test genome's regime is never known to the model that predicts it, the
classifier has to infer it the same way the real deployment pipeline would.

Reports three versions so the routing choice itself can be checked: hard
routing (use whichever regressor the classifier picked), soft routing
(blend both regressors' predictions by the classifier's predicted
probability), and the single-model Huber baseline from the same folds, for
a fair, identical-split comparison.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", type=str, default="features_sample_gapfilled.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    split_hours = data_cfg["doubling_time_split_hours"]
    all_branches = [b["name"] for b in model_cfg["branches"]]

    df = pd.read_csv(processed_dir / args.features_file)
    target_log = np.log(df["doubling_time_hours_ref"].to_numpy())
    y_true = df["doubling_time_hours_ref"].to_numpy()
    stratify_labels = (y_true < split_hours).astype(int)

    tensors, _ = build_branch_tensors(df, model_cfg, branches=all_branches)
    raw_x = concat_branch_tensors(tensors).numpy()

    n = len(df)
    oof_single = np.full(n, np.nan)
    oof_hard = np.full(n, np.nan)
    oof_soft = np.full(n, np.nan)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for train_idx, test_idx in skf.split(np.zeros(n), stratify_labels):
        x_train, x_test = raw_x[train_idx], raw_x[test_idx]
        y_train = target_log[train_idx]
        labels_train = stratify_labels[train_idx]

        single = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3, loss="huber", alpha=0.9)
        single.fit(x_train, y_train)
        oof_single[test_idx] = single.predict(x_test)

        clf = GradientBoostingClassifier(random_state=args.seed, n_estimators=200, max_depth=3)
        clf.fit(x_train, labels_train)
        proba_fast_test = clf.predict_proba(x_test)[:, list(clf.classes_).index(1)]
        hard_pred_regime = (proba_fast_test >= 0.5).astype(int)

        fast_mask = labels_train == 1
        slow_mask = labels_train == 0
        fast_reg = GradientBoostingRegressor(random_state=args.seed, n_estimators=100, max_depth=2, loss="huber", alpha=0.9)
        fast_reg.fit(x_train[fast_mask], y_train[fast_mask])
        slow_reg = GradientBoostingRegressor(random_state=args.seed, n_estimators=100, max_depth=2, loss="huber", alpha=0.9)
        slow_reg.fit(x_train[slow_mask], y_train[slow_mask])

        fast_pred_test = fast_reg.predict(x_test)
        slow_pred_test = slow_reg.predict(x_test)

        oof_hard[test_idx] = np.where(hard_pred_regime == 1, fast_pred_test, slow_pred_test)
        oof_soft[test_idx] = proba_fast_test * fast_pred_test + (1 - proba_fast_test) * slow_pred_test

    predictions = {
        "single_huber": np.exp(oof_single),
        "two_stage_hard": np.exp(oof_hard),
        "two_stage_soft": np.exp(oof_soft),
    }
    table = stratified_benchmark(predictions, y_true, split_hours)
    logger.info(f"k={args.k}-fold cross-validation (seed={args.seed}), n={len(df)}, single Huber vs. two-stage classify-then-regress")
    print(table.to_string(index=False))

    out_path = processed_dir / "benchmark_results_two_stage.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
