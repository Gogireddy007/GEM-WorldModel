#!/usr/bin/env python
"""Week 5: fine-tune the growth-rate head on the pretrained context encoder,
reproduce gRodon/Phydon baselines on the same split, and run the stratified
(<5h / >=5h) benchmark.
"""

import pickle

import numpy as np
import pandas as pd
import torch

from gem_worldmodel.eval.benchmark import stratified_benchmark
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.finetune import finetune
from gem_worldmodel.training.pretrain import load_checkpoint
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    train_cfg = load_config("train")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    ckpt_dir = resolve_path(train_cfg["pretrain"]["checkpoint_dir"])

    df = pd.read_csv(processed_dir / "features_sample.csv")
    with open(ckpt_dir / "standardizers.pkl", "rb") as f:
        standardizers = pickle.load(f)
    tensors, _ = build_branch_tensors(df, model_cfg, standardizers)

    jepa = load_checkpoint(model_cfg, ckpt_dir / "jepa_pretrained.pt")
    target_log = torch.tensor(np.log(df["doubling_time_hours_ref"].to_numpy()), dtype=torch.float32)

    ft_result = finetune(jepa, tensors, target_log, train_cfg)
    splits = ft_result["splits"]
    test_idx = splits["test"]
    if len(test_idx) == 0:
        logger.warning("empty test split at this sample size, falling back to train+val for the benchmark demo")
        test_idx = np.concatenate([splits["train"], splits["val"]])

    with torch.no_grad():
        test_batch = {name: tensor[torch.tensor(test_idx)] for name, tensor in tensors.items()}
        z = ft_result["jepa"].context_encoder(test_batch)
        our_pred = np.exp(ft_result["head"](z).numpy())

    df_test = df.iloc[test_idx].reset_index(drop=True)
    grodon = GRodonBaseline().fit(df.iloc[splits["train"]])
    phydon = PhydonBaseline().fit(df.iloc[splits["train"]])

    predictions = {
        "gem_worldmodel": our_pred,
        "grodon_reproduction": grodon.predict(df_test),
        "phydon_reproduction": phydon.predict(df_test),
    }
    y_true = df_test["doubling_time_hours_ref"].to_numpy()

    table = stratified_benchmark(predictions, y_true, data_cfg["doubling_time_split_hours"])
    logger.info(f"n_test={len(test_idx)}")
    print(table.to_string(index=False))

    out_path = processed_dir / "week5_benchmark.csv"
    table.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
