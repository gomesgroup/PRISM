#!/usr/bin/env python3
"""
Build TRAIN and TEST predictions for the 6-member NNLS ensemble
(cpv2 4 + XGB + CatCPU), including per-point UQ (stds) and split labels.

Inputs:
- results/ensemble_all6_metrics.json (for NNLS weights/intercept)
- results/predictions_each_8_optuna_cpv2_gpu{0,1,2,3}.csv
- results/predictions_each_8_zoo_S97_{XGB,CatCPU}.csv

Output:
- results/ensemble_all6_predictions_full.csv
  Columns: acyl_chlorides, amines, split, y_true, pred_nnls,
           std_unweighted, std_weighted_nnls
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


METRICS_JSON = "results/ensemble_all6_metrics.json"
OUTPUT_CSV = "results/ensemble_all6_predictions_full.csv"

MEMBER_PATHS = [
    "results/predictions_each_8_optuna_cpv2_gpu0.csv",
    "results/predictions_each_8_optuna_cpv2_gpu1.csv",
    "results/predictions_each_8_optuna_cpv2_gpu2.csv",
    "results/predictions_each_8_optuna_cpv2_gpu3.csv",
    "results/predictions_each_8_zoo_S97_XGB.csv",
    "results/predictions_each_8_zoo_S97_CatCPU.csv",
]


def align_split(paths: List[str], split: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[df["split"] == split].copy()
        df = df[["acyl_chlorides", "amines", "y_true", "y_pred"]]
        df = df.rename(columns={"y_pred": f"pred_{Path(p).stem}"})
        frames.append(df)
    if not frames:
        raise SystemExit(f"No frames found for split {split}")
    aligned = frames[0]
    for frm in frames[1:]:
        aligned = aligned.merge(frm, on=["acyl_chlorides", "amines", "y_true"], how="inner")
    pred_cols = [c for c in aligned.columns if c.startswith("pred_")]
    y_true = aligned["y_true"].to_numpy()
    X = np.column_stack([aligned[c].to_numpy() for c in pred_cols])
    return aligned, y_true, X, pred_cols


def main() -> None:
    with open(METRICS_JSON, "r") as f:
        m = json.load(f)
    w = np.asarray(m["nnls"]["weights"], dtype=float)
    b = float(m["nnls"]["intercept"])

    out_frames: List[pd.DataFrame] = []
    for split in ("TRAIN", "TEST"):
        aligned, y_true, X, cols = align_split(MEMBER_PATHS, split)
        y_pred = X.dot(w) + b
        # UQ
        std_unw = X.std(axis=1)
        w_norm = w / (w.sum() if w.sum() > 0 else 1.0)
        mean_w = (X * w_norm[None, :]).sum(axis=1)
        var_w = (w_norm[None, :] * (X - mean_w[:, None]) ** 2).sum(axis=1)
        std_w = np.sqrt(var_w)
        df = aligned[["acyl_chlorides", "amines"]].copy()
        df["split"] = split
        df["y_true"] = y_true
        df["pred_nnls"] = y_pred
        df["std_unweighted"] = std_unw
        df["std_weighted_nnls"] = std_w
        out_frames.append(df)

    full = pd.concat(out_frames, axis=0, ignore_index=True)
    full.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} with {len(full)} rows")


if __name__ == "__main__":
    main()





