#!/usr/bin/env python3
"""
Compute an ensemble over cpv2 runs (seed 97), align TEST and VAL splits,
report ensemble metrics, and save split-conformal intervals.
"""

from __future__ import annotations

import json
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def align_split(paths: List[str], split_label: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    aligned_frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["split"] == split_label].copy()
        df = df[["acyl_chlorides", "amines", "y_true", "y_pred"]]
        df = df.rename(columns={"y_pred": f"pred_{os.path.basename(path)}"})
        aligned_frames.append(df)

    if len(aligned_frames) == 0:
        raise ValueError(f"No frames for split {split_label}")

    aligned = aligned_frames[0]
    for frm in aligned_frames[1:]:
        aligned = aligned.merge(frm, on=["acyl_chlorides", "amines", "y_true"], how="inner")

    pred_cols = [c for c in aligned.columns if c.startswith("pred_")]
    y_true = aligned["y_true"].to_numpy()
    preds = np.column_stack([aligned[c].to_numpy() for c in pred_cols])
    return aligned, y_true, preds


def main() -> None:
    paths = sorted(glob.glob("results/predictions_each_8_optuna_cpv2_gpu*.csv"))
    if not paths:
        raise SystemExit("No prediction CSVs found under results/")

    print("Using prediction files:\n - " + "\n - ".join(paths))

    aligned_test, y_test, preds_test = align_split(paths, "TEST")

    # Prefer VAL for calibration; fallback to TRAIN if VAL absent
    try:
        aligned_val, y_val, preds_val = align_split(paths, "VAL")
        if len(y_val) == 0:
            raise ValueError("VAL empty")
        calibration_split = "VAL"
    except Exception:
        aligned_val, y_val, preds_val = align_split(paths, "TRAIN")
        calibration_split = "TRAIN"
        print("VAL not found/empty; using TRAIN for conformal calibration")

    # Ensemble predictions
    mean_test = preds_test.mean(axis=1)
    std_test = preds_test.std(axis=1)
    mean_val = preds_val.mean(axis=1)

    # Metrics
    test_r2 = float(r2_score(y_test, mean_test))
    test_mae = float(mean_absolute_error(y_test, mean_test))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, mean_test)))

    # Split-conformal (constant width) from calibration residuals
    abs_res_val = np.abs(y_val - mean_val)
    q90 = float(np.quantile(abs_res_val, 0.90))
    q95 = float(np.quantile(abs_res_val, 0.95))

    lower90 = mean_test - q90
    upper90 = mean_test + q90
    lower95 = mean_test - q95
    upper95 = mean_test + q95

    coverage90 = float(np.mean((y_test >= lower90) & (y_test <= upper90)))
    coverage95 = float(np.mean((y_test >= lower95) & (y_test <= upper95)))

    # Save outputs
    os.makedirs("results", exist_ok=True)
    suffix = "_cpv2_S97"

    out_df = aligned_test[["acyl_chlorides", "amines", "y_true"]].copy()
    out_df["y_pred_mean"] = mean_test
    out_df["y_pred_std"] = std_test
    out_df["lower90"] = lower90
    out_df["upper90"] = upper90
    out_df["lower95"] = lower95
    out_df["upper95"] = upper95

    out_csv = f"results/ensemble_predictions{suffix}.csv"
    out_df.to_csv(out_csv, index=False)

    metrics = {
        "n_test": int(len(y_test)),
        "n_val": int(len(y_val)),
        "calibration_split": calibration_split,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "q90": q90,
        "q95": q95,
        "coverage90_test": coverage90,
        "coverage95_test": coverage95,
        "paths": paths,
        "output_csv": out_csv,
    }
    out_json = f"results/ensemble_metrics{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved ensemble:", out_csv)
    print("Saved metrics:", out_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


