#!/usr/bin/env python3
"""
Compute per-holdout (TEST1/TEST2) metrics for equal-mean, Ridge, and NNLS ensembles
using existing predictions and the split labels from the combined DataFrame.

Outputs:
- results/ensemble_summary_by_holdout_cpv2_S97.csv
- results/ensemble_summary_by_holdout_cpv2_S97.json

Also prints a compact table to stdout.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


RESULTS_DIR = "results"
COMBINED_PATH = os.path.join(RESULTS_DIR, "combined_df_each_8_optuna_cpv2_gpu0.csv")
EQUAL_MEAN_PATH = os.path.join(RESULTS_DIR, "ensemble_predictions_cpv2_S97.csv")
RIDGE_PATH = os.path.join(RESULTS_DIR, "ensemble_weighted_cpv2_S97_ridge.csv")
NNLS_PATH = os.path.join(RESULTS_DIR, "ensemble_weighted_cpv2_S97_nnls.csv")

OUT_CSV = os.path.join(RESULTS_DIR, "ensemble_summary_by_holdout_cpv2_S97.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "ensemble_summary_by_holdout_cpv2_S97.json")


def _ensure_int_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("acyl_chlorides", "amines"):
        if col in df.columns:
            # Robust cast via float → int to accommodate any accidental decimal strings
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float).astype("Int64").astype(int)
    return df


def _manual_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    n = y_true.shape[0]
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    y_bar = float(np.mean(y_true))
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_bar) ** 2))
    r2 = 0.0 if sst == 0.0 else 1.0 - (sse / sst)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(sse / n))
    return r2, mae, rmse


def _coverage(y_true: pd.Series, lower: Optional[pd.Series], upper: Optional[pd.Series]) -> Optional[float]:
    if lower is None or upper is None:
        return None
    valid = pd.concat([y_true, lower, upper], axis=1).dropna()
    if valid.empty:
        return None
    yt = valid.iloc[:, 0].to_numpy()
    lo = valid.iloc[:, 1].to_numpy()
    hi = valid.iloc[:, 2].to_numpy()
    return float(np.mean((yt >= lo) & (yt <= hi)))


def _load_splits() -> pd.DataFrame:
    comb = pd.read_csv(COMBINED_PATH, usecols=["acyl_chlorides", "amines", "test splits"])\
        .rename(columns={"test splits": "split"})
    comb = _ensure_int_id_cols(comb)
    return comb


def _load_equal_mean() -> pd.DataFrame:
    df = pd.read_csv(EQUAL_MEAN_PATH)
    # Normalize schema to y_pred and interval columns
    if "y_pred" not in df.columns and "y_pred_mean" in df.columns:
        df = df.rename(columns={"y_pred_mean": "y_pred"})
    needed = ["acyl_chlorides", "amines", "y_true", "y_pred", "lower90", "upper90", "lower95", "upper95"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Equal-mean CSV missing columns: {missing}")
    df = _ensure_int_id_cols(df)
    return df[needed]


def _load_weighted(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["acyl_chlorides", "amines", "y_true", "y_pred", "lower90", "upper90", "lower95", "upper95"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Weighted CSV {os.path.basename(path)} missing columns: {missing}")
    df = _ensure_int_id_cols(df)
    return df[needed]


def _eval_by_holdout(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for holdout in ("TEST1", "TEST2"):
        sub = df[df["split"] == holdout]
        if sub.empty:
            continue
        y = sub["y_true"].to_numpy()
        yp = sub["y_pred"].to_numpy()
        r2, mae, rmse = _manual_regression_metrics(y, yp)
        cov90 = _coverage(sub["y_true"], sub.get("lower90"), sub.get("upper90"))
        cov95 = _coverage(sub["y_true"], sub.get("lower95"), sub.get("upper95"))
        out[holdout] = dict(
            test_r2=r2,
            test_mae=mae,
            test_rmse=rmse,
            coverage90_test=cov90,
            coverage95_test=cov95,
            n_test=int(sub.shape[0]),
        )
    return out


def main() -> None:
    if not os.path.exists(COMBINED_PATH):
        raise FileNotFoundError(f"Missing combined splits CSV: {COMBINED_PATH}")
    if not os.path.exists(EQUAL_MEAN_PATH):
        raise FileNotFoundError(f"Missing equal-mean predictions CSV: {EQUAL_MEAN_PATH}")
    if not os.path.exists(RIDGE_PATH) or not os.path.exists(NNLS_PATH):
        raise FileNotFoundError("Missing weighted predictions CSVs; expected ridge and nnls.")

    comb = _load_splits()
    eq = _load_equal_mean().merge(comb, on=["acyl_chlorides", "amines"], how="left")
    ridge = _load_weighted(RIDGE_PATH).merge(comb, on=["acyl_chlorides", "amines"], how="left")
    nnls = _load_weighted(NNLS_PATH).merge(comb, on=["acyl_chlorides", "amines"], how="left")

    results_rows: List[Dict[str, object]] = []
    for name, df in (("equal_mean", eq), ("ridge", ridge), ("nnls", nnls)):
        by = _eval_by_holdout(df)
        for holdout, m in by.items():
            results_rows.append({
                "method": name,
                "holdout": holdout,
                **m,
            })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if results_rows:
        pd.DataFrame(results_rows).to_csv(OUT_CSV, index=False)
        with open(OUT_JSON, "w") as f:
            json.dump(results_rows, f, indent=2)
        print(f"Saved {OUT_CSV}")
        print(f"Saved {OUT_JSON}")
        # Pretty print table
        df = pd.DataFrame(results_rows)
        cols = ["method", "holdout", "test_r2", "test_mae", "test_rmse", "coverage90_test", "coverage95_test", "n_test"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
    else:
        print("No TEST1/TEST2 rows found in predictions.")


if __name__ == "__main__":
    main()


