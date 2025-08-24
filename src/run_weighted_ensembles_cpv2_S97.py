#!/usr/bin/env python3
"""
Weighted ensembles (Ridge and NNLS) for cpv2 runs (seed 97).
Align TRAIN/VAL/TEST predictions across runs, fit weights on TRAIN,
calibrate split-conformal on VAL, evaluate on TEST, and save metrics/plots.
"""

from __future__ import annotations

import json
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional


def align_split(paths: List[str], split_label: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["split"] == split_label].copy()
        df = df[["acyl_chlorides", "amines", "y_true", "y_pred"]]
        df = df.rename(columns={"y_pred": f"pred_{os.path.basename(path)}"})
        frames.append(df)

    if len(frames) == 0:
        raise ValueError(f"No frames for split {split_label}")

    aligned = frames[0]
    for frm in frames[1:]:
        aligned = aligned.merge(frm, on=["acyl_chlorides", "amines", "y_true"], how="inner")

    pred_cols = [c for c in aligned.columns if c.startswith("pred_")]
    y_true = aligned["y_true"].to_numpy()
    preds = np.column_stack([aligned[c].to_numpy() for c in pred_cols])
    return aligned, y_true, preds, pred_cols


def fit_ridge_weights(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    model = RidgeCV(alphas=np.logspace(-6, 3, 20), fit_intercept=True)
    model.fit(X_train, y_train)
    coef = model.coef_.astype(float)
    intercept = float(model.intercept_)
    info = {"alpha": float(model.alpha_)}
    return coef, intercept, info


def fit_nnls_weights(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
    try:
        from scipy.optimize import nnls  # type: ignore
    except Exception:
        # Fallback: non-negative least-squares via simple clipping of OLS weights
        w, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        w = np.clip(w, 0.0, None)
        # Derive intercept to match means
        intercept = float(y_train.mean() - X_train.mean(axis=0).dot(w))
        return w.astype(float), intercept, {"solver": "lstsq_clipped"}

    w, _ = nnls(X_train, y_train)
    # Derive intercept to match means (nnls has no intercept)
    intercept = float(y_train.mean() - X_train.mean(axis=0).dot(w))
    return w.astype(float), intercept, {"solver": "nnls"}


def compute_weighted_predictions(
    X: np.ndarray, weights: np.ndarray, intercept: float
) -> np.ndarray:
    return X.dot(weights) + intercept


def compute_calibration(abs_residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                        alphas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    coverages = []
    widths = []
    for a in alphas:
        q = np.quantile(abs_residuals, a)
        lower = y_pred - q
        upper = y_pred + q
        cov = np.mean((y_true >= lower) & (y_true <= upper))
        coverages.append(cov)
        widths.append(2.0 * q)
    return np.asarray(coverages), np.asarray(widths)


def save_predictions(out_path: str, aligned: pd.DataFrame, y_pred: np.ndarray,
                     member_preds: np.ndarray, q90: float, q95: float,
                     weights: np.ndarray | None = None) -> None:
    df = aligned[["acyl_chlorides", "amines", "y_true"]].copy()
    df["y_pred"] = y_pred
    # Unweighted std across members
    df["y_std_unweighted"] = member_preds.std(axis=1)
    # Weighted std (normalized weights)
    if weights is not None:
        w = weights.astype(float)
        if w.sum() > 0:
            w = w / w.sum()
            mean = (member_preds * w[None, :]).sum(axis=1)
            var = (w[None, :] * (member_preds - mean[:, None]) ** 2).sum(axis=1)
            df["y_std_weighted"] = np.sqrt(var)
    df["lower90"] = y_pred - q90
    df["upper90"] = y_pred + q90
    df["lower95"] = y_pred - q95
    df["upper95"] = y_pred + q95
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    paths = sorted(glob.glob("results/predictions_each_8_optuna_cpv2_gpu*.csv"))
    if not paths:
        raise SystemExit("No prediction CSVs found under results/")
    print("Using prediction files:\n - " + "\n - ".join(paths))

    # Align across splits
    tr_aligned, y_tr, X_tr, _ = align_split(paths, "TRAIN")
    va_aligned, y_va, X_va, _ = align_split(paths, "VAL")
    te_aligned, y_te, X_te, _ = align_split(paths, "TEST")

    results: Dict[str, dict] = {}
    out_prefix = "results/ensemble_weighted_cpv2_S97"

    # Ridge
    w_ridge, b_ridge, ridge_info = fit_ridge_weights(X_tr, y_tr)
    y_va_ridge = compute_weighted_predictions(X_va, w_ridge, b_ridge)
    y_te_ridge = compute_weighted_predictions(X_te, w_ridge, b_ridge)
    # Conformal from VAL
    abs_res_val = np.abs(y_va - y_va_ridge)
    q90 = float(np.quantile(abs_res_val, 0.90))
    q95 = float(np.quantile(abs_res_val, 0.95))
    cov90 = float(np.mean((y_te >= y_te_ridge - q90) & (y_te <= y_te_ridge + q90)))
    cov95 = float(np.mean((y_te >= y_te_ridge - q95) & (y_te <= y_te_ridge + q95)))
    # Metrics
    r2 = float(r2_score(y_te, y_te_ridge))
    mae = float(mean_absolute_error(y_te, y_te_ridge))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_te_ridge)))
    # Save predictions
    save_predictions(out_prefix + "_ridge.csv", te_aligned, y_te_ridge, X_te, q90, q95, w_ridge)
    results["ridge"] = {
        "weights": w_ridge.tolist(),
        "intercept": b_ridge,
        "alpha": ridge_info.get("alpha"),
        "test_r2": r2,
        "test_mae": mae,
        "test_rmse": rmse,
        "coverage90_test": cov90,
        "coverage95_test": cov95,
        "q90": q90,
        "q95": q95,
    }

    # NNLS
    w_nnls, b_nnls, nnls_info = fit_nnls_weights(X_tr, y_tr)
    y_va_nnls = compute_weighted_predictions(X_va, w_nnls, b_nnls)
    y_te_nnls = compute_weighted_predictions(X_te, w_nnls, b_nnls)
    abs_res_val = np.abs(y_va - y_va_nnls)
    q90 = float(np.quantile(abs_res_val, 0.90))
    q95 = float(np.quantile(abs_res_val, 0.95))
    cov90 = float(np.mean((y_te >= y_te_nnls - q90) & (y_te <= y_te_nnls + q90)))
    cov95 = float(np.mean((y_te >= y_te_nnls - q95) & (y_te <= y_te_nnls + q95)))
    r2 = float(r2_score(y_te, y_te_nnls))
    mae = float(mean_absolute_error(y_te, y_te_nnls))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_te_nnls)))
    save_predictions(out_prefix + "_nnls.csv", te_aligned, y_te_nnls, X_te, q90, q95, w_nnls)
    results["nnls"] = {
        "weights": w_nnls.tolist(),
        "intercept": b_nnls,
        "solver": nnls_info.get("solver"),
        "test_r2": r2,
        "test_mae": mae,
        "test_rmse": rmse,
        "coverage90_test": cov90,
        "coverage95_test": cov95,
        "q90": q90,
        "q95": q95,
    }

    # Calibration curves for equal-mean, Ridge, NNLS
    y_va_mean = X_va.mean(axis=1)
    y_te_mean = X_te.mean(axis=1)
    alphas = np.linspace(0.5, 0.99, 50)

    eq_abs_res = np.abs(y_va - y_va_mean)
    eq_covs, eq_widths = compute_calibration(eq_abs_res, y_te, y_te_mean, alphas)
    results["equal_mean_calibration_curve"] = {
        "alphas": alphas.tolist(),
        "coverages": eq_covs.tolist(),
        "widths": eq_widths.tolist(),
    }

    ridge_abs_res = np.abs(y_va - y_va_ridge)
    ridge_covs, ridge_widths = compute_calibration(ridge_abs_res, y_te, y_te_ridge, alphas)
    results["ridge_calibration_curve"] = {
        "alphas": alphas.tolist(),
        "coverages": ridge_covs.tolist(),
        "widths": ridge_widths.tolist(),
    }

    nnls_abs_res = np.abs(y_va - y_va_nnls)
    nnls_covs, nnls_widths = compute_calibration(nnls_abs_res, y_te, y_te_nnls, alphas)
    results["nnls_calibration_curve"] = {
        "alphas": alphas.tolist(),
        "coverages": nnls_covs.tolist(),
        "widths": nnls_widths.tolist(),
    }

    # Plot calibration curves
    try:
        if plt is not None:
            os.makedirs("plots", exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(alphas, alphas, linestyle="--", color="gray", label="ideal")
            ax.plot(alphas, eq_covs, label="Equal-mean")
            ax.plot(alphas, ridge_covs, label="Ridge")
            ax.plot(alphas, nnls_covs, label="NNLS")
            ax.set_xlabel("Target quantile (alpha)")
            ax.set_ylabel("Empirical coverage on TEST")
            ax.set_title("Split-conformal calibration")
            ax.set_ylim(0.5, 1.01)
            ax.grid(True, alpha=0.3)
            ax.legend()
            out_png = "plots/calibration_weighted_cpv2_S97.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            results["calibration_plot"] = out_png
    except Exception:
        pass

    # Save metrics
    out_json = out_prefix + "_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics:", out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


