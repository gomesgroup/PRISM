import os
import argparse
from pathlib import Path
import json

# Restrict to GPUs 0 and 1
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import polars as pl
import numpy as np
from typing import Dict, Any, List

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def read_dataset(dataset_csv: Path) -> pl.DataFrame:
    df = pl.read_csv(dataset_csv)
    # Create ln NMR if available
    if "NMR_rate" in df.columns:
        df = df.with_columns(
            pl.col("NMR_rate").map_elements(function=lambda x: float(np.log(x)) if x is not None and x > 0 else None, return_dtype=pl.Float64).alias("NMR_lnk")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("NMR_lnk"))
    return df


def build_feature_matrix(df: pl.DataFrame) -> pl.DataFrame:
    features = []
    # Start with reaction energy features if present
    for c in ["barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B"]:
        if c in df.columns:
            features.append(c)

    # Add a small subset of descriptor columns if present (prefixed from prior scripts)
    desc_candidates = [c for c in df.columns if (c.startswith("acid_") or c.startswith("amine_"))]
    features.extend(desc_candidates[:20])

    # Fallback to use HTE_lnk and pred_nnls as auxiliary features
    for c in ["HTE_lnk", "pred_nnls"]:
        if c in df.columns:
            features.append(c)

    # Dedup and filter
    seen = set()
    uniq = []
    for f in features:
        if f in df.columns and f not in seen:
            uniq.append(f)
            seen.add(f)

    if not uniq:
        raise ValueError("No features found for co-kriging model")

    return df.select(uniq).with_columns([pl.col(c).cast(dtype=pl.Float64, strict=False).fill_null(0.0) for c in uniq])


def prepare_joined_table(dataset_csv: Path, predictions_csv: Path) -> pl.DataFrame:
    ds = read_dataset(dataset_csv)
    preds = pl.read_csv(predictions_csv)
    # Normalize std column name
    if "std_weighted_nnls" in preds.columns:
        preds = preds.rename({"std_weighted_nnls": "std_pred"})
    elif "std_unweighted" in preds.columns:
        preds = preds.rename({"std_unweighted": "std_pred"})

    ds_idx = ds.with_row_index(name="row_id")
    pr_idx = preds.with_row_index(name="row_id")

    keep_cols = [c for c in ["row_id", "pred_nnls", "std_pred", "split"] if c in pr_idx.columns]
    pr_small = pr_idx.select(keep_cols)

    merged = ds_idx.join(pr_small, on="row_id", how="left")
    return merged


def fit_rho_closed_form(y_lf: np.ndarray, y_hf: np.ndarray) -> float:
    # rho = argmin ||y_hf - rho*y_lf||^2 => (y_lf^T y_hf)/(y_lf^T y_lf)
    denom = float(np.dot(y_lf, y_lf))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(y_lf, y_hf) / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if (np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8) else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson}


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Co-Kriging: NMR_lnk = rho * HTE_lnk + GP_delta(X)")
    parser.add_argument("--dataset-csv", type=str, default="data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv")
    parser.add_argument("--predictions-csv", type=str, default="results/ensemble_all6_predictions_full.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-prefix", type=str, default="cokriging_mf")
    args = parser.parse_args()

    merged = prepare_joined_table(Path(args.dataset_csv), Path(args.predictions_csv))

    # Attach external features if available
    # Reaction energies
    try:
        rxn = pl.read_csv(Path("data/reaction_energies/reaction_TSB_w_aimnet2.csv"))
        if "acid_chlorides" in rxn.columns:
            rxn = rxn.rename({"acid_chlorides": "acyl_chlorides"})
        keep = [c for c in ["acyl_chlorides", "amines", "barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B"] if c in rxn.columns]
        merged = merged.join(rxn.select(keep), on=["acyl_chlorides", "amines"], how="left")
    except (FileNotFoundError, pl.ComputeError):
        pass

    # Descriptors (prefix to avoid collisions)
    try:
        acid = pl.read_csv(Path("data/features/descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv"))
        amine = pl.read_csv(Path("data/features/descriptors_amines_morfeus_addn_w_xtb.csv"))
        acid_num = [c for c in acid.columns if c not in ["acyl_chlorides", "class", "smiles", "name"] and not c.startswith("has_")][:10]
        amine_num = [c for c in amine.columns if c not in ["amines", "class", "smiles", "name"] and not c.startswith("has_")][:10]
        acid_map = {c: f"acid_{c}" for c in acid_num}
        amine_map = {c: f"amine_{c}" for c in amine_num}
        merged = merged.join(acid.select(["acyl_chlorides", *acid_num]).rename(acid_map), on="acyl_chlorides", how="left")
        merged = merged.join(amine.select(["amines", *amine_num]).rename(amine_map), on="amines", how="left")
    except (FileNotFoundError, pl.ComputeError):
        pass

    # Keep only HF rows for CV
    hf_df = merged.filter(merged["NMR_lnk"].is_not_null())

    # Features and targets
    X_df = build_feature_matrix(hf_df)
    X = X_df.to_numpy()
    y_hf = hf_df["NMR_lnk"].to_numpy()
    y_lf = hf_df["HTE_lnk"].to_numpy() if "HTE_lnk" in hf_df.columns else hf_df["pred_nnls"].to_numpy()

    # Scalars for GP inputs
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    # Out-of-fold containers for HF rows
    oof_pred = np.full(shape=y_hf.shape, fill_value=np.nan, dtype=float)
    oof_std = np.full(shape=y_hf.shape, fill_value=np.nan, dtype=float)

    # Kernel for delta GP
    base_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_scaled.shape[1]), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))

    for train_idx, test_idx in kf.split(X_scaled):
        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
        y_lf_tr, y_lf_te = y_lf[train_idx], y_lf[test_idx]
        y_hf_tr, y_hf_te = y_hf[train_idx], y_hf[test_idx]

        # Fit rho on training HF points using measured LF values
        rho = fit_rho_closed_form(y_lf_tr, y_hf_tr)

        # Train GP on delta residuals
        delta_tr = y_hf_tr - rho * y_lf_tr
        gp = GaussianProcessRegressor(kernel=base_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)
        gp.fit(X_tr, delta_tr)

        # Predict on test
        pred_out = gp.predict(X_te, return_std=True)
        if isinstance(pred_out, tuple):
            # Some sklearn versions may return (mean, std) or (mean, std, cov)
            delta_pred_te = pred_out[0]
            std_te = pred_out[1] if len(pred_out) > 1 else np.zeros_like(delta_pred_te)
        else:
            delta_pred_te = pred_out
            std_te = np.zeros_like(delta_pred_te)
        y_pred_te = rho * y_lf_te + delta_pred_te

        y_true_all.extend(y_hf_te.tolist())
        y_pred_all.extend(y_pred_te.tolist())
        # Store OOF
        oof_pred[test_idx] = y_pred_te
        oof_std[test_idx] = std_te

    y_true_all_np = np.asarray(y_true_all, dtype=float)
    y_pred_all_np = np.asarray(y_pred_all, dtype=float)
    metrics_cv = compute_metrics(y_true_all_np, y_pred_all_np)

    # Fit final model on all HF data
    rho_all = fit_rho_closed_form(y_lf, y_hf)
    gp_all = GaussianProcessRegressor(kernel=base_kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42)
    gp_all.fit(X_scaled, y_hf - rho_all * y_lf)

    # Save predictions for HF set
    pred_all_out = gp_all.predict(X_scaled, return_std=True)
    if isinstance(pred_all_out, tuple):
        delta_all = pred_all_out[0]
        std_all = pred_all_out[1] if len(pred_all_out) > 1 else np.zeros_like(y_hf)
    else:
        delta_all = pred_all_out
        std_all = np.zeros_like(y_hf)
    y_pred_all_final = rho_all * y_lf + delta_all

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cols = [c for c in ["acyl_chlorides", "amines", "split"] if c in hf_df.columns]
    preds_df = hf_df.select(base_cols) if base_cols else pl.DataFrame({})
    preds_df = preds_df.with_columns(
        pl.Series("y_true_nmr_lnk", y_hf),
        pl.Series("y_pred_nmr_lnk", y_pred_all_final),
        pl.Series("y_std_delta", std_all),
        pl.lit(rho_all).alias("rho")
    )
    preds_path = out_dir / f"{args.output_prefix}_predictions_nmr.csv"
    preds_df.write_csv(preds_path)

    # Split-wise metrics
    metrics: Dict[str, Any] = {"cv": metrics_cv, "rho": rho_all}
    if "split" in hf_df.columns:
        for sp in hf_df["split"].unique().drop_nulls().to_list():
            mask = (hf_df["split"] == sp).to_numpy()
            if mask.any():
                metrics[f"split_{sp}"] = compute_metrics(y_hf[mask], y_pred_all_final[mask])

    # Overall
    metrics["overall_hf_insample"] = compute_metrics(y_hf, y_pred_all_final)
    # Save OOF predictions if complete
    if not np.isnan(oof_pred).any():
        metrics["overall_hf_oof"] = compute_metrics(y_hf, oof_pred)
        preds_oof_df = hf_df.select([c for c in ["acyl_chlorides", "amines", "split"] if c in hf_df.columns]) if any(c in hf_df.columns for c in ["acyl_chlorides","amines","split"]) else pl.DataFrame({})
        preds_oof_df = preds_oof_df.with_columns(
            pl.Series("y_true_nmr_lnk", y_hf),
            pl.Series("y_pred_nmr_lnk", oof_pred),
            pl.Series("y_std_delta", oof_std),
        )
        preds_oof_path = out_dir / f"{args.output_prefix}_predictions_nmr_oof.csv"
        preds_oof_df.write_csv(preds_oof_path)

    metrics_path = out_dir / f"{args.output_prefix}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to: {preds_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()


