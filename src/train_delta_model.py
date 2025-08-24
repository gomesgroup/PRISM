import os
import argparse
from pathlib import Path
import json

# Enforce GPU visibility to devices 0 and 1 only, if any downstream library uses CUDA
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import polars as pl
import numpy as np
from typing import Dict, List, Any

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold


def read_predictions(pred_csv: Path) -> pl.DataFrame:
    """
    Read ensemble predictions CSV produced for the parity plot.

    Expected columns:
    - pred_nnls: predicted ln k from big-FFN ensemble
    - std_weighted_nnls or std_unweighted: per-row ensemble std (optional)
    - split: TRAIN/VAL/TEST or similar
    """
    df = pl.read_csv(pred_csv)
    # Normalize std column name
    if "std_weighted_nnls" in df.columns:
        df = df.rename({"std_weighted_nnls": "std_pred"})
    elif "std_unweighted" in df.columns:
        df = df.rename({"std_unweighted": "std_pred"})
    else:
        df = df.with_columns(pl.lit(None).alias("std_pred"))
    return df


def read_dataset(dataset_csv: Path) -> pl.DataFrame:
    """
    Read the primary dataset CSV used for HTE with optional NMR column.

    Required columns:
    - acyl_chlorides, amines
    - HTE_lnk
    - NMR_rate (may contain nulls)
    - test splits (train/test labeling), if present
    - amine_smiles, acid_smiles
    """
    df = pl.read_csv(dataset_csv)
    # Ensure expected columns exist
    for col in ["acyl_chlorides", "amines", "HTE_lnk"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {dataset_csv}")
    # Create ln NMR if available
    if "NMR_rate" in df.columns:
        df = df.with_columns(
            pl.col("NMR_rate").map_elements(
                lambda x: float(np.log(x)) if x is not None and x > 0 else None,
                return_dtype=pl.Float64,
            ).alias("NMR_lnk")
        )
    else:
        df = df.with_columns(pl.lit(None).alias("NMR_lnk"))
    return df


def join_predictions_with_dataset(ds: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """
    Join predictions to dataset by row index using a synthetic row_id to guarantee alignment.
    Assumes predictions were generated over the same dataset in the same order.
    """
    ds_idx = ds.with_row_index(name="row_id")
    pr_idx = preds.with_row_index(name="row_id")

    # Select only needed prediction columns
    pr_idx = pr_idx.select(["row_id", "pred_nnls", "std_pred", "split"]) if "split" in pr_idx.columns else pr_idx.select(["row_id", "pred_nnls", "std_pred"]) 

    merged = ds_idx.join(pr_idx, on="row_id", how="left")
    return merged


def build_feature_matrix(df: pl.DataFrame, include_reaction_features: bool = True) -> pl.DataFrame:
    """
    Construct features for delta model.
    Core features:
    - pred_nnls (required)
    - std_pred (optional)

    Optional:
    - a handful of reaction energy features if available
    - a small subset of molecular descriptors if available
    """
    features = ["pred_nnls"]
    if "std_pred" in df.columns:
        features.append("std_pred")

    augmented = df

    if include_reaction_features:
        # Attempt to load reaction energies and basic descriptors
        data_dir = Path("data")
        rxn_path = data_dir / "reaction_energies" / "reaction_TSB_w_aimnet2.csv"
        acid_desc_path = data_dir / "features" / "descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv"
        amine_desc_path = data_dir / "features" / "descriptors_amines_morfeus_addn_w_xtb.csv"

        try:
            rxn = pl.read_csv(rxn_path)
            # Normalize join keys
            if "acid_chlorides" in rxn.columns:
                rxn = rxn.rename({"acid_chlorides": "acyl_chlorides"})
            rxn_keep = [c for c in ["acyl_chlorides", "amines", "barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B"] if c in rxn.columns]
            augmented = augmented.join(rxn.select(rxn_keep), on=["acyl_chlorides", "amines"], how="left")
            for c in ["barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B"]:
                if c in augmented.columns:
                    features.append(c)
        except FileNotFoundError:
            pass
        except pl.exceptions.ComputeError:
            pass

        # Add a small subset of numeric descriptors to avoid high dimensionality
        try:
            acid = pl.read_csv(acid_desc_path)
            amine = pl.read_csv(amine_desc_path)
            # Keep numeric columns excluding identifiers
            acid_num = [c for c in acid.columns if c not in ["acyl_chlorides", "class", "smiles", "name"] and not c.startswith("has_")]
            amine_num = [c for c in amine.columns if c not in ["amines", "class", "smiles", "name"] and not c.startswith("has_")]
            # Take the first 10 to keep model simple
            acid_num_sel = acid_num[:10]
            amine_num_sel = amine_num[:10]

            # Prefix to avoid collisions
            acid_map = {c: f"acid_{c}" for c in acid_num_sel}
            amine_map = {c: f"amine_{c}" for c in amine_num_sel}

            acid_pref = acid.select(["acyl_chlorides", *acid_num_sel]).rename(acid_map)
            amine_pref = amine.select(["amines", *amine_num_sel]).rename(amine_map)

            augmented = augmented.join(acid_pref, on="acyl_chlorides", how="left")
            augmented = augmented.join(amine_pref, on="amines", how="left")

            features.extend(acid_map.values())
            features.extend(amine_map.values())
        except FileNotFoundError:
            pass
        except pl.exceptions.ComputeError:
            pass

    # Ensure features exist in df; fill nulls with 0
    # Deduplicate feature names to avoid duplicate expressions
    seen = set()
    uniq_features = []
    for f in features:
        if f in augmented.columns and f not in seen:
            uniq_features.append(f)
            seen.add(f)
    augmented = augmented.with_columns([pl.col(f).cast(dtype=pl.Float64, strict=False).fill_null(0.0) for f in uniq_features])

    return augmented.select(uniq_features)


def compute_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    # Pearson correlation
    if y_true_arr.size > 1 and np.std(y_pred_arr) > 1e-8:
        pearson = float(np.corrcoef(y_true_arr, y_pred_arr)[0, 1])
    else:
        pearson = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson}


def cross_validate_delta(X: Any, y_delta: Any, base_pred: Any, n_splits: int = 5) -> Dict[str, float]:
    """
    Cross-validate delta model (ElasticNetCV) and report metrics on reconstructed NMR ln k.
    """
    X = np.asarray(X, dtype=float)
    y_delta = np.asarray(y_delta, dtype=float)
    base_pred = np.asarray(base_pred, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_delta[train_idx], y_delta[test_idx]
        base_te = base_pred[test_idx]

        model = ElasticNetCV(l1_ratio=[0.0, 0.25, 0.5, 0.75, 1.0], alphas=np.logspace(-4, 2, 60), cv=3, random_state=42, max_iter=10000)
        model.fit(X_tr, y_tr)
        delta_pred = model.predict(X_te)
        nmr_pred = base_te + delta_pred

        y_true_all.extend((base_te + y_te).tolist())
        y_pred_all.extend(nmr_pred.tolist())

    y_true_all_np = np.asarray(y_true_all, dtype=float)
    y_pred_all_np = np.asarray(y_pred_all, dtype=float)

    return compute_metrics(y_true_all_np, y_pred_all_np)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a delta-ML model to map CheMeleon ensemble ln k to NMR ln k.")
    parser.add_argument("--dataset-csv", type=str, default="data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv", help="Path to the primary dataset CSV")
    parser.add_argument("--predictions-csv", type=str, default="results/ensemble_all6_predictions_full.csv", help="Path to the ensemble predictions CSV (pred_nnls)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds on NMR subset")
    parser.add_argument("--include-reaction-features", action="store_true", help="Include reaction energy and descriptor features in delta model")
    parser.add_argument("--output-prefix", type=str, default="delta_ml", help="Prefix for saved outputs in results/")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    predictions_csv = Path(args.predictions_csv)

    # Read data
    ds = read_dataset(dataset_csv)
    preds = read_predictions(predictions_csv)

    merged = join_predictions_with_dataset(ds, preds)

    # Keep rows with NMR measurements
    nmr_mask = merged["NMR_lnk"].is_not_null()
    df_nmr = merged.filter(nmr_mask)

    # Build features and targets
    X_df = build_feature_matrix(merged if args.include_reaction_features else merged.select([c for c in merged.columns if c in ["pred_nnls", "std_pred"]]))

    # Align arrays
    X_all = X_df.to_numpy()
    base_pred_all = merged["pred_nnls"].to_numpy()

    # NMR subset
    X_nmr = X_df.filter(nmr_mask).to_numpy()
    base_pred_nmr = df_nmr["pred_nnls"].to_numpy()
    y_nmr = df_nmr["NMR_lnk"].to_numpy()

    # Delta target
    y_delta = y_nmr - base_pred_nmr

    # Cross-validate
    metrics_cv = cross_validate_delta(X_nmr, y_delta, base_pred_nmr, n_splits=args.n_splits)

    # Fit final model on all NMR data
    final_model = ElasticNetCV(l1_ratio=[0.0, 0.25, 0.5, 0.75, 1.0], alphas=np.logspace(-4, 2, 60), cv=5, random_state=42, max_iter=10000)
    final_model.fit(np.asarray(X_nmr, dtype=float), np.asarray(y_delta, dtype=float))

    # Predict for all rows
    delta_pred_all = final_model.predict(np.asarray(X_all, dtype=float))
    nmr_pred_all = np.asarray(base_pred_all, dtype=float) + delta_pred_all

    # Prepare outputs
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for rows where NMR exists (evaluation set)
    eval_df = pl.DataFrame({
        "acyl_chlorides": merged["acyl_chlorides"],
        "amines": merged["amines"],
        "split": merged["split"] if "split" in merged.columns else pl.lit(None),
        "pred_base": base_pred_all,
        "pred_delta": delta_pred_all,
        "pred_nmr": nmr_pred_all,
        "true_nmr": merged["NMR_lnk"],
    })

    eval_path = out_dir / f"{args.output_prefix}_predictions_nmr.csv"
    eval_df.write_csv(eval_path)

    # Compute split-wise metrics on available NMR rows
    metrics: Dict[str, Dict[str, float]] = {"cv": metrics_cv}

    if "split" in merged.columns:
        for sp in merged["split"].unique().drop_nulls().to_list():
            mask_sp = (merged["split"] == sp) & merged["NMR_lnk"].is_not_null()
            if mask_sp.any():
                y_true = merged.filter(mask_sp)["NMR_lnk"].to_numpy()
                y_pred = pl.Series(nmr_pred_all).filter(mask_sp).to_numpy()
                metrics[f"split_{sp}"] = compute_metrics(y_true, y_pred)

    # Overall metrics on NMR subset
    mask_np = merged["NMR_lnk"].is_not_null().to_numpy()
    metrics["overall_nmr_subset"] = compute_metrics(y_nmr, nmr_pred_all[mask_np])

    metrics_path = out_dir / f"{args.output_prefix}_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # Save a small README with GPU-constrained chemprop command
    readme_path = out_dir / f"{args.output_prefix}_HOWTO.txt"
    with readme_path.open("w") as f:
        f.write(
            """
Delta-ML pipeline outputs generated.

If you need to regenerate CheMeleon ensemble predictions using ONLY GPUs 0 and 1:

  1) Activate chemprop environment
     eval "$(conda shell.zsh hook)" && conda activate chemprop-v2

  2) Run prediction constrained to GPUs 0 and 1 (uses one device per run)
     CUDA_VISIBLE_DEVICES=0,1 chemprop predict \
       -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
       -s amine_smiles acid_smiles \
       --model-paths \
         runs/chemeleon_lnk_bigffn_ens5/model_0/best.pt \
         runs/chemeleon_lnk_bigffn_ens5/model_1/best.pt \
         runs/chemeleon_lnk_bigffn_ens5/model_2/best.pt \
         runs/chemeleon_lnk_bigffn_ens5/model_3/best.pt \
         runs/chemeleon_lnk_bigffn_ens5/model_4/best.pt \
       -o results/bigffn_ens5_all_predictions.csv \
       --accelerator gpu --devices 1

Then rebuild the plotting CSV (if needed) and rerun:

     python src/train_delta_model.py --include-reaction-features

"""
        )

    print(f"Saved predictions to: {eval_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
