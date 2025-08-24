#!/usr/bin/env python3
"""
Evaluate external drug-scope holdout using a small ensemble trained on the original dataset.

Pipeline:
- Load base HTE dataset and generate pair-grouped 60/20/20 splits with seed 97
- Build curated-8 feature combined dataset (no reaction energies to avoid missing cols)
- Train 3 strong regressors on TRAIN: XGBoost (GPU if available), CatBoost (GPU if available), ExtraTrees (CPU)
- Calibrate split-conformal (absolute residual quantiles) on VAL
- Load external CSV new_data/combined_features_hte_rates_drug_scope.csv and build features for those IDs
- Predict per-model on external, compute equal-mean and per-sample std, add conformal 90/95 intervals
- Save predictions CSV and metrics JSON

Outputs:
- results/external_drug_scope_predictions_cpv2_S97.csv
- results/external_drug_scope_metrics_cpv2_S97.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Local imports
from data_processing import (
    load_hte_data,
    load_and_process_features,
)
from splits import generate_random_splits
from model_building import create_combined_dataset

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore


RESULTS_DIR = "results"
EXTERNAL_CSV = "new_data/combined_features_hte_rates_drug_scope.csv"
TARGET_COL = "HTE_lnk_corrected"

# Curated 8 features used in prior runs
CURATED_FEATURES = [
    'amine_class_1_mixture',
    'acyl_class_aromatic',
    'acyl_Charges_secondary_1',
    'amine_Charges_secondary_1',
    'acyl_pka_aHs_x_has_acidic_H',
    'amine_pka_basic',
    'acyl_BV_secondary_2',
    'amine_BV_secondary_avg',
]


def build_train_val_test() -> Tuple[pd.DataFrame, List[str]]:
    df = load_hte_data(analysis_type='hte_prediction', target_col=TARGET_COL)
    df = generate_random_splits(df, group_by='pair', ratios=(0.6, 0.2, 0.2), seed=97)
    acid_feats, amine_feats = load_and_process_features(df, target_col=TARGET_COL)
    combined_df, feat_cols = create_combined_dataset(
        acid_feats, amine_feats, df, selected_features=CURATED_FEATURES, save_df=False,
        rxn_features=None, add_interactions=True,
    )
    return combined_df, feat_cols


def fit_models(X_train: np.ndarray, y_train: np.ndarray) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = []
    # XGBoost
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=97,
            objective='reg:squarederror',
            n_jobs=0,
        )
        try:
            xgb.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
        except Exception:
            pass
        xgb.fit(X_train, y_train)
        models.append(("XGBoost", xgb))
    # CatBoost
    if CatBoostRegressor is not None:
        cat = CatBoostRegressor(
            iterations=1200,
            depth=8,
            learning_rate=0.04,
            loss_function='RMSE',
            random_state=97,
            verbose=False,
        )
        try:
            cat.set_params(task_type='GPU')
        except Exception:
            pass
        cat.fit(X_train, y_train)
        models.append(("CatBoost", cat))
    # Extra Trees (CPU)
    etr = ExtraTreesRegressor(n_estimators=800, random_state=97, n_jobs=-1)
    etr.fit(X_train, y_train)
    models.append(("ExtraTrees", etr))
    return models


def split_conformal_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    abs_res = np.abs(y_true - y_pred)
    return {
        'q90': float(np.quantile(abs_res, 0.90)),
        'q95': float(np.quantile(abs_res, 0.95)),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Train on original dataset
    combined_df, feat_cols = build_train_val_test()
    # Build splits masks
    train_mask = combined_df['test splits'] == 'TRAIN'
    val_mask = combined_df['test splits'] == 'VAL'

    X_cols = [c for c in feat_cols if c in combined_df.columns]
    # Add interaction features when available
    X_cols += [c for c in combined_df.columns if c.startswith('int_')]
    X_cols = list(dict.fromkeys(X_cols))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(combined_df.loc[train_mask, X_cols])
    y_train = combined_df.loc[train_mask, TARGET_COL].to_numpy()

    X_val = scaler.transform(combined_df.loc[val_mask, X_cols])
    y_val = combined_df.loc[val_mask, TARGET_COL].to_numpy()

    models = fit_models(X_train, y_train)

    # VAL predictions for conformal
    val_preds = []
    for _, m in models:
        vp = m.predict(X_val)
        val_preds.append(vp)
    val_preds = np.column_stack(val_preds)
    val_mean = np.mean(val_preds, axis=1)
    qs = split_conformal_quantiles(y_val, val_mean)

    # Load external and build features
    ext_df = pd.read_csv(EXTERNAL_CSV)
    # Ensure required id columns present
    if not {'acyl_chlorides', 'amines'}.issubset(ext_df.columns):
        raise RuntimeError("External CSV missing acyl_chlorides/amines columns")

    # We only need features; target may be absent
    # Synthesize minimal df with required columns for feature loaders
    stub = ext_df[['acyl_chlorides', 'amines']].copy()
    # Add dummy target col if missing to satisfy downstream code expectations
    if TARGET_COL not in stub.columns:
        stub[TARGET_COL] = 0.0

    acid_feats_ext, amine_feats_ext = load_and_process_features(stub, target_col=TARGET_COL)
    # Build combined with curated features only and interactions
    ext_comb, ext_feat_cols = create_combined_dataset(
        acid_feats_ext, amine_feats_ext, stub, selected_features=CURATED_FEATURES,
        save_df=False, rxn_features=None, add_interactions=True,
    )

    # Align columns to training columns
    ext_X = ext_comb.reindex(columns=X_cols, fill_value=0.0)
    ext_X_scaled = scaler.transform(ext_X)

    # Predict per-model
    ext_preds = []
    for _, m in models:
        yp = m.predict(ext_X_scaled)
        ext_preds.append(yp)
    ext_preds = np.column_stack(ext_preds)
    ext_mean = np.mean(ext_preds, axis=1)
    ext_std = np.std(ext_preds, axis=1)

    # Intervals via split-conformal calibrated on VAL
    q90 = qs['q90']
    q95 = qs['q95']
    lower90 = ext_mean - q90
    upper90 = ext_mean + q90
    lower95 = ext_mean - q95
    upper95 = ext_mean + q95

    # Compose output DataFrame
    out_df = pd.DataFrame({
        'acyl_chlorides': ext_comb['acyl_chlorides'].to_numpy(),
        'amines': ext_comb['amines'].to_numpy(),
        'y_true': ext_df.get(TARGET_COL, pd.Series([np.nan]*len(ext_mean))).to_numpy(),
        'y_pred_mean': ext_mean,
        'y_pred_std': ext_std,
        'lower90': lower90,
        'upper90': upper90,
        'lower95': lower95,
        'upper95': upper95,
    })

    # If external CSV has HTE_lnk or NMR_lnk, compute metrics too
    metrics: Dict[str, float] = {'val_q90': q90, 'val_q95': q95}
    for truth_col in [TARGET_COL, 'HTE_lnk', 'NMR_lnk']:
        if truth_col in ext_df.columns:
            yt = pd.to_numeric(ext_df[truth_col], errors='coerce').to_numpy()
            mask = ~np.isnan(yt)
            if mask.any():
                yt = yt[mask]
                yp = ext_mean[mask]
                sse = float(np.sum((yt - yp)**2))
                sst = float(np.sum((yt - float(np.mean(yt)))**2))
                r2 = 0.0 if sst == 0.0 else 1.0 - sse/sst
                mae = float(np.mean(np.abs(yt - yp)))
                rmse = float(np.sqrt(np.mean((yt - yp)**2)))
                # Coverage
                l90 = lower90[mask]
                u90 = upper90[mask]
                l95 = lower95[mask]
                u95 = upper95[mask]
                cov90 = float(np.mean((yt >= l90) & (yt <= u90)))
                cov95 = float(np.mean((yt >= l95) & (yt <= u95)))
                prefix = f'external_{truth_col}'
                metrics.update({
                    f'{prefix}_n': int(mask.sum()),
                    f'{prefix}_r2': r2,
                    f'{prefix}_mae': mae,
                    f'{prefix}_rmse': rmse,
                    f'{prefix}_coverage90': cov90,
                    f'{prefix}_coverage95': cov95,
                })

    pred_path = os.path.join(RESULTS_DIR, 'external_drug_scope_predictions_cpv2_S97.csv')
    out_df.to_csv(pred_path, index=False)
    print(f"Saved {pred_path}")

    metrics_path = os.path.join(RESULTS_DIR, 'external_drug_scope_metrics_cpv2_S97.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")


if __name__ == '__main__':
    main()



