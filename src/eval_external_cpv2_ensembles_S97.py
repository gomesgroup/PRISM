#!/usr/bin/env python3
"""
Evaluate external drug-scope holdout using saved cpv2 runs (ens7_gpu0..3).

For each cpv2 run:
- Load scaler, regressor, and features via run suffix
- Load its combined_df and predictions CSV to extract VAL (y_true, y_pred) for conformal
- Build external features aligned to the run's training columns, predict external
- Calibrate split-conformal on VAL residuals for that run

Then build ensembles over runs:
- Equal-mean across runs
- Ridge and NNLS-weighted ensembles using TRAIN predictions across runs if available
  (fallback to VAL for weight fitting if TRAIN not present)

Outputs:
- results/external_cpv2_per_run_predictions_S97.csv
- results/external_cpv2_ensembles_S97.csv
- results/external_cpv2_metrics_S97.json
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_processing import load_and_process_features
from model_building import create_combined_dataset, load_models

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = "results"
EXTERNAL_CSV = "new_data/combined_features_hte_rates_drug_scope.csv"
# TARGET_COL is determined dynamically from the external CSV columns
TARGET_CANDIDATES = ["HTE_lnk_corrected", "NMR_lnk"]
RUN_TAGS = ["cpv2_gpu0", "cpv2_gpu1", "cpv2_gpu2", "cpv2_gpu3"]


def load_run_artifacts(tag: str):
    suffix = f"_each_8_optuna_{tag}"
    loaded = load_models(suffix)
    if loaded is None:
        return None
    classifier, regressor, scaler_class, scaler_reg, features = loaded
    # Load combined_df and predictions for VAL residuals
    combined_path = os.path.join(RESULTS_DIR, f"combined_df{suffix}.csv")
    preds_path = os.path.join(RESULTS_DIR, f"predictions{suffix}.csv")
    if not (os.path.exists(combined_path) and os.path.exists(preds_path)):
        return None
    combined_df = pd.read_csv(combined_path)
    preds_df = pd.read_csv(preds_path)
    return {
        'suffix': suffix,
        'regressor': regressor,
        'scaler_reg': scaler_reg,
        'features': features,
        'combined_df': combined_df,
        'preds_df': preds_df,
    }


def build_external_feature_matrix(run: Dict[str, Any], ext_df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    stub = ext_df[['acyl_chlorides', 'amines']].copy()
    if target_col not in stub.columns:
        stub[target_col] = 0.0
    loaded = load_and_process_features(stub, target_col=target_col)
    # Handle potential variant returns robustly
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        acid_ext, amine_ext = loaded[0], loaded[1]
    else:
        raise RuntimeError("load_and_process_features did not return expected outputs")
    combined_ext, feat_cols = create_combined_dataset(
        acid_ext, amine_ext, stub, selected_features=list(run['features']), save_df=False,
        rxn_features=None, add_interactions=True,
    )
    # Columns used in training could include interactions; align by scaler.feature_names_in_ if present
    scaler = run['scaler_reg']
    trained_cols = getattr(scaler, 'feature_names_in_', None)
    if trained_cols is not None:
        try:
            X_ext = combined_ext.reindex(columns=list(trained_cols), fill_value=0.0)
        except Exception:
            common = [c for c in trained_cols if c in combined_ext.columns]
            X_ext = combined_ext.reindex(columns=common, fill_value=0.0)
    else:
        X_ext = combined_ext.reindex(columns=list(run['features']), fill_value=0.0)
        # include interaction terms if present in combined_ext
        inter_cols = [c for c in combined_ext.columns if c.startswith('int_')]
        if inter_cols:
            X_ext = pd.concat([X_ext, combined_ext[inter_cols]], axis=1)
    X_ext_scaled = scaler.transform(X_ext)
    return X_ext_scaled, list(X_ext.columns), combined_ext[['acyl_chlorides', 'amines']]


def per_run_predict_and_conformal(run: Dict[str, Any], X_ext_scaled: np.ndarray) -> Dict[str, Any]:
    reg = run['regressor']
    ext_pred = reg.predict(X_ext_scaled)
    # Calibrate on VAL
    combined_df = run['combined_df']
    preds_df = run['preds_df']
    val_mask = combined_df['test splits'] == 'VAL'
    if val_mask.any() and {'y_true', 'y_pred'}.issubset(preds_df.columns):
        # Align VAL ids ordering with preds_df VAL slice
        val_ids = combined_df.loc[val_mask, ['acyl_chlorides', 'amines']].reset_index(drop=True)
        _val_df = pd.concat([val_ids, preds_df.loc[preds_df['split'] == 'VAL', ['y_true', 'y_pred']].reset_index(drop=True)], axis=1)
        if 'y_true' in _val_df.columns and 'y_pred' in _val_df.columns:
            vtrue = pd.to_numeric(_val_df['y_true'], errors='coerce').to_numpy()
            vpred = pd.to_numeric(_val_df['y_pred'], errors='coerce').to_numpy()
            mask = ~np.isnan(vtrue) & ~np.isnan(vpred)
            abs_res = np.abs(vtrue[mask] - vpred[mask])
            if abs_res.size > 0:
                q90 = float(np.quantile(abs_res, 0.90))
                q95 = float(np.quantile(abs_res, 0.95))
            else:
                q90 = q95 = float('nan')
        else:
            q90 = q95 = float('nan')
    else:
        q90 = q95 = float('nan')
    return {'pred': ext_pred, 'q90': q90, 'q95': q95}


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Load external CSV
    ext_df = pd.read_csv(EXTERNAL_CSV)
    # Determine target column from candidates
    target_col = None
    for cand in TARGET_CANDIDATES:
        if cand in ext_df.columns:
            target_col = cand
            break
    if target_col is None:
        target_col = TARGET_CANDIDATES[0]
    # Load runs
    runs: List[Dict] = []
    for tag in RUN_TAGS:
        r = load_run_artifacts(tag)
        if r is not None:
            runs.append(r)
    if not runs:
        print("No cpv2 runs found; aborting.")
        return

    per_run_outputs: Dict[str, Dict[str, Any]] = {}
    per_run_cols: Dict[str, List[str]] = {}
    ids: pd.DataFrame = pd.DataFrame(columns=['acyl_chlorides', 'amines'])
    for r in runs:
        X_ext_scaled, cols, id_df = build_external_feature_matrix(r, ext_df, target_col)
        pr = per_run_predict_and_conformal(r, X_ext_scaled)
        per_run_outputs[r['suffix']] = pr
        per_run_cols[r['suffix']] = cols
        if ids.empty:
            ids = id_df.copy()

    # Compose per-run predictions table
    out = ids.copy()
    for suf, pr in per_run_outputs.items():
        out[f'pred_{suf}'] = pr['pred']
        out[f'q90_{suf}'] = pr['q90']
        out[f'q95_{suf}'] = pr['q95']
    # Mean prediction across runs (ignore NaNs)
    pred_cols = [c for c in out.columns if c.startswith('pred_')]
    pred_mat = np.column_stack([pd.to_numeric(out[c], errors='coerce').to_numpy() for c in pred_cols])
    mean_pred = np.nanmean(pred_mat, axis=1)
    std_pred = np.nanstd(pred_mat, axis=1)
    # Build VAL alignment across runs to calibrate ensemble intervals properly
    val_rows: List[pd.DataFrame] = []
    for r in runs:
        dfc = r['combined_df']
        p = r['preds_df']
        val_mask = p['split'] == 'VAL'
        if val_mask.any():
            ids_df = dfc.loc[dfc['test splits'] == 'VAL', ['acyl_chlorides', 'amines']].reset_index(drop=True)
            tmp = ids_df.copy()
            tmp['y_true'] = pd.to_numeric(p.loc[val_mask, 'y_true'], errors='coerce').reset_index(drop=True)
            tmp[r['suffix']] = pd.to_numeric(p.loc[val_mask, 'y_pred'], errors='coerce').reset_index(drop=True)
            val_rows.append(tmp)
    ens_q90 = float('nan')
    ens_q95 = float('nan')
    q90_ridge = float('nan')
    q95_ridge = float('nan')
    q90_nnls = float('nan')
    q95_nnls = float('nan')
    member_order = [r['suffix'] for r in runs]
    X_val = None
    y_val = None
    member_cols_val: List[str] = []
    val_df = None
    if val_rows:
        _val_df = val_rows[0]
        for df in val_rows[1:]:
            _val_df = pd.merge(_val_df, df, on=['acyl_chlorides', 'amines', 'y_true'], how='inner')
        if _val_df is not None and not _val_df.empty:
            val_df = _val_df
            # Build matrix in member order
            member_cols_val = [c for c in member_order if c in val_df.columns]
            if member_cols_val:
                X_val = val_df[member_cols_val].to_numpy()
                y_val = val_df['y_true'].to_numpy()
                # Equal-mean on VAL and residual quantiles
                mean_val = np.nanmean(X_val, axis=1)
                abs_res_val_mean = np.abs(y_val - mean_val)
                if abs_res_val_mean.size > 0 and np.isfinite(abs_res_val_mean).any():
                    ens_q90 = float(np.quantile(abs_res_val_mean, 0.90))
                    ens_q95 = float(np.quantile(abs_res_val_mean, 0.95))

    if target_col in ext_df.columns:
        out['y_true'] = pd.to_numeric(ext_df[target_col], errors='coerce')
    else:
        out['y_true'] = pd.Series(np.nan, index=out.index)
    out['pred_mean'] = mean_pred
    out['pred_std'] = std_pred
    out['lower90_mean'] = mean_pred - ens_q90
    out['upper90_mean'] = mean_pred + ens_q90
    out['lower95_mean'] = mean_pred - ens_q95
    out['upper95_mean'] = mean_pred + ens_q95

    # Fit weights (Ridge and NNLS) on TRAIN predictions if available from each run
    # Build TRAIN matrix by joining each run's TRAIN predictions via ids
    train_rows: List[pd.DataFrame] = []
    for r in runs:
        dfc = r['combined_df']
        p = r['preds_df']
        train_mask = p['split'] == 'TRAIN'
        if train_mask.any():
            ids_df = dfc.loc[dfc['test splits'] == 'TRAIN', ['acyl_chlorides', 'amines']].reset_index(drop=True)
            tmp = ids_df.copy()
            tmp['y_true'] = pd.to_numeric(p.loc[train_mask, 'y_true'], errors='coerce').reset_index(drop=True)
            tmp[r['suffix']] = pd.to_numeric(p.loc[train_mask, 'y_pred'], errors='coerce').reset_index(drop=True)
            train_rows.append(tmp)
    # Align across runs on common ids
    if train_rows:
        train_df = train_rows[0]
        for df in train_rows[1:]:
            train_df = pd.merge(train_df, df, on=['acyl_chlorides', 'amines', 'y_true'], how='inner')
        # Build X and y
        member_cols = [c for c in train_df.columns if c.startswith('pred_') or c.startswith('_each') or 'cpv2' in c]
        # In our case, per-run names are suffix strings; select exact run columns
        member_cols = [c for c in train_df.columns if c in [r['suffix'] for r in runs]]
        if member_cols:
            X_tr = train_df[member_cols].to_numpy()
            y_tr = train_df['y_true'].to_numpy()
            # Ridge weights
            ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False)
            try:
                ridge.fit(X_tr, y_tr)
                w_ridge = ridge.coef_
            except Exception:
                w_ridge = None
            # NNLS weights (use non-negative least squares via np.linalg.lstsq and clip)
            try:
                w_nnls_full = np.linalg.lstsq(np.clip(X_tr, 0.0, None), y_tr, rcond=None)
                w_nnls = np.maximum(w_nnls_full[0], 0.0)
            except Exception:
                w_nnls = None
            # External weighted preds
            # Build external members in the same order
            ext_members = np.column_stack([out[f'pred_{s}'].to_numpy() for s in member_order])
            if w_ridge is not None and ext_members.shape[1] == len(w_ridge):
                out['pred_ridge'] = ext_members.dot(w_ridge)
            if w_nnls is not None and ext_members.shape[1] == len(w_nnls):
                out['pred_nnls'] = ext_members.dot(w_nnls)
            # Calibrate Ridge/NNLS conformal on VAL by combining member VAL predictions
            if val_df is not None and member_cols_val and X_val is not None and y_val is not None:
                if w_ridge is not None and len(w_ridge) == X_val.shape[1]:
                    pred_val_ridge = X_val.dot(w_ridge)
                    abs_res_val_ridge = np.abs(y_val - pred_val_ridge)
                    if abs_res_val_ridge.size > 0:
                        q90_ridge = float(np.quantile(abs_res_val_ridge, 0.90))
                        q95_ridge = float(np.quantile(abs_res_val_ridge, 0.95))
                if w_nnls is not None and len(w_nnls) == X_val.shape[1]:
                    pred_val_nnls = X_val.dot(w_nnls)
                    abs_res_val_nnls = np.abs(y_val - pred_val_nnls)
                    if abs_res_val_nnls.size > 0:
                        q90_nnls = float(np.quantile(abs_res_val_nnls, 0.90))
                        q95_nnls = float(np.quantile(abs_res_val_nnls, 0.95))
    # Conformal for weighted ensembles: reuse mean of per-run q as approximation
    # Apply ensemble-specific conformal for weighted ensembles when available
    if 'pred_ridge' in out.columns:
        if np.isfinite(q90_ridge):
            out['lower90_ridge'] = out['pred_ridge'] - q90_ridge
            out['upper90_ridge'] = out['pred_ridge'] + q90_ridge
        else:
            out['lower90_ridge'] = out['pred_ridge'] * np.nan
            out['upper90_ridge'] = out['pred_ridge'] * np.nan
        if np.isfinite(q95_ridge):
            out['lower95_ridge'] = out['pred_ridge'] - q95_ridge
            out['upper95_ridge'] = out['pred_ridge'] + q95_ridge
        else:
            out['lower95_ridge'] = out['pred_ridge'] * np.nan
            out['upper95_ridge'] = out['pred_ridge'] * np.nan
    if 'pred_nnls' in out.columns:
        if np.isfinite(q90_nnls):
            out['lower90_nnls'] = out['pred_nnls'] - q90_nnls
            out['upper90_nnls'] = out['pred_nnls'] + q90_nnls
        else:
            out['lower90_nnls'] = out['pred_nnls'] * np.nan
            out['upper90_nnls'] = out['pred_nnls'] * np.nan
        if np.isfinite(q95_nnls):
            out['lower95_nnls'] = out['pred_nnls'] - q95_nnls
            out['upper95_nnls'] = out['pred_nnls'] + q95_nnls
        else:
            out['lower95_nnls'] = out['pred_nnls'] * np.nan
            out['upper95_nnls'] = out['pred_nnls'] * np.nan

    # Save per-run and ensemble outputs
    per_run_path = os.path.join(RESULTS_DIR, 'external_cpv2_per_run_predictions_S97.csv')
    out.to_csv(per_run_path, index=False)
    print(f"Saved {per_run_path}")

    # Summarize metrics if y_true available
    metrics: Dict[str, float] = {'n_runs': len(runs), 'ens_q90': ens_q90, 'ens_q95': ens_q95}
    def add_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, l90: np.ndarray, u90: np.ndarray, l95: np.ndarray, u95: np.ndarray):
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not mask.any():
            return
        yt = y_true[mask]
        yp = y_pred[mask]
        sse = float(np.sum((yt - yp) ** 2))
        sst = float(np.sum((yt - float(np.mean(yt))) ** 2))
        r2 = 0.0 if sst == 0.0 else 1.0 - sse / sst
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        cov90 = float(np.mean((yt >= l90[mask]) & (yt <= u90[mask])))
        cov95 = float(np.mean((yt >= l95[mask]) & (yt <= u95[mask])))
        metrics.update({
            f'{prefix}_n': int(mask.sum()),
            f'{prefix}_r2': r2,
            f'{prefix}_mae': mae,
            f'{prefix}_rmse': rmse,
            f'{prefix}_coverage90': cov90,
            f'{prefix}_coverage95': cov95,
        })

    y_true = pd.to_numeric(out['y_true'], errors='coerce').to_numpy()
    # mean ensemble metrics
    add_metrics('external_mean', y_true, out['pred_mean'].to_numpy(), out['lower90_mean'].to_numpy(), out['upper90_mean'].to_numpy(), out['lower95_mean'].to_numpy(), out['upper95_mean'].to_numpy())
    # ridge
    if 'pred_ridge' in out.columns:
        l90 = out['lower90_ridge'].to_numpy() if 'lower90_ridge' in out.columns else np.full_like(out['pred_ridge'].to_numpy(), np.nan)
        u90 = out['upper90_ridge'].to_numpy() if 'upper90_ridge' in out.columns else np.full_like(out['pred_ridge'].to_numpy(), np.nan)
        l95 = out['lower95_ridge'].to_numpy() if 'lower95_ridge' in out.columns else np.full_like(out['pred_ridge'].to_numpy(), np.nan)
        u95 = out['upper95_ridge'].to_numpy() if 'upper95_ridge' in out.columns else np.full_like(out['pred_ridge'].to_numpy(), np.nan)
        add_metrics('external_ridge', y_true, out['pred_ridge'].to_numpy(), l90, u90, l95, u95)
    # nnls
    if 'pred_nnls' in out.columns:
        l90 = out['lower90_nnls'].to_numpy() if 'lower90_nnls' in out.columns else np.full_like(out['pred_nnls'].to_numpy(), np.nan)
        u90 = out['upper90_nnls'].to_numpy() if 'upper90_nnls' in out.columns else np.full_like(out['pred_nnls'].to_numpy(), np.nan)
        l95 = out['lower95_nnls'].to_numpy() if 'lower95_nnls' in out.columns else np.full_like(out['pred_nnls'].to_numpy(), np.nan)
        u95 = out['upper95_nnls'].to_numpy() if 'upper95_nnls' in out.columns else np.full_like(out['pred_nnls'].to_numpy(), np.nan)
        add_metrics('external_nnls', y_true, out['pred_nnls'].to_numpy(), l90, u90, l95, u95)

    metrics_path = os.path.join(RESULTS_DIR, 'external_cpv2_metrics_S97.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")


if __name__ == '__main__':
    main()



