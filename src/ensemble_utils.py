#!/usr/bin/env python3
"""
Ensemble Utilities
==================
Aggregate per-run prediction CSVs into an ensemble mean and per-point std,
evaluating blended Test R² on the canonical split.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def load_predictions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'acyl_chlorides', 'amines', 'y_true', 'y_pred', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def align_test_predictions(paths: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = []
    for p in paths:
        df = load_predictions_csv(p)
        df = df[df['split'] == 'TEST'].copy()
        df = df[['acyl_chlorides', 'amines', 'y_true', 'y_pred']]
        df = df.rename(columns={'y_pred': f"pred_{os.path.basename(p)}"})
        frames.append(df)

    # Inner-join on identifiers to enforce common test set across runs
    aligned = frames[0]
    for frm in frames[1:]:
        aligned = aligned.merge(frm, on=['acyl_chlorides', 'amines', 'y_true'], how='inner')

    pred_cols = [c for c in aligned.columns if c.startswith('pred_')]
    y_true = aligned['y_true'].to_numpy()
    return aligned, y_true, np.column_stack([aligned[c].to_numpy() for c in pred_cols])


def aggregate_predictions(paths: List[str], out_suffix: str = "") -> dict:
    if not paths:
        raise ValueError("No prediction paths provided")
    aligned, y_true, preds = align_test_predictions(paths)
    mean_pred = preds.mean(axis=1)
    std_pred = preds.std(axis=1)
    r2 = r2_score(y_true, mean_pred)

    out = aligned[['acyl_chlorides', 'amines', 'y_true']].copy()
    out['y_pred_mean'] = mean_pred
    out['y_pred_std'] = std_pred

    os.makedirs('results', exist_ok=True)
    out_path = f"results/ensemble_predictions{out_suffix}.csv"
    out.to_csv(out_path, index=False)

    return {
        'test_r2': float(r2),
        'n': int(len(out)),
        'paths': paths,
        'output': out_path
    }


