#!/usr/bin/env python3
"""
Splitting utilities
===================
Create statistically meaningful Train/Val/Test splits (default 60/20/20),
optionally respecting group constraints by 'pair', 'amine', or 'acyl'.
Writes diagnostics to results/split_diagnostics.json.
"""

from __future__ import annotations

import os
import json
from typing import Tuple, Optional
import numpy as np
import pandas as pd


def _assign_groups_randomly(unique_groups: np.ndarray,
                            ratios: Tuple[float, float, float],
                            seed: int = 42) -> Tuple[set, set, set]:
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    # ensure all groups assigned
    n_train = min(n_train, n)
    n_val = min(n_val, max(n - n_train, 0))
    n_test = max(n - n_train - n_val, 0)
    train = set(shuffled[:n_train])
    val = set(shuffled[n_train:n_train + n_val])
    test = set(shuffled[n_train + n_val: n_train + n_val + n_test])
    return train, val, test


def generate_random_splits(df: pd.DataFrame,
                           group_by: Optional[str] = None,
                           ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                           seed: int = 42,
                           save_diagnostics: bool = True) -> pd.DataFrame:
    """Generate Train/Val/Test splits and set df['test splits'] accordingly.

    - group_by: None | 'pair' | 'amine' | 'acyl'
    - ratios: (train, val, test)
    - labels: 'TRAIN', 'VAL', 'TEST1' (to be compatible with existing code)
    """
    df = df.copy()

    if group_by is None or group_by == 'none':
        # row-level split
        groups = np.arange(len(df))
        train, val, test = _assign_groups_randomly(groups, ratios, seed)
        split_labels = np.array(['TRAIN'] * len(df), dtype=object)
        split_labels[list(val)] = 'VAL'
        split_labels[list(test)] = 'TEST1'
        df['test splits'] = split_labels
    else:
        if group_by == 'pair':
            if not {'acyl_chlorides', 'amines'}.issubset(df.columns):
                raise ValueError("Dataframe missing 'acyl_chlorides' or 'amines' columns for pair grouping")
            group_keys = (df['acyl_chlorides'].astype(str) + '|' + df['amines'].astype(str)).values
        elif group_by == 'amine':
            group_keys = df['amines'].astype(str).values
        elif group_by == 'acyl':
            group_keys = df['acyl_chlorides'].astype(str).values
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        unique = np.unique(group_keys)
        train_groups, val_groups, test_groups = _assign_groups_randomly(unique, ratios, seed)
        def _label(g):
            if g in train_groups:
                return 'TRAIN'
            if g in val_groups:
                return 'VAL'
            return 'TEST1'
        df['test splits'] = np.array([_label(g) for g in group_keys], dtype=object)

    if save_diagnostics:
        os.makedirs('results', exist_ok=True)
        diag = {
            'counts': df['test splits'].value_counts().to_dict(),
            'ratios': ratios,
            'group_by': group_by or 'none',
        }
        # simple target distribution if present
        for target_col in ['HTE_lnk_corrected', 'HTE_rate_corrected', 'bias']:
            if target_col in df.columns:
                stats = {}
                for split in ['TRAIN', 'VAL', 'TEST1']:
                    s = df.loc[df['test splits'] == split, target_col]
                    if len(s) > 0:
                        stats[split] = {
                            'n': int(len(s)),
                            'mean': float(np.nanmean(s)),
                            'std': float(np.nanstd(s)),
                            'min': float(np.nanmin(s)),
                            'max': float(np.nanmax(s)),
                        }
                diag[f'{target_col}_by_split'] = stats
        with open('results/split_diagnostics.json', 'w') as f:
            json.dump(diag, f, indent=2)

    return df


