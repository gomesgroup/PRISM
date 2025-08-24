#!/usr/bin/env python3
"""
Parity plots for NMR ln k predictions using:
- Co-Kriging Multi-Fidelity model
- CheMeleon fine-tuned model (encoder frozen)

Figures mimic the visual style of `plot_gpg_style_parity_all6.py`.

Inputs:
- Co-Kriging predictions: results/cokriging_mf_predictions_nmr.csv
  - Columns: y_true_nmr_lnk, y_pred_nmr_lnk, y_std_delta (optional), split (optional)
- CheMeleon fine-tuned predictions (joined): results/nmr_finetune_predictions_joined.csv
  - Columns: amine_smiles, acid_smiles, NMR_lnk (true), NMR_lnk_pred (pred), test splits (optional)

Outputs:
- plots/parity_nmr_cokriging_mf.png
- plots/parity_nmr_chemeleon_finetuned.png
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D


def set_gpg_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (3.5, 3.2),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def _compute_split_metrics(y_true: np.ndarray, y_pred: np.ndarray, split: np.ndarray, include_val: bool) -> list[tuple[str, float, float, str]]:
    def _metrics(yt, yp):
        sse = float(((yt - yp) ** 2).sum())
        sst = float(((yt - yt.mean()) ** 2).sum())
        r2s = 0.0 if sst == 0.0 else 1.0 - sse / sst
        maes = float(np.mean(np.abs(yt - yp)))
        return r2s, maes

    group_colors = {
        "TRAIN": "#000000",
        "TEST": "#d62728",
        "VAL": "#FDB515",
        "TEST1": "#ef5959",
        "TEST2": "#a51616",
    }
    entries = [("train", group_colors["TRAIN"], (split == "TRAIN")), ("test", group_colors["TEST"], (split == "TEST") | (split == "TEST1") | (split == "TEST2"))]
    if include_val:
        entries.insert(1, ("val", group_colors["VAL"], (split == "VAL")))

    labels: list[tuple[str, float, float, str]] = []
    for lab, color, msk in entries:
        if np.any(msk):
            r2s, maes = _metrics(y_true[msk], y_pred[msk])
            labels.append((lab, r2s, maes, color))
    return labels


def parity_plot(
    df: pd.DataFrame,
    out_path: str,
    y_true_col: str,
    y_pred_col: str,
    std_col: Optional[str] = None,
    split_col: Optional[str] = None,
    font_family: Optional[str] = None,
    include_val: bool = False,
) -> None:
    # Prepare columns
    y_true_series = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred_series = pd.to_numeric(df[y_pred_col], errors="coerce")
    if std_col is not None and std_col in df.columns:
        std_series = pd.to_numeric(df[std_col], errors="coerce")
    else:
        std_series = pd.Series(np.ones(len(df), dtype=float))  # constant alpha
    if split_col is not None and split_col in df.columns:
        split_series = df[split_col].astype(str)
    else:
        split_series = pd.Series(["TEST"] * len(df))

    mask = (~y_true_series.isna()) & (~y_pred_series.isna()) & (~std_series.isna())
    y_true = y_true_series[mask].to_numpy()
    y_pred = y_pred_series[mask].to_numpy()
    std = std_series[mask].to_numpy()
    split = split_series[mask].to_numpy()

    # Drop VAL if requested
    if not include_val:
        keep = split != "VAL"
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        std = std[keep]
        split = split[keep]

    if len(y_true) == 0:
        print(f"No valid rows to plot for {out_path}")
        return

    # Axes limits and padding
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    pad = 0.1 * (y_max - y_min)
    y_lo = min(y_min - pad, -1.0)
    y_hi = max(y_max + pad, 4.0)

    fig, ax = plt.subplots()
    # 1:1 line
    ax.plot([y_lo, y_hi], [y_lo, y_hi], color="black", linewidth=1.0, zorder=1.5)

    # TEST-only linear fit line
    try:
        idx_test = (split == "TEST") | (split == "TEST1") | (split == "TEST2")
        if np.any(idx_test):
            coeffs = np.polyfit(y_true[idx_test], y_pred[idx_test], 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs[0] * xs + coeffs[1], color="#bdbdbd", linewidth=2.0, alpha=0.8, zorder=1.0)
    except Exception:
        pass

    # Map std to alpha range [0.2, 1.0]
    smin, smax = float(np.nanmin(std)), float(np.nanmax(std))
    if smax > smin:
        rel = (std - smin) / (smax - smin)
    else:
        rel = np.zeros_like(std)
    alpha_vals = 1.0 - 0.8 * rel

    # Colors per split
    group_colors = {
        "TRAIN": "#000000",
        "TEST": "#d62728",
        "VAL": "#FDB515",
        "TEST1": "#ef5959",
        "TEST2": "#a51616",
    }
    labs = ("TRAIN", "VAL", "TEST", "TEST1", "TEST2") if include_val else ("TRAIN", "TEST", "TEST1", "TEST2")
    for lab in labs:
        if lab not in group_colors:
            continue
        color = group_colors[lab]
        idx = (split == lab)
        if not np.any(idx):
            continue
        face_rgba = np.array([mcolors.to_rgba(color, a) for a in alpha_vals[idx]])
        ax.scatter(
            y_true[idx],
            y_pred[idx],
            s=6,
            marker="s",
            facecolors=face_rgba,
            edgecolors=color,
            linewidths=0.8,
            zorder=3.0,
            label=lab.lower(),
        )

    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(y_lo, y_hi)
    tick_vals = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)

    label_fp = None
    ax.set_xlabel("measured ln k", fontproperties=label_fp)
    ax.set_ylabel("predicted ln k", fontproperties=label_fp)
    ax.xaxis.label.set_fontstyle("normal")
    ax.yaxis.label.set_fontstyle("normal")
    ax.set_axisbelow(True)
    ax.grid(True, color="#ececec", linewidth=0.6, zorder=0.1)

    # Legend with split metrics
    try:
        labels = _compute_split_metrics(y_true, y_pred, split, include_val)
        handles = []
        for lab, r2s, maes, color in labels:
            lbl = f"{lab:<5} R²={r2s:.3f}   MAE={maes:.3f}"
            handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=color, markeredgecolor=color, markersize=4, label=lbl))
        ax.legend(
            handles=handles,
            loc='lower right',
            frameon=False,
            handletextpad=0.3,
            labelspacing=0.3,
            borderaxespad=0.25,
            borderpad=0.2,
            markerscale=0.8,
            handlelength=0.9,
            columnspacing=0.6,
        )
    except Exception:
        pass

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    set_gpg_style()

    # 1) Co-Kriging MF
    ck_csv = "results/cokriging_mf_predictions_nmr.csv"
    if os.path.exists(ck_csv):
        df_ck = pd.read_csv(ck_csv)
        # Standardize split column name
        if "split" in df_ck.columns:
            split_col = "split"
        elif "test splits" in df_ck.columns:
            split_col = "test splits"
        else:
            split_col = None
        parity_plot(
            df=df_ck,
            out_path="plots/parity_nmr_cokriging_mf.png",
            y_true_col="y_true_nmr_lnk",
            y_pred_col="y_pred_nmr_lnk",
            std_col="y_std_delta" if "y_std_delta" in df_ck.columns else None,
            split_col=split_col,
            include_val=False,
        )
    else:
        print(f"Missing Co-Kriging predictions: {ck_csv}")

    # 2) CheMeleon fine-tuned
    ft_csv = "results/nmr_finetune_predictions_joined.csv"
    if os.path.exists(ft_csv):
        df_ft = pd.read_csv(ft_csv)
        # Identify split column if any
        split_col = None
        if "split" in df_ft.columns:
            split_col = "split"
        elif "test splits" in df_ft.columns:
            split_col = "test splits"
        # Parity
        y_pred_col = "NMR_lnk_pred" if "NMR_lnk_pred" in df_ft.columns else "NMR_lnk_x"
        y_true_col = "NMR_lnk" if "NMR_lnk" in df_ft.columns else "NMR_lnk_y"
        parity_plot(
            df=df_ft,
            out_path="plots/parity_nmr_chemeleon_finetuned.png",
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            std_col=None,
            split_col=split_col,
            include_val=False,
        )
    else:
        print(f"Missing fine-tuned predictions: {ft_csv}")


if __name__ == "__main__":
    main()


