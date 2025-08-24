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
    y_label: str = "predicted ln k",
) -> None:
    # Prepare columns
    y_true_series = pd.to_numeric(df[y_true_col], errors="coerce")
    y_pred_series = pd.to_numeric(df[y_pred_col], errors="coerce")
    std_series: Optional[pd.Series]
    if std_col is not None and std_col in df.columns:
        std_series = pd.to_numeric(df[std_col], errors="coerce")
    else:
        std_series = None
    if split_col is not None and split_col in df.columns:
        split_series = df[split_col].astype(str)
    else:
        split_series = pd.Series(["TEST"] * len(df))

    mask = (~y_true_series.isna()) & (~y_pred_series.isna())
    if std_series is not None:
        mask = mask & (~std_series.isna())
    y_true = y_true_series[mask].to_numpy()
    y_pred = y_pred_series[mask].to_numpy()
    std = std_series[mask].to_numpy() if std_series is not None else None
    split = split_series[mask].to_numpy()

    # Drop VAL if requested
    if not include_val:
        keep = split != "VAL"
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        if std is not None:
            std = std[keep]
        split = split[keep]

    if len(y_true) == 0:
        print(f"No valid rows to plot for {out_path}")
        return

    # Axes limits with padding and fixed tick coverage [-1, 4]
    y_min = float(min(y_true.min(), y_pred.min()))
    y_max = float(max(y_true.max(), y_pred.max()))
    pad = 0.1 * (y_max - y_min if y_max > y_min else 1.0)
    y_lo = min(y_min - pad, -1.0)
    y_hi = max(y_max + pad, 4.0)

    fig, ax = plt.subplots()
    # 1:1 diagonal
    ax.plot([y_lo, y_hi], [y_lo, y_hi], color="black", linewidth=1.0, zorder=1.5)

    # Thin, faded linear fits for TRAIN/TEST/VAL (if present)
    try:
        idx_test = (split == "TEST") | (split == "TEST1") | (split == "TEST2")
        if np.any(idx_test):
            xs = np.linspace(y_lo, y_hi, 200)
            coeffs_t = np.polyfit(np.asarray(y_true[idx_test], dtype=float), np.asarray(y_pred[idx_test], dtype=float), 1)
            ax.plot(xs, coeffs_t[0] * xs + coeffs_t[1], color="#d62728", linewidth=0.8, alpha=0.25, zorder=1.0)
        idx_train = (split == "TRAIN")
        if np.any(idx_train):
            xs = np.linspace(y_lo, y_hi, 200)
            coeffs_tr = np.polyfit(np.asarray(y_true[idx_train], dtype=float), np.asarray(y_pred[idx_train], dtype=float), 1)
            ax.plot(xs, coeffs_tr[0] * xs + coeffs_tr[1], color="#B0B0B0", linewidth=0.8, alpha=0.25, zorder=1.0)
        if include_val:
            idx_val = (split == "VAL")
            if np.any(idx_val):
                xs = np.linspace(y_lo, y_hi, 200)
                coeffs_v = np.polyfit(np.asarray(y_true[idx_val], dtype=float), np.asarray(y_pred[idx_val], dtype=float), 1)
                ax.plot(xs, coeffs_v[0] * xs + coeffs_v[1], color="#FDB515", linewidth=0.8, alpha=0.25, zorder=1.0)
    except Exception:
        pass

    # Alpha mapping from std if available; else constant
    if std is not None:
        smin = float(np.nanmin(std))
        smax = float(np.nanmax(std))
        if smax > smin:
            rel = (std - smin) / (smax - smin)
        else:
            rel = np.zeros_like(std)
        alpha_vals = 1.0 - 0.8 * rel
    else:
        alpha_vals = np.full_like(y_true, 0.85, dtype=float)

    # Colors and draw order (VAL on top)
    group_colors = {
        "TRAIN": "#D0D0D0",
        "TEST": "#d62728",
        "VAL": "#FDB515",
        "TEST1": "#ef5959",
        "TEST2": "#a51616",
    }
    labs = ("TRAIN", "TEST", "TEST1", "TEST2", "VAL") if include_val else ("TRAIN", "TEST", "TEST1", "TEST2")
    for lab in labs:
        if lab not in group_colors:
            continue
        color = group_colors[lab]
        idx = (split == lab)
        if not np.any(idx):
            continue
        face_rgba = np.array([mcolors.to_rgba(color, a) for a in alpha_vals[idx]])
        lw = 0.5
        size = 4.5 if lab == "TRAIN" else 6
        ax.scatter(
            y_true[idx],
            y_pred[idx],
            s=size,
            marker="s",
            facecolors=face_rgba,
            edgecolors=color,
            linewidths=lw,
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
    ax.set_ylabel(y_label, fontproperties=label_fp)
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
    # Prefer OOF predictions if available
    ck_csv = "results/cokriging_mf_predictions_nmr_oof.csv"
    if not os.path.exists(ck_csv):
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
            y_label="predicted ln k (Co-Kriging MF)",
        )
    else:
        print(f"Missing Co-Kriging predictions: {ck_csv}")

    # 2) CheMeleon fine-tuned
    # Prefer 70/15/15 joined predictions if available
    ft_csv = "results/nmr_finetune_70_15_15_predictions_joined.csv"
    if not os.path.exists(ft_csv):
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
        y_pred_col = "NMR_lnk_pred" if "NMR_lnk_pred" in df_ft.columns else ("NMR_lnk_x" if "NMR_lnk_x" in df_ft.columns else "NMR_lnk")
        y_true_col = "NMR_lnk" if "NMR_lnk" in df_ft.columns else "NMR_lnk_y"
        # TEST-only figure
        if split_col is not None:
            df_test = df_ft[df_ft[split_col].astype(str).isin(["TEST"])].copy()
            if not df_test.empty:
                parity_plot(
                    df=df_test,
                    out_path="plots/parity_nmr_chemeleon_finetuned_TEST.png",
                    y_true_col=y_true_col,
                    y_pred_col=y_pred_col,
                    std_col=None,
                    split_col=split_col,
                    include_val=False,
                    y_label="predicted ln k (CheMeleon FT)",
                )
        # All splits figure
        parity_plot(
            df=df_ft,
            out_path="plots/parity_nmr_chemeleon_finetuned_TRAIN_VAL_TEST.png",
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            std_col=None,
            split_col=split_col,
            include_val=True,
            y_label="predicted ln k (CheMeleon FT)",
        )
    else:
        print(f"Missing fine-tuned predictions: {ft_csv}")

    # 3) CheMeleon CV (replicate validation predictions aggregated)
    cv_files = [f"results/nmr_cv5_rep{r}_val_pred.csv" for r in range(5)]
    cv_files = [p for p in cv_files if os.path.exists(p)]
    if len(cv_files) > 0:
        # Concatenate predictions and join true labels from TRAIN set
        dfs = []
        for p in cv_files:
            d = pd.read_csv(p)
            if "NMR_lnk" in d.columns:
                d = d.rename(columns={"NMR_lnk": "NMR_lnk_pred"})
            d["replicate_file"] = os.path.basename(p)
            dfs.append(d)
        dcat = pd.concat(dfs, ignore_index=True)
        train_true = pd.read_csv("data/rates/nmr_lnk_smiles_train_70.csv")
        dcat = dcat.merge(train_true[["amine_smiles","acid_smiles","NMR_lnk"]], on=["amine_smiles","acid_smiles"], how="left")
        # Mark as VAL for plotting
        dcat["split"] = "VAL"
        # Plot CV VAL-only parity
        parity_plot(
            df=dcat,
            out_path="plots/parity_nmr_chemeleon_finetuned_CV_VAL.png",
            y_true_col="NMR_lnk",
            y_pred_col="NMR_lnk_pred",
            std_col=None,
            split_col="split",
            include_val=True,
            y_label="predicted ln k (CheMeleon FT)",
        )
    else:
        print("No CV validation prediction files found; skipping CV parity plot.")


if __name__ == "__main__":
    main()


