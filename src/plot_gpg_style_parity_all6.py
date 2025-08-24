#!/usr/bin/env python3
"""
GPG-style parity plot for the all-6 NNLS ensemble with conformal intervals.

Inputs:
- results/ensemble_all6_predictions.csv (requires columns y_true, pred_nnls, lower95_nnls, upper95_nnls, lower90_nnls, upper90_nnls)
- results/ensemble_all6_metrics.json (optional; used for annotating R2/MAE/RMSE)

Outputs:
- plots/parity_all6_nnls_95.png (pred vs true with 95% conformal error bars)
- plots/parity_all6_nnls_90.png (pred vs true with 90% conformal error bars)
"""

from __future__ import annotations

import json
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

from utils.plot_style import apply_default_style
apply_default_style()


PREDICTIONS_CSV = "results/ensemble_all6_predictions_full.csv"
METRICS_JSON = "results/ensemble_all6_metrics.json"
PLOT_95 = "plots/parity_chemeleon_bigffn_model0_fixedsplits_TEST1_as_TEST_TEST2_as_VAL.png"
PLOT_90 = "plots/parity_chemeleon_bigffn_model0_fixedsplits_TEST1_as_TEST_TEST2_as_VAL_90.png"
PLOT_95_WITH_VAL = "plots/parity_chemeleon_bigffn_model0_fixedsplits_TEST1_as_TEST_TEST2_as_VAL_with_val.png"
PLOT_95_ALT = None  # disable alternate output to avoid duplicate files


def set_gpg_style() -> None:
    # Keep only non-font plot rcParams; rely on SciencePlots styles for appearance
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
    return None


def load_metrics(path: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        with open(path, "r") as f:
            m = json.load(f)
        r2 = m.get("nnls", {}).get("test_r2")
        mae = m.get("nnls", {}).get("test_mae")
        rmse = m.get("nnls", {}).get("test_rmse")
        return float(r2), float(mae), float(rmse)
    except Exception:
        return None, None, None


def parity_plot(
    df: pd.DataFrame,
    out_path: str,
    r2: Optional[float],
    mae: Optional[float],
    rmse: Optional[float],
    font_family: Optional[str] = None,
    include_val: bool = True,
) -> None:
    # Map uncertainty to alpha (lighter == higher uncertainty)
    # Use std_weighted_nnls if present, else std_unweighted
    std_col = "std_weighted_nnls" if "std_weighted_nnls" in df.columns else "std_unweighted"
    std = pd.to_numeric(df[std_col], errors="coerce")
    y_true = pd.to_numeric(df["y_true"], errors="coerce")
    y_pred = pd.to_numeric(df["pred_nnls"], errors="coerce")
    split = df["split"].astype(str)
    mask = (~y_true.isna()) & (~y_pred.isna()) & (~std.isna())
    y_true = y_true[mask].to_numpy()
    y_pred = y_pred[mask].to_numpy()
    std = std[mask].to_numpy()
    split = split[mask].to_numpy()

    # Explicitly drop validation points if requested
    if not include_val:
        keep = split != "VAL"
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        std = std[keep]
        split = split[keep]

    if len(y_true) == 0:
        print(f"No valid rows to plot for {out_path}")
        return

    # Axes limits with small padding
    y_min = min(y_true.min(), y_pred.min())
    y_max = max(y_true.max(), y_pred.max())
    pad = 0.1 * (y_max - y_min)
    y_lo = y_min - pad
    y_hi = y_max + pad
    # Ensure axes cover the desired tick range [-1, 4]
    y_lo = min(y_lo, -1.0)
    y_hi = max(y_hi, 4.0)

    fig, ax = plt.subplots()
    # Diagonal 1:1
    ax.plot([y_lo, y_hi], [y_lo, y_hi], color="black", linewidth=1.0, zorder=1.5)
    # Light linear fit line on TEST for visual guidance (combine TEST/TEST1/TEST2)
    try:
        idx_test = (split == "TEST") | (split == "TEST1") | (split == "TEST2")
        if np.any(idx_test):
            coeffs = np.polyfit(y_true[idx_test], y_pred[idx_test], 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs[0] * xs + coeffs[1], color="#bdbdbd", linewidth=2.0, alpha=0.8, zorder=1.0)
    except Exception:
        pass

    # Normalize std to alpha: lighter == higher UQ
    # Map std to [0.2, 1.0] inverted (higher std -> lower alpha)
    smin, smax = float(np.nanmin(std)), float(np.nanmax(std))
    if smax > smin:
        rel = (std - smin) / (smax - smin)
    else:
        rel = np.zeros_like(std)
    alpha_vals = 1.0 - 0.8 * rel  # in [0.2,1.0]

    # Plot groups separately, squares
    # Colors: TRAIN black, TEST red, VAL gold (CMU Gold Thread #FDB515)
    # Distinguish TEST1/TEST2 as lighter/darker reds for clarity
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
        # Use alpha on facecolors; keep a solid edgecolor for better contrast
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
    # Force matching tick positions on both axes
    tick_vals = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ax.set_xticks(tick_vals)
    ax.set_yticks(tick_vals)
    # Enforce font on axis labels and ticks via FontProperties
    label_fp = None
    if font_family is not None:
        try:
            label_fp = fm.FontProperties(family=font_family)
        except Exception:
            label_fp = None
    ax.set_xlabel("measured ln k", fontproperties=label_fp)
    ax.set_ylabel("predicted ln k (big-FFN ens-5)", fontproperties=label_fp)
    ax.xaxis.label.set_fontstyle("normal")
    ax.yaxis.label.set_fontstyle("normal")
    # Ensure ticks use the specified font family
    if label_fp is not None:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            try:
                tick.set_fontproperties(label_fp)
            except Exception:
                break
    # Legend will be created after computing split metrics to include R²/MAE
    # Subtle background grid behind points
    ax.set_axisbelow(True)
    ax.grid(True, color="#ececec", linewidth=0.6, zorder=0.1)

    # Per-split metrics; include inside legend (no separate boxes)
    try:
        def _snap_to_grid(ax, x_target_frac: float, y_target_frac: float) -> tuple[float, float]:
            xlo, xhi = ax.get_xlim()
            ylo, yhi = ax.get_ylim()
            def frac(vals, lo, hi):
                return [(v - lo) / (hi - lo) for v in vals if lo <= v <= hi]
            x_fracs = frac(ax.get_xticks(), xlo, xhi)
            y_fracs = frac(ax.get_yticks(), ylo, yhi)
            if len(x_fracs) == 0:
                xf = x_target_frac
            else:
                xf = min(x_fracs, key=lambda f: abs(f - x_target_frac))
            if len(y_fracs) == 0:
                yf = y_target_frac
            else:
                yf = min(y_fracs, key=lambda f: abs(f - y_target_frac))
            return float(xf), float(yf)

        def _metrics(y_t, y_p):
            yt = y_t
            yp = y_p
            sse = float(((yt - yp) ** 2).sum())
            sst = float(((yt - yt.mean()) ** 2).sum())
            r2s = 0.0 if sst == 0.0 else 1.0 - sse / sst
            maes = float(np.mean(np.abs(yt - yp)))
            return r2s, maes
        # Compute split metrics for TRAIN/VAL/TEST (TEST aggregates TEST/TEST1/TEST2)
        labels = []
        handles = []
        entries = [
            ("train", "#000000", (split == "TRAIN")),
            ("test", "#d62728", (split == "TEST") | (split == "TEST1") | (split == "TEST2")),
        ]
        if include_val:
            entries.insert(1, ("val", "#FDB515", (split == "VAL")))
        for lab, color, base_mask in entries:
            mask = base_mask
            if np.any(mask):
                r2s, maes = _metrics(y_true[mask], y_pred[mask])
                labels.append((lab, r2s, maes, color))
        # Build legend entries with metrics
        for lab, r2s, maes, color in labels:
            lbl = f"{lab:<5} R²={r2s:.3f}   MAE={maes:.3f}"
            handles.append(Line2D([0], [0], marker='s', linestyle='None', markerfacecolor=color, markeredgecolor=color, markersize=4, label=lbl))
        leg_props = fm.FontProperties(family=font_family) if font_family is not None else None
        ax.legend(
            handles=handles,
            loc='lower right',
            # fontsize=4,
            prop=leg_props,
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
    # Also save an alternate filename to avoid viewer caching issues
    # Single output only
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Missing predictions: {PREDICTIONS_CSV}")
        return
    set_gpg_style()
    df = pd.read_csv(PREDICTIONS_CSV)
    df = df[df["y_true"].notna()].copy()
    r2, mae, rmse = load_metrics(METRICS_JSON)
    # Plot baseline (TRAIN/TEST only)
    parity_plot(df, out_path=PLOT_95, r2=r2, mae=mae, rmse=rmse, font_family=None, include_val=False)

    # If a VAL-augmented CSV exists, produce a second figure explicitly
    alt_csv = os.path.join(os.path.dirname(PREDICTIONS_CSV), "ensemble_all6_predictions_full_with_val.csv")
    if os.path.exists(alt_csv):
        df2 = pd.read_csv(alt_csv)
        df2 = df2[df2["y_true"].notna()].copy()
        parity_plot(df2, out_path=PLOT_95_WITH_VAL, r2=r2, mae=mae, rmse=rmse, font_family=None, include_val=True)


if __name__ == "__main__":
    main()


