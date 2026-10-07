"""Figures, in the style used by the rest of the PRISM manuscript.

Panel geometry, tick direction, marker shape and the grey/teal/blue series
colours follow the existing figures so that a reader turning from the paper to
the supporting information does not have to re-learn how to read a plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from . import metrics

#: label -> (observed, predicted); ordered, because draw order is meaningful
OrderedSplits = Mapping[str, Tuple[Sequence[float], Sequence[float]]]
#: title -> (truth, predicted)
PanelSpec = Mapping[str, Tuple[Sequence[int], Sequence[int]]]

#: Series colours carried over from the manuscript figures.
GREY, TEAL, BLUE = "#D0D0D0", "#8AA2A2", "#6380A6"
#: Darker companions, used where a panel needs its own ramp.
SLATE, NAVY, JADE = "#829292", "#285A92", "#48988F"
GRID = "#ececec"
#: Resolution of saved figures; scripts may raise it (e.g. holdout.py --dpi).
SAVE_DPI = 300
#: Marker areas. The manuscript's parity figures use these rather than the
#: smaller defaults, which disappear at print size.
MARKER_FIRST, MARKER_REST = 18, 20
FOLD_COLOUR = {"acyl": TEAL, "amine": BLUE, "pair": GREY}
FOLD_LABEL = {"acyl": "unseen acyl", "amine": "unseen amine", "pair": "both unseen"}
#: Saturated, colour-blind-safe regime colours (Okabe-Ito) for plots where the
#: muted manuscript palette above is too faint to tell the regimes apart.
REGIME_STRONG = {"acyl": "#009E73", "amine": "#0072B2", "pair": "#E69F00"}


def set_plot_style() -> None:
    plt.rcParams.update({
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
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": False,
        "legend.frameon": False,
        "font.size": 9,
    })


def _save(fig: plt.Figure, path: Path, auto_layout: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if auto_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_parity(observed: Sequence[float], predicted: Sequence[float],
                fold_kind: Sequence[str], output_path: Path,
                xlabel: str = r"Observed log$_{10}$ k",
                ylabel: str = r"Predicted log$_{10}$ k") -> Path:
    """Cross-validated rate constants against measured values.

    Every point is out of fold: the reactant it belongs to was absent from the
    model that predicted it.
    """
    set_plot_style()
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    fold_kind = np.asarray(fold_kind)

    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0.1)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, zorder=1.5)

    handles = []
    for kind in ("pair", "acyl", "amine"):
        mask = fold_kind == kind
        if not mask.any():
            continue
        ax.scatter(observed[mask], predicted[mask], s=MARKER_REST, marker="s",
                   facecolors=FOLD_COLOUR[kind], edgecolors=FOLD_COLOUR[kind],
                   linewidths=0.5, alpha=0.8, zorder=3.0)
        handles.append(Line2D(
            [0], [0], marker="s", linestyle="None",
            markerfacecolor=FOLD_COLOUR[kind], markeredgecolor=FOLD_COLOUR[kind],
            markersize=4,
            label=f"{FOLD_LABEL[kind]}  R²={metrics.r2(observed[mask], predicted[mask]):.2f}"
                  f"   MAE={metrics.mae(observed[mask], predicted[mask]):.2f}"))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0.25,
              borderpad=0.2, markerscale=0.8, handlelength=0.9, fontsize=7.5)
    return _save(fig, output_path)


def plot_confusion(truth: Sequence[int], predicted: Sequence[int],
                   output_path: Path,
                   labels: Sequence[str] = ("un-reactive", "reactive"),
                   note: Optional[str] = None) -> Path:
    """Cross-validated reactivity calls as a confusion matrix.

    ``note`` is printed under the matrix. Use it when a row or column is empty
    by construction, so an all-zero row reads as "this set contains no such
    case" rather than as a model that never makes that call.
    """
    set_plot_style()
    truth = np.asarray(truth, int)
    predicted = np.asarray(predicted, int)
    cm = np.array([[int(((truth == t) & (predicted == p)).sum())
                    for p in (0, 1)] for t in (0, 1)])

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    # Annotate in whichever ink stays legible on that cell.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1], labels=[f"predicted\n{l}" for l in labels])
    ax.set_yticks([0, 1], labels=[f"actual\n{l}" for l in labels], rotation=90,
                  va="center", ha="center", multialignment="center")
    ax.tick_params(axis="y", pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    report = metrics.classification_report(truth, predicted)
    ax.set_title(f"accuracy {report['accuracy']:.2f}   "
                 f"(majority {report['majority']:.2f})", fontsize=9, pad=10)
    if note:
        ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.5, color="0.35", wrap=True)
    return _save(fig, output_path)


def plot_candidate_ranking(frame: pd.DataFrame, output_path: Path,
                           highlight: Optional[str] = None) -> Path:
    """Cross-validated score for each feature set, best at the top."""
    set_plot_style()
    ranked = frame.sort_values("joint")
    n = len(ranked)
    fig, ax = plt.subplots(figsize=(5.6, 0.55 * n + 1.5))
    y = np.arange(n)
    colours = [BLUE if name == highlight else GREY for name in ranked.name]
    ax.barh(y, ranked.joint, color=colours, height=0.62, zorder=3)
    ax.set_yticks(y, labels=[f"{l}\n{c}" for l, c in zip(ranked.label, ranked.cost)],
                  fontsize=8)
    # The bars are truncated at `lo`, so every annotation has to sit inside the
    # visible axis: anything outside it is dragged back in by bbox_inches and
    # pads the saved figure with empty space. Only the score is labelled —
    # its two components are shown in the regime figure and the results table,
    # and putting them inside the bars collides on the shortest one.
    lo = max(0.0, float(ranked.joint.min()) - 0.04)
    hi = float(ranked.joint.max()) + 0.022
    span = hi - lo
    ax.set_xlim(lo, hi)
    for i, value in enumerate(ranked.joint):
        ax.text(value + 0.018 * span, i, f"{value:.3f}", va="center",
                fontsize=8.5, color="#333333", zorder=4)
    ax.set_xlabel("cross-validated score  (½ accuracy + ½ R²)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0.1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # Long two-line tick labels need the room reserved explicitly; tight_layout
    # collapses the axes when it tries to fit them itself.
    fig.subplots_adjust(left=0.30, right=0.97, top=0.96, bottom=0.20)
    return _save(fig, output_path, auto_layout=False)


def plot_regime_breakdown(frame: pd.DataFrame, output_path: Path,
                          highlight: Optional[str] = None) -> Path:
    """Accuracy and rate-constant error per set, split by which reactant is unseen."""
    set_plot_style()
    ranked = frame.sort_values("joint", ascending=False)
    n = len(ranked)
    x = np.arange(n)
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    axes[0].bar(x - width / 2, ranked.accuracy_acyl, width, color=TEAL,
                label=FOLD_LABEL["acyl"], zorder=3)
    axes[0].bar(x + width / 2, ranked.accuracy_amine, width, color=BLUE,
                label=FOLD_LABEL["amine"], zorder=3)
    axes[0].set_ylabel("classification accuracy")
    axes[0].set_ylim(0.5, max(0.80, float(ranked[["accuracy_acyl", "accuracy_amine"]].max().max()) + 0.03))

    axes[1].bar(x - width / 2, ranked.mae_acyl, width, color=TEAL, zorder=3)
    axes[1].bar(x + width / 2, ranked.mae_amine, width, color=BLUE, zorder=3)
    axes[1].set_ylabel(r"rate constant MAE (log$_{10}$ k)")

    for ax in axes:
        ax.set_xticks(x, labels=ranked.label, rotation=30, ha="right", fontsize=8)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0.1)
        ax.spines[["top", "right"]].set_visible(False)
        if highlight is not None:
            for tick, name in zip(ax.get_xticklabels(), ranked.name):
                if name == highlight:
                    tick.set_fontweight("bold")
    axes[0].legend(loc="upper right", fontsize=7.5, frameon=False)
    return _save(fig, output_path)


def plot_parity_splits(splits: OrderedSplits, output_path: Path,
                       xlabel: str = r"True log$_{10}$ k",
                       ylabel: str = r"Predicted log$_{10}$ k") -> Path:
    """Parity for several named sets on one panel, with R2 and MAE per set.

    ``splits`` is an ordered mapping of label -> (observed, predicted). The
    first is drawn faintest and smallest, on the assumption that it is the
    largest and least interesting set; later ones sit on top.
    """
    set_plot_style()
    colours = [GREY, TEAL, BLUE, SLATE, NAVY]
    trend = ["#B0B0B0", "#455C5C", "#6380A6", "#829292", "#285A92"]
    items = list(splits.items())

    all_true = np.concatenate([np.asarray(v[0], float) for v in splits.values()])
    all_pred = np.concatenate([np.asarray(v[1], float) for v in splits.values()])
    lo, hi = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots()
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, zorder=1.5)

    # Colour follows the caller's order, so the legend stays stable; draw order
    # is by decreasing size, so the densest set does not bury the others. The
    # cross-validated set is much the largest — a reaction appears once per fold
    # that holds it out — and without this it hides everything beneath it.
    handles = [None] * len(items)
    order = sorted(range(len(items)), key=lambda k: -len(items[k][1][0]))
    for i in order:
        label, (obs, pred) = items[i]
        obs, pred = np.asarray(obs, float), np.asarray(pred, float)
        colour = colours[i % len(colours)]
        if len(obs) > 1:
            slope, intercept = np.polyfit(obs, pred, 1)
            xs = np.linspace(lo, hi, 200)
            ax.plot(xs, slope * xs + intercept, color=trend[i % len(trend)],
                    linewidth=0.8, alpha=0.25 if i == 0 else 0.4, zorder=1.0)
        ax.scatter(obs, pred, s=MARKER_FIRST if i == 0 else MARKER_REST, marker="s",
                   facecolors=colour, edgecolors=colour, linewidths=0.5,
                   alpha=0.7 if i == 0 else 0.8, zorder=3.0)
        # Indexed, not appended: the legend keeps the caller's order even
        # though the marks were drawn largest-set-first.
        handles[i] = Line2D(
            [0], [0], marker="s", linestyle="None", markerfacecolor=colour,
            markeredgecolor=colour, markersize=4,
            label=f"{label:<6s} R\u00b2={metrics.r2(obs, pred):.2f}   "
                  f"MAE={metrics.mae(obs, pred):.2f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0.1)
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0.25,
              borderpad=0.2, markerscale=0.8, handlelength=0.9, fontsize=7.5)
    return _save(fig, output_path)


def residual_outliers(observed: Sequence[float], predicted: Sequence[float],
                      n_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals (measured - predicted) and the points :func:`plot_residuals`
    marks as outliers: more than ``n_sigma`` SD from the mean residual."""
    residual = np.asarray(observed, float) - np.asarray(predicted, float)
    mean, sd = float(np.mean(residual)), float(np.std(residual))
    return residual, np.abs(residual - mean) > n_sigma * sd


def plot_residuals(observed: Sequence[float], predicted: Sequence[float],
                   output_path: Path, names: Optional[Sequence[str]] = None,
                   n_sigma: float = 2.0, colour: str = NAVY,
                   kinds: Optional[Sequence[str]] = None) -> Path:
    """Residuals against the measured value, and their distribution.

    Points beyond ``n_sigma`` standard deviations from the mean residual are
    marked and, where ``names`` is given, labelled. The intent is to see
    whether the error is structured — growing with rate, say — or whether a
    few identifiable reactions carry it.

    With ``kinds`` ("acyl" / "amine" / "pair", as in :func:`plot_parity`) each
    point is coloured by which reactant was unseen, so a regime that carries
    the error stands out. Outliers are then ringed rather than recoloured, so
    they keep their regime colour.
    """
    set_plot_style()
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    residual, outlier = residual_outliers(observed, predicted, n_sigma)
    mean, sd = float(np.mean(residual)), float(np.std(residual))
    threshold = n_sigma * sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    if kinds is None:
        ax1.scatter(observed[~outlier], residual[~outlier], marker="s", s=MARKER_REST,
                    color=colour, alpha=0.7, edgecolor="k", linewidth=0.5)
        ax1.scatter(observed[outlier], residual[outlier], marker="s", s=MARKER_REST,
                    color="#AA183B", alpha=0.8, edgecolor="k", linewidth=0.5,
                    label=f"outliers (>{n_sigma:g}\u03c3)")
    else:
        kinds = np.asarray(kinds)
        for kind in ("acyl", "amine", "pair"):
            m = kinds == kind
            if m.any():
                ax1.scatter(observed[m], residual[m], marker="s", s=MARKER_REST,
                            color=REGIME_STRONG[kind], alpha=0.95, edgecolor="k",
                            linewidth=0.5, zorder=2,
                            label=f"{FOLD_LABEL[kind]} (n={int(m.sum())})")
        if outlier.any():
            ax1.scatter(observed[outlier], residual[outlier], marker="s",
                        s=MARKER_REST * 2.6, facecolors="none", edgecolors="#AA183B",
                        linewidth=1.3, zorder=3,
                        label=f"outliers (>{n_sigma:g}\u03c3)")
    ax1.axhline(mean + threshold, color="b", linestyle="--", alpha=0.3)
    ax1.axhline(mean - threshold, color="b", linestyle="--", alpha=0.3,
                label=f"\u00b1{n_sigma:g}\u03c3")
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    if names is not None:
        # Outliers cluster, so labels are staggered and alternate side; placing
        # them all at one offset overlaps them into unreadability.
        marked = sorted(zip(observed[outlier], residual[outlier],
                            np.asarray(names)[outlier]), key=lambda t: t[0])
        for j, (x, r, nm) in enumerate(marked):
            above = r < 0
            ax1.annotate(str(nm), (x, r), fontsize=6.5,
                         textcoords="offset points",
                         xytext=(0, (12 if above else -12) - (j % 2) * 9),
                         ha="center", va="bottom" if above else "top",
                         color="#6E1026")
    ax1.set_xlabel(r"Measured log$_{10}$ k")
    ax1.set_ylabel("Residual (measured \u2212 predicted)")
    ax1.set_title(f"Residuals\n{int(outlier.sum())} of {len(residual)} beyond "
                  f"{n_sigma:g}\u03c3", fontsize=10)
    ax1.margins(y=0.16)
    # Below the panel, not inside it: residuals fill the whole plot area, so
    # any in-axes position eventually sits on top of points.
    ax1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17),
               ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
    ax1.set_axisbelow(True)
    ax1.grid(True, color=GRID, linewidth=0.6)

    ax2.hist(residual, bins=20, color=colour, alpha=0.7, edgecolor="k", linewidth=0.5)
    ax2.axvline(mean, color="#AA183B", linestyle="--", alpha=0.8,
                label=f"mean {mean:+.3f}")
    ax2.axvline(mean + sd, color="orange", linestyle="--", alpha=0.6,
                label=f"\u00b11\u03c3  {sd:.3f}")
    ax2.axvline(mean - sd, color="orange", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Residual (measured \u2212 predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of residuals", fontsize=10)
    ax2.legend(fontsize=7.5)
    ax2.set_axisbelow(True)
    ax2.grid(True, color=GRID, linewidth=0.6)
    return _save(fig, output_path)


def plot_confusion_panels(panels: PanelSpec, output_path: Path,
                          labels: Sequence[str] = ("un-reactive", "reactive"),
                          xlabel: str = "Predicted class",
                          ylabel: str = "Measured class") -> Path:
    """One confusion matrix per panel, sharing a single x-axis label.

    ``panels`` is an ordered mapping of title -> (truth, predicted). Each gets
    its own white-to-colour ramp so panels stay distinguishable in print, and
    carries its accuracy in the title.
    """
    set_plot_style()
    colours = [SLATE, NAVY, JADE, "#AA183B"]
    items = list(panels.items())
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.5))
    if n == 1:
        axes = [axes]

    for i, (title, (truth, predicted)) in enumerate(items):
        truth = np.asarray(truth, int)
        predicted = np.asarray(predicted, int)
        cm = np.array([[int(((truth == t) & (predicted == p)).sum())
                        for p in (0, 1)] for t in (0, 1)])
        cmap = LinearSegmentedColormap.from_list(f"panel{i}", ["white",
                                                               colours[i % len(colours)]])
        ax = axes[i]
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]:d}", ha="center", va="center", fontsize=11,
                        color="white" if cm[r, c] > cm.max() * 0.6 else "black")
        ax.set_xticks([0, 1], labels=labels, fontsize=8)
        accuracy = float((truth == predicted).mean()) if len(truth) else float("nan")
        ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}", fontsize=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i == 0:
            ax.set_yticks([0, 1], labels=labels, fontsize=8, rotation=90, va="center")
            ax.set_ylabel(ylabel, fontsize=9)
        else:
            ax.set_yticks([0, 1], labels=["", ""])
            ax.tick_params(left=False, labelleft=False)

    fig.text(0.5, -0.04, xlabel, ha="center", fontsize=9)
    return _save(fig, output_path)


def plot_nmr_comparison_linear(nmr_k: Sequence[float], prism_k: Sequence[float],
                               output_path: Path, low_confidence: Sequence[bool] = None,
                               low_confidence_label: str = "low-confidence PRISM fit",
                               title: str = "Comparison of PRISM and NMR rate constants",
                               figsize: tuple = (5.2, 4.4)) -> Path:
    """The same comparison as :func:`plot_nmr_comparison`, on the rate
    constants themselves rather than their logarithms. Fit and R\u00b2 are on k."""
    set_plot_style()
    nmr = np.asarray(nmr_k, dtype=float)
    prism = np.asarray(prism_k, dtype=float)
    finite = np.isfinite(nmr) & np.isfinite(prism)
    low = (np.zeros(len(nmr), dtype=bool) if low_confidence is None
           else np.asarray(low_confidence, dtype=bool))
    faint, solid = finite & low, finite & ~low

    fig, ax = plt.subplots(figsize=figsize)
    hi = float(max(nmr[finite].max(), prism[finite].max())) * 1.05
    limits = (0.0, hi)
    ax.plot(limits, limits, color=GREY, linestyle="--", linewidth=1.2, zorder=1,
            label="identity (y = x)")
    if faint.any():
        ax.scatter(nmr[faint], prism[faint], s=50, marker="s", alpha=0.25, color=BLUE,
                   edgecolor="black", linewidth=0.6, zorder=2, label=low_confidence_label)
    ax.scatter(nmr[solid], prism[solid], s=50, marker="s", alpha=0.85, color=BLUE,
               edgecolor="black", linewidth=0.6, zorder=3)
    if finite.sum() >= 2:
        slope, intercept = np.polyfit(nmr[finite], prism[finite], 1)
        r_squared = np.corrcoef(nmr[finite], prism[finite])[0, 1] ** 2
        line_x = np.linspace(0.0, nmr[finite].max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color=GREY, alpha=0.55, linewidth=2,
                zorder=1.5, label=f"fit: y = {slope:.2f}x + {intercept:.1f}\n"
                                  f"R\u00b2 = {r_squared:.2f}, n = {int(finite.sum())}")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"NMR $k$")
    ax.set_ylabel(r"PRISM $k$")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7.5)
    return _save(fig, output_path)


#: Train / test series for the chemical-space figures (Okabe-Ito, colour-blind safe).
SPLIT_COLOUR = {"all pairs": "#6F6F6F", "train-train": "#0072B2", "test-test": "#E69F00",
                "test-train": "#009E73"}
#: Similarity ramp of the original chemical-space figures (yellow-orange-red).
SIMILARITY_CMAP = "YlOrRd"


def plot_similarity_heatmap(matrix: np.ndarray, row_labels: Sequence[str],
                            col_labels: Sequence[str], output_path: Path, title: str,
                            row_split: int = None, col_split: int = None,
                            row_test: Sequence[bool] = None,
                            col_test: Sequence[bool] = None,
                            xlabel: str = "", ylabel: str = "") -> Path:
    """Tanimoto similarity matrix. Optional split lines separate the training
    block from the test block; test reactants are labelled in bold blue."""
    n_rows, n_cols = matrix.shape
    cell = 0.16
    fig, ax = plt.subplots(figsize=(max(3.0, n_cols * cell + 1.6),
                                    max(1.6, n_rows * cell + 0.9)))
    image = ax.imshow(matrix, cmap=SIMILARITY_CMAP, vmin=0, vmax=1, aspect="equal",
                      interpolation="nearest")
    for split, axis in ((row_split, "h"), (col_split, "v")):
        if split:
            (ax.axhline if axis == "h" else ax.axvline)(split - 0.5, color="black",
                                                        linewidth=1.4)
    fs = max(4.5, min(7.0, 140 / max(n_rows, n_cols)))
    ax.set_xticks(range(n_cols), labels=col_labels, rotation=90, fontsize=fs)
    ax.set_yticks(range(n_rows), labels=row_labels, fontsize=fs)
    ax.tick_params(length=0)
    for ticks, flags in ((ax.get_xticklabels(), col_test), (ax.get_yticklabels(), row_test)):
        for tick, is_test in zip(ticks, flags if flags is not None else []):
            if is_test:
                tick.set_color("#0072B2")
                tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Tanimoto similarity")
    bar.outline.set_visible(False)
    return _save(fig, output_path)


def plot_similarity_distributions(panels: Mapping[str, Mapping[str, Sequence[float]]],
                                  output_path: Path) -> Path:
    """Pairwise similarity distributions, one panel per reactant kind.

    `panels` maps a panel title to {series label: similarities}. Series are
    normalised to densities so sets of different size compare directly; the
    dashed line marks each series' mean.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.9), sharey=False)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 1, 26)
    for ax, (title, series) in zip(axes, panels.items()):
        for label, values in series.items():
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                continue
            colour = SPLIT_COLOUR.get(label, BLUE)
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.25,
                    color=colour)
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.4,
                    color=colour, label=f"{label} (n = {values.size}, mean {values.mean():.2f})")
            ax.axvline(values.mean(), color=colour, linestyle="--", linewidth=1.1)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Tanimoto similarity")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    axes[0].set_ylabel("density")
    return _save(fig, output_path)


def plot_diversity_distribution(series: Mapping[str, Sequence[float]],
                                colours: Mapping[str, str], output_path: Path) -> Path:
    """Pairwise Tanimoto diversity (1 - similarity), one overlaid histogram per
    reactant kind, normalised to densities; the dashed line marks each mean."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bins = np.linspace(0, 1, 26)
    for label, values in series.items():
        values = np.asarray(values, dtype=float)
        colour = colours.get(label, BLUE)
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.25,
                color=colour)
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.4,
                color=colour)
        ax.axvline(values.mean(), color=colour, linestyle="--", linewidth=1.2)
        ax.plot([], [], color=colour, linewidth=1.4,
                label=f"{label} (n = {values.size} pairs)\nmean {values.mean():.3f} "
                      f"\u00b1 {values.std():.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pairwise Tanimoto diversity (1 \u2212 similarity)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7.5, loc="upper left")
    return _save(fig, output_path)


def plot_nearest_training(panels: Mapping[str, pd.DataFrame], output_path: Path) -> Path:
    """Similarity of each reactant to its nearest training reactant.

    Training reactants (measured against the rest of training) form the
    reference strip; test reactants are labelled points against it. A test
    reactant inside the training spread is as well covered as a typical
    training reactant; one below it is structurally novel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, (title, table) in zip(axes, panels.items()):
        for x, split, colour in ((0, "train", SPLIT_COLOUR["train-train"]),
                                 (1, "test", "#E69F00")):
            rows = table[table.split == split]
            y = rows.nearest_train_similarity.to_numpy(float)
            jitter = rng.uniform(-0.12, 0.12, y.size) if split == "train" else np.zeros(y.size)
            ax.scatter(x + jitter, y, s=MARKER_REST, color=colour, edgecolor="white",
                       linewidth=0.5, zorder=3)
            if split == "train" and y.size:
                ax.hlines(np.median(y), x - 0.25, x + 0.25, color="black", linewidth=1.2,
                          zorder=4)
            if split == "test":
                for value, rid in zip(y, rows.id):
                    ax.annotate(str(rid), (x, value), xytext=(6, 0),
                                textcoords="offset points", va="center", fontsize=7)
        ax.set_xticks([0, 1], labels=["train\n(vs rest of train)", "test\n(vs train)"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].set_ylabel("Tanimoto similarity to nearest\ntraining reactant")
    return _save(fig, output_path)


def plot_nmr_comparison(nmr_logk: Sequence[float], prism_logk: Sequence[float],
                        output_path: Path, low_confidence: Sequence[bool] = None,
                        title: str = "Comparison of PRISM and NMR rate constants",
                        low_confidence_label: str = "low-confidence PRISM fit",
                        figsize: tuple = (5.2, 4.4)) -> Path:
    """Compare two independent measurements of the same rate constants.

    Every pair measured by both methods is plotted and every pair is fitted.
    Points flagged by ``low_confidence`` are drawn more transparently, so the
    less certain plate-reader fits read as fainter without being hidden or
    silently dropped from the regression.

    The identity line is drawn because this is a method comparison, not a
    prediction: agreement means falling on y = x, and a slope away from 1 is a
    rate-dependent disagreement that a regression line alone would not reveal.
    """
    set_plot_style()
    nmr = np.asarray(nmr_logk, dtype=float)
    prism = np.asarray(prism_logk, dtype=float)
    finite = np.isfinite(nmr) & np.isfinite(prism)
    low = (np.zeros(len(nmr), dtype=bool) if low_confidence is None
           else np.asarray(low_confidence, dtype=bool))
    faint = finite & low
    solid = finite & ~low

    fig, ax = plt.subplots(figsize=figsize)

    lo = float(min(nmr[finite].min(), prism[finite].min()))
    hi = float(max(nmr[finite].max(), prism[finite].max()))
    pad = 0.08 * (hi - lo)
    limits = (lo - pad, hi + pad)
    ax.plot(limits, limits, color=GREY, linestyle="--", linewidth=1.2,
            zorder=1, label="identity (y = x)")

    if faint.any():
        ax.scatter(nmr[faint], prism[faint], s=50, marker="s", alpha=0.25,
                   color=BLUE, edgecolor="black", linewidth=0.6, zorder=2,
                   label=low_confidence_label)
    ax.scatter(nmr[solid], prism[solid], s=50, marker="s", alpha=0.85,
               color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)

    if finite.sum() >= 2:
        slope, intercept = np.polyfit(nmr[finite], prism[finite], 1)
        r_squared = np.corrcoef(nmr[finite], prism[finite])[0, 1] ** 2
        line_x = np.linspace(nmr[finite].min(), nmr[finite].max(), 100)
        # Above the identity line (zorder 1), behind every point (2 and 3).
        ax.plot(line_x, slope * line_x + intercept, color=GREY, alpha=0.55,
                linewidth=2, zorder=1.5,
                label=f"fit: y = {slope:.2f}x + {intercept:.2f}\n"
                      f"R\u00b2 = {r_squared:.2f}, n = {int(finite.sum())}")

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"NMR log$_{10}$($k$)")
    ax.set_ylabel(r"PRISM log$_{10}$($k$)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)
    return _save(fig, output_path)


#: Series colours for the Hammett comparison, one per reactant varied.
HAMMETT_COLOUR = {"amine": "#6079AC", "acyl": "#19AB87"}


def _chem_label(text) -> str:
    """Substituent label for a plot: digits as subscripts (CF3 -> CF$_3$) and
    a locant t / p before a capital in italics (tBu, OPhpF)."""
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r"([tp])(?=[A-Z])", r"$\\mathit{\1}$", text)
    return re.sub(r"(\d+)", r"$_{\1}$", text)


#: Label positions tried, nearest first (offsets in points). Beyond the first
#: two, a leader line ties the label to its point.
_LABEL_OFFSETS = [(0, 7), (0, -7), (10, 9), (-10, 9), (10, -9), (-10, -9),
                  (0, 17), (0, -17), (20, 12), (-20, 12), (20, -12), (-20, -12),
                  (0, 27), (0, -27), (28, 0), (-28, 0), (24, 24), (-24, 24),
                  (24, -24), (-24, -24), (0, 38), (0, -38)]


def _place_labels(ax, xs, ys, texts, fontsize: float = 7) -> None:
    """Put each label at the nearest offset that overlaps no other label and
    no marker; labels moved away from their point get a leader line."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    to_px = fig.dpi / 72.0
    pts = ax.transData.transform(np.c_[xs, ys])
    half = 5.0 * to_px                      # half the marker width, in pixels
    from matplotlib.transforms import Bbox
    markers = [Bbox.from_extents(x - half, y - half, x + half, y + half) for x, y in pts]
    frame = ax.get_window_extent(renderer)
    placed = []
    for k in np.argsort(xs, kind="stable"):
        chosen = None
        for n, (dx, dy) in enumerate(_LABEL_OFFSETS):
            ann = ax.annotate(
                texts[k], (xs[k], ys[k]), fontsize=fontsize, zorder=5,
                textcoords="offset points", xytext=(dx, dy),
                ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                va="center" if dy == 0 else ("bottom" if dy > 0 else "top"),
                arrowprops=(None if n < 2 else
                            dict(arrowstyle="-", color="0.35", lw=0.6,
                                 shrinkA=0.5, shrinkB=3.5)))
            box = ann.get_window_extent(renderer).padded(1.0 * to_px)
            clear = (frame.containsx(box.x0) and frame.containsx(box.x1)
                     and frame.containsy(box.y0) and frame.containsy(box.y1)
                     and not any(box.overlaps(b) for b in placed + markers))
            if clear or n == len(_LABEL_OFFSETS) - 1:
                chosen = box
                break
            ann.remove()
        placed.append(chosen)


def plot_hammett_panels(frame: pd.DataFrame, output_path: Path,
                        series: str, columns: Mapping[str, str],
                        reference: str = "F",
                        low_confidence: Optional[Sequence[bool]] = None,
                        low_confidence_panels: Sequence[str] = ("PRISM",),
                        figsize: tuple = None,
                        labels: Optional[Sequence[str]] = None) -> Path:
    """Stacked Hammett plots, one panel per method, on a shared y scale.

    ``columns`` maps a panel title to the column holding its log rate ratio.
    Every panel is fitted on all of its finite points; the slope is the
    reaction constant rho, printed in the panel title. Points flagged by
    ``low_confidence`` are drawn more transparently rather than dropped, so a
    less certain rate is visible as fainter without being hidden from the fit.

    The flag describes the plate-reader fit, so it is applied only to the
    panels named in ``low_confidence_panels``. Fading an NMR or a computed
    point because the *plate reader* was unsure about that substrate would
    attribute an uncertainty to the wrong measurement.

    A shared y scale across panels is deliberate: the panels are compared to
    each other, and independent scales would make two very different slopes
    look alike.
    """
    set_plot_style()
    colour = HAMMETT_COLOUR.get(series, BLUE)
    sigma = pd.to_numeric(frame["Hammett value"], errors="coerce").to_numpy(float)
    low = (np.zeros(len(frame), dtype=bool) if low_confidence is None
           else np.asarray(low_confidence, dtype=bool))

    stacked = [pd.to_numeric(frame[c], errors="coerce") for c in columns.values()]
    finite_all = np.concatenate([s.to_numpy(float) for s in stacked])
    finite_all = finite_all[np.isfinite(finite_all)]
    if not finite_all.size:
        raise ValueError("no finite rate ratios to plot")
    margin = 0.1 * (finite_all.max() - finite_all.min())
    limits = (finite_all.min() - margin, finite_all.max() + margin)

    figsize = figsize or (4.2, 3.2 * len(columns))
    fig, axes = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    to_label = []

    for ax, (label, column) in zip(axes, columns.items()):
        y = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        ok = np.isfinite(sigma) & np.isfinite(y)
        panel_low = low if label in low_confidence_panels else np.zeros_like(low)
        faint, solid = ok & panel_low, ok & ~panel_low
        if faint.any():
            ax.scatter(sigma[faint], y[faint], s=60, marker="s", alpha=0.3,
                       color=colour, edgecolor="black", linewidth=0.6, zorder=2)
        ax.scatter(sigma[solid], y[solid], s=60, marker="s", alpha=0.9,
                   color=colour, edgecolor="black", linewidth=0.6, zorder=3)

        title = label
        if ok.sum() >= 2:
            rho, intercept = np.polyfit(sigma[ok], y[ok], 1)
            r_squared = np.corrcoef(sigma[ok], y[ok])[0, 1] ** 2
            line_x = np.linspace(sigma[ok].min(), sigma[ok].max(), 100)
            ax.plot(line_x, rho * line_x + intercept, color=colour,
                    linestyle="--", alpha=0.75, linewidth=1.8, zorder=4,
                    label=f"R² = {r_squared:.2f}, n = {int(ok.sum())}")
            title = f"{label}: ρ = {rho:.1f}"
            ax.legend(loc="best", fontsize=7.5)
        if labels is not None:
            to_label.append((ax, sigma[ok], y[ok],
                             [_chem_label(labels[i]) for i in np.flatnonzero(ok)]))
        ax.set_title(title)
        ax.set_ylabel(rf"log($k_\mathrm{{R}}$/$k_\mathrm{{{reference}}}$)")
        ax.set_ylim(*limits)
        ax.axhline(0.0, color=GREY, linewidth=0.8, alpha=0.6, zorder=1)

    axes[-1].set_xlabel(r"$\sigma_{\mathrm{P}}$")
    if not to_label:
        return _save(fig, output_path)
    # Labels are placed in final screen coordinates, so lay the figure out first.
    fig.tight_layout()
    fig.canvas.draw()
    for ax, xs, ys, texts in to_label:
        _place_labels(ax, xs, ys, texts)
    return _save(fig, output_path, auto_layout=False)


def plot_drug_scope(frame: pd.DataFrame, output_path: Path,
                    figsize: tuple = (6.2, 4.6)) -> Path:
    """Predicted rate constants for the drug scope, one row per reaction.

    With eleven reactions and two measured rate constants, a parity plot would
    imply a regression that cannot be fitted. A row per reaction shows each
    prediction beside whatever was measured for it, and makes the censoring
    visible: an `instant` pair has no plate-reader value because its rate
    constant was above the measurable window, so its true position is to the
    right of anything shown.
    """
    set_plot_style()
    frame = frame.reset_index(drop=True)
    order = frame.sort_values(["rate_class", "predicted_logk"]).index
    frame = frame.loc[order].reset_index(drop=True)
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=figsize)
    for i, row in frame.iterrows():
        ax.plot([-4, 5], [i, i], color=GREY, linewidth=0.4, alpha=0.25, zorder=0)

    ax.scatter(frame.predicted_logk, y, s=64, marker="s", color=BLUE,
               edgecolor="black", linewidth=0.6, zorder=3, label="predicted")
    measured = frame.logk.notna()
    if measured.any():
        ax.scatter(frame.logk[measured], y[measured], s=64, marker="o",
                   color=TEAL, edgecolor="black", linewidth=0.6, zorder=3,
                   label="measured (plate reader)")
    nmr = frame.get("logk_nmr")
    if nmr is not None and nmr.notna().any():
        ok = nmr.notna()
        ax.scatter(nmr[ok], y[ok], s=52, marker="^", color="white",
                   edgecolor="black", linewidth=0.9, zorder=3,
                   label="measured (NMR)")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.reaction}  ({r.rate_class})" for r in frame.itertuples()],
                       fontsize=7.5)
    ax.set_xlabel(r"log$_{10}$($k$)")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_title("Drug scope: predicted and measured rate constants")
    ax.legend(loc="lower right", fontsize=7.5)
    return _save(fig, output_path)
