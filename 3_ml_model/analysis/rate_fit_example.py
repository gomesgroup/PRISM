#!/usr/bin/env python3
"""Example PRISM rate fit: 1/[A]0 against t1/2 for one reaction (Figure 2B).

    python analysis/rate_fit_example.py [--reaction rxn_15_03]

For A + B -> P with [A]0 = [B]0, the integrated second-order rate law is
1/[A] - 1/[A]0 = k t. At the half-life [A] = [A]0/2, so 1/[A]0 = k t1/2: the
slope of 1/[A]0 against t1/2 is the rate constant k.

Reads the well-level output of 2_rate_fitting (prism_perwell.py). Red: wells
that passed the per-well hydrolysis control and were fitted. Open grey: wells
with a reading that did not pass. Grey triangles: the no-amine control at the
same concentrations. The fit is the one prism_perwell.py makes, so the slope
equals the k_corrected reported for the reaction.

Writes, in --output-dir: rate_fit_<reaction>.png and rate_fit_<reaction>.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WELLS = REPO.parent / "2_rate_fitting" / "outputs" / "well_level_kept_excluded.csv"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reaction", default="rxn_15_03")
    p.add_argument("--wells", type=Path, default=WELLS)
    p.add_argument("--margin", default="1p5x")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis" / "results" / "rate_fit_example")
    args = p.parse_args()

    rates = pd.read_excel(config.RATES_XLSX)
    row = rates[rates.reaction == args.reaction]
    if len(row) != 1 or isinstance(row.batch.iloc[0], str):
        raise SystemExit(f"{args.reaction}: needs exactly one plate with a rate constant")
    row = row.iloc[0]
    wells = pd.read_csv(args.wells)
    w = wells[(wells.group == row.group) & (wells.acyl_chloride == row.acyl_chloride)
              & (wells.amine == row.amine)].copy()
    status = w["status_" + args.margin]
    w["inv_A0"] = w.stored_ordinate_2_over_A0 / 2          # 1/[A]0, M^-1
    kept = w[status == "kept"].sort_values("conc_mM", ascending=False)
    other = w[(status != "kept") & w.amine_time_min.gt(0) & w.inv_A0.notna()]
    ctrl = w[w.control_time_min.notna() & w.inv_A0.notna()]

    k, b = np.polyfit(kept.amine_time_min, kept.inv_A0, 1)
    pred = k * kept.amine_time_min + b
    r2 = 1 - ((kept.inv_A0 - pred) ** 2).sum() / ((kept.inv_A0 - kept.inv_A0.mean()) ** 2).sum()
    print(f"{args.reaction}: k = {k:.3f} M^-1 min^-1 (table: {row.k_corrected:.3f}), "
          f"intercept {b:.1f}, R2 {r2:.3f}, {len(kept)} wells")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(ctrl.control_time_min, ctrl.inv_A0, marker="^", s=40, color="0.6",
               label="no-amine control")
    if len(other):
        ax.scatter(other.amine_time_min, other.inv_A0, marker="s", s=50, facecolor="none",
                   edgecolor="0.4", label="excluded (failed control)")
    ax.scatter(kept.amine_time_min, kept.inv_A0, marker="s", s=60, color="#D7301F",
               zorder=3, label=f"R$^2$ = {r2:.3f}")
    for i, (x, y) in enumerate(zip(kept.amine_time_min, kept.inv_A0), start=1):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(-4, 8),
                    fontsize=10, fontweight="bold")
    x_line = np.linspace(0, kept.amine_time_min.max() * 1.05, 100)
    ax.plot(x_line, k * x_line + b, color="0.4", linewidth=1.5,
            label=f"y = {k:.2f}x {'+' if b >= 0 else '−'} {abs(b):.1f}")

    ax.set_xlim(0, max(kept.amine_time_min.max(), ctrl.control_time_min.max()) * 1.08)
    ax.set_ylim(0, kept.inv_A0.max() * 1.25)
    ax.set_xlabel(r"$t_{1/2}$ (min)")
    ax.set_ylabel(r"1/[A]$_0$ (M$^{-1}$)")
    ax.set_title(f"{row.acyl_chloride} (B{row.acyl_label}) + {row.amine} (A{row.amine_label})",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.03, 0.97,
            "rate = $k$[A][B],  [A]$_0$ = [B]$_0$\n"
            r"1/[A]$_0$ = $k\,t_{1/2}$" + "\n" + f"$k$ = {k:.1f} M$^{{-1}}$ min$^{{-1}}$",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / f"rate_fit_{args.reaction}.png", dpi=300)
    out = w[["conc_mM", "amine_time_min", "control_time_min", "inv_A0",
             "control_over_amine_ratio", "status_" + args.margin]].rename(
        columns={"inv_A0": "inv_A0_M-1", "status_" + args.margin: "status"})
    out.to_csv(args.output_dir / f"rate_fit_{args.reaction}.csv", index=False)
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
