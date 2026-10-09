#!/usr/bin/env python3
"""Summary of the modeling data: class counts and the rate constants.

    python analysis/data_summary.py

Classification: reactions in each class (measurable / instant / slow, and
reactive / un-reactive), for all data, the training pool and each held-out
regime, plus the excluded substrate scope and the drug scope.

Regression: the measured log10 k values (count, mean, SD, range, percentiles),
for all data and for the training pool and the held-out test set, with the
PRISM confidence tiers.

Outputs, in --output-dir:

    class_counts.csv          reactions per class, per set
    reactant_counts.csv       per reactant: reactions in each class and its split
    logk_summary.csv          log10 k statistics, per set
    data_summary.txt          the same, as text
    figures/
      class_counts.png        reactions per class, per set
      logk_distribution.png   measured log10 k, training pool vs held-out test
      reaction_class_map.png  class of every acyl chloride x amine pair
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config, data, holdout, plotting  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (after plotting sets the backend)

CLASSES = ("measurable", "instant", "slow")
CLASS_LABEL = {"measurable": "measurable (rate constant)", "instant": "too fast",
               "slow": "un-reactive"}
CLASS_COLOUR = {"measurable": plotting.NAVY, "instant": plotting.TEAL,
                "slow": plotting.GREY}
SETS = (("all", "All data"), ("pool", "Training pool"), ("test", "Held-out test"),
        ("new_acyl", "Unseen acyl chloride"), ("new_amine", "Unseen amine"),
        ("both_new", "Both unseen"))


def class_table(table: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    rows = []
    for key, label in SETS:
        if key == "all":
            sub = table
        elif key == "test":
            sub = table[regime != "pool"]
        else:
            sub = table[regime == key]
        counts = sub.rate_class.value_counts()
        row = {"set": label, "reactions": len(sub),
               "acyl_chlorides": sub.acyl.nunique(), "amines": sub.amine.nunique()}
        row.update({c: int(counts.get(c, 0)) for c in CLASSES})
        row["reactive"] = row["measurable"] + row["instant"]
        row["un-reactive"] = row["slow"]
        row["fraction_reactive"] = round(row["reactive"] / len(sub), 3) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def logk_stats(values: pd.Series, label: str) -> dict:
    v = values.dropna().to_numpy(float)
    q = np.percentile(v, [5, 25, 50, 75, 95]) if v.size else [np.nan] * 5
    return {"set": label, "n": int(v.size), "mean": v.mean(), "sd": v.std(ddof=1),
            "min": v.min(), "p05": q[0], "p25": q[1], "median": q[2], "p75": q[3],
            "p95": q[4], "max": v.max(), "span_orders_of_magnitude": v.max() - v.min()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis" / "results" / "data_summary")
    args = p.parse_args()
    out, figures = args.output_dir, args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plotting.set_plot_style()

    reactions = data.load_reactions().table
    acyl, amine = data.load_features()
    split = holdout.build_split(reactions, acyl, amine)
    regime = split.label(reactions)

    # ---- classification counts
    classes = class_table(reactions, regime)
    classes.to_csv(out / "class_counts.csv", index=False)

    scope = pd.read_excel(config.SCOPE_XLSX, sheet_name="excluded")
    scope_counts = scope.note.astype(str).str.strip().str.lower().value_counts()
    drug = pd.read_excel(config.DRUG_SCOPE_XLSX)
    drug_counts = drug.rate_class.astype(str).str.strip().value_counts()

    per_reactant = []
    for kind, test_ids in (("acyl", split.test_acyls), ("amine", split.test_amines)):
        counts = pd.crosstab(reactions[kind], reactions.rate_class).reindex(
            columns=list(CLASSES), fill_value=0)
        counts["reactions"] = counts.sum(axis=1)
        counts["split"] = np.where(counts.index.isin(test_ids), "test", "train")
        counts.insert(0, "kind", kind)
        per_reactant.append(counts.rename_axis("id").reset_index())
    per_reactant = pd.concat(per_reactant, ignore_index=True)
    per_reactant.to_csv(out / "reactant_counts.csv", index=False)

    # ---- regression: measured log10 k
    measured = reactions[reactions.rate_class == "measurable"].copy()
    measured["set"] = np.where(regime[measured.index] == "pool", "pool", "test")
    stats = pd.DataFrame([logk_stats(measured.logk, "All data"),
                          logk_stats(measured[measured.set == "pool"].logk, "Training pool"),
                          logk_stats(measured[measured.set == "test"].logk, "Held-out test")])
    stats.round(3).to_csv(out / "logk_summary.csv", index=False)

    # Tiers and well counts of the rate constants used. Replicate reactions
    # (4-fluorobenzoyl chloride batches 1A/2B/3C) are already combined in
    # PRISM_rates.xlsx (scripts/build_rate_tables.py); the tier of a combined
    # reaction is that of its replicate fitted on the most wells.
    rates = pd.read_excel(config.RATES_XLSX)
    used = rates[rates.k_corrected > 0]
    tiers = used.confidence_tier.value_counts() if "confidence_tier" in used else pd.Series(dtype=int)
    wells = used.n_wells_used.value_counts().sort_index() if "n_wells_used" in used else pd.Series(dtype=int)
    repeat_note = ""
    if "n_replicates_combined" in used.columns:
        combined = used[used.n_replicates_combined > 1]
        single = used[(used.n_replicates_combined == 1) & used.batch.notna()] if "batch" in used else used.iloc[0:0]
        repeat_note = (f"  replicate reactions (acyl chloride 1, batches 1A/2B/3C): "
                       f"{len(combined) + len(single)} with a rate constant, "
                       f"{len(combined)} of them geometric means of 2-3 replicates")

    # ---- figures
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    shown = classes[classes.set != "Held-out test"].reset_index(drop=True)
    y = np.arange(len(shown))[::-1]
    left = np.zeros(len(shown))
    for c in CLASSES:
        ax.barh(y, shown[c], left=left, color=CLASS_COLOUR[c], edgecolor="white",
                linewidth=1.0, height=0.62, label=CLASS_LABEL[c])
        for yi, l, v in zip(y, left, shown[c]):
            if v >= 0.06 * shown.reactions.max():
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center", fontsize=7,
                        color="white" if c != "slow" else "black")
        left += shown[c].to_numpy()
    for yi, (_, r) in zip(y, shown.iterrows()):
        ax.text(r.reactions + 0.01 * shown.reactions.max(), yi,
                f"{int(r.reactions)} ({int(r.measurable)} / {int(r.instant)} / {int(r.slow)})",
                va="center", fontsize=7)
    ax.set_yticks(y, labels=shown.set)
    ax.set_xlabel("Reactions")
    ax.set_xlim(0, shown.reactions.max() * 1.28)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plotting._save(fig, figures / "class_counts.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bins = np.arange(np.floor(measured.logk.min() * 2) / 2,
                     np.ceil(measured.logk.max() * 2) / 2 + 0.25, 0.25)
    for key, label, colour in (("pool", "Training pool", plotting.NAVY),
                               ("test", "Held-out test", "#E69F00")):
        v = measured[measured.set == key].logk
        ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.25, color=colour)
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=1.4, color=colour,
                label=f"{label} (n = {len(v)}, median {v.median():.2f})")
        ax.axvline(v.median(), color=colour, linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"Measured log$_{10}$ $k$")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plotting._save(fig, figures / "logk_distribution.png")

    # Every acyl chloride x amine combination, coloured by class; held-out
    # reactants are set apart at the right / bottom.
    from matplotlib.colors import ListedColormap
    acyls = sorted(reactions.acyl.unique(), key=lambda i: (i in split.test_acyls, i))
    amines = sorted(reactions.amine.unique(), key=lambda i: (i in split.test_amines, i))
    code = {"measurable": 1, "instant": 2, "slow": 3}
    grid = np.zeros((len(amines), len(acyls)))
    for row in reactions.itertuples():
        grid[amines.index(row.amine), acyls.index(row.acyl)] = code[row.rate_class]
    cmap = ListedColormap(["white", CLASS_COLOUR["measurable"], CLASS_COLOUR["instant"],
                           CLASS_COLOUR["slow"]])
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_xticks(range(len(acyls)), labels=[str(i) for i in acyls], fontsize=6.5, rotation=90)
    ax.set_yticks(range(len(amines)), labels=[str(i) for i in amines], fontsize=6.5)
    for ticks, ids, test in ((ax.get_xticklabels(), acyls, split.test_acyls),
                             (ax.get_yticklabels(), amines, split.test_amines)):
        for t, i in zip(ticks, ids):
            if i in test:
                t.set_color("#B26B00"); t.set_fontweight("bold")
    ax.axvline(len(acyls) - len(split.test_acyls) - 0.5, color="black", linewidth=1.2)
    ax.axhline(len(amines) - len(split.test_amines) - 0.5, color="black", linewidth=1.2)
    ax.set_xticks(np.arange(len(acyls) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(amines) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Acyl chloride")
    ax.set_ylabel("Amine")
    ax.set_title("Reaction class for every pair (held-out reactants in orange)", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CLASS_COLOUR[c], label=CLASS_LABEL[c]) for c in CLASSES],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    plotting._save(fig, figures / "reaction_class_map.png")

    # ---- text
    a = classes.set_index("set")
    lines = ["Data summary",
             f"Rate constants: {config.RATES_XLSX.name}; reactions without one: "
             f"{config.EXCLUDED_XLSX.name}", "",
             "Classification",
             "--------------"]
    for key, label in SETS:
        r = a.loc[label]
        lines.append(f"  {label:22s} {int(r.reactions):5d} reactions "
                     f"({int(r.acyl_chlorides)} acyl chlorides, {int(r.amines)} amines): "
                     f"measurable {int(r.measurable)}, too fast {int(r.instant)}, "
                     f"un-reactive {int(r.slow)}  ->  reactive {int(r.reactive)} "
                     f"({r.fraction_reactive:.0%}), un-reactive {int(r['un-reactive'])}")
    lines += [f"  held-out reactants: acyl chlorides {list(map(int, split.test_acyls))}, "
              f"amines {list(map(int, split.test_amines))}",
              f"  excluded substrate scope ({len(scope)} substrates): "
              + ", ".join(f"{k} {v}" for k, v in scope_counts.items()),
              f"  drug scope ({len(drug)} reactions): "
              + ", ".join(f"{k} {v}" for k, v in drug_counts.items()), "",
              "Regression (measured log10 k)",
              "-----------------------------"]
    for _, s in stats.iterrows():
        lines.append(f"  {s.set:14s} n {int(s.n):4d}  mean {s['mean']:.2f}  SD {s.sd:.2f}  "
                     f"median {s['median']:.2f}  range {s['min']:.2f} to {s['max']:.2f} "
                     f"({s.span_orders_of_magnitude:.1f} orders of magnitude)")
    if repeat_note:
        lines.append(repeat_note)
    if len(tiers):
        lines.append(f"  PRISM confidence tiers ({len(used)} rate constants used): "
                     + ", ".join(f"{k} {v}" for k, v in tiers.items()))
    if len(wells):
        lines.append("  wells used per rate constant: "
                     + ", ".join(f"{k} wells {v}" for k, v in wells.items()))
    text = "\n".join(lines)
    (out / "data_summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
