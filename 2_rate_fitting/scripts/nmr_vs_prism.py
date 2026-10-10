#!/usr/bin/env python3
"""Pair each NMR rate with its PRISM rate (per hydrolysis margin) and summarise agreement.

Usage: python nmr_vs_prism.py --nmr PRISM_nmr_rates.xlsx --key label_key_acyl_amine.csv \
                              --prism-dir <outdir of prism_perwell.py> --outdir <folder> [--scatter-margin 1.5]

Writes paired_nmr_prism_rates.csv, agreement_by_margin.csv and margin_vs_nmr.png (RMS error and number of
NMR-matched pairs against the margin, for pairs fitted on >= 2 and >= 3 wells, and PRISM vs NMR at one margin).

NMR file: one row per reaction, ID 'rxn_<acyl label>_<amine label>', k_NMR in M^-1 min^-1 (25 C).
Labels -> compound names via label_key_acyl_amine.csv (from the '25C Rates' sheet of the lab workbook).
Acyl 1 (4-fluorobenzoyl chloride) was run on three plates (1A/2B/3C); the PRISM value is the geometric mean
of the replicates that have a rate. Acyl labels 32-36 and amines run only in the test set have no per-well
PRISM data here, so their PRISM columns are blank.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--nmr", required=True); ap.add_argument("--key", required=True)
ap.add_argument("--prism-dir", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--scatter-margin", type=float, default=1.5)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

key = pd.read_csv(a.key)
acyl_names = {r.label: r.name_in_PRISM_plate_files.split(" | ") for r in key[key.kind == "acyl_chloride"].itertuples() if isinstance(r.name_in_PRISM_plate_files, str) and r.name_in_PRISM_plate_files}
acyl_disp = {r.label: r.name_in_label_sheet for r in key[key.kind == "acyl_chloride"].itertuples()}
amine_name = {r.label: r.name_in_label_sheet for r in key[key.kind == "amine"].itertuples()}

nmr = pd.read_excel(a.nmr)
files = {float(re.search(r"fit_(\d+)p?(\d*)x", os.path.basename(f)).expand(r"\1.\2" if re.search(r"p", f) else r"\1")): f
         for f in glob.glob(os.path.join(a.prism_dir, "pair_level_fit_*.csv"))}
margins = sorted(files)
fits = {m: pd.read_csv(files[m]) for m in margins}
excl = {m: pd.read_csv(os.path.join(a.prism_dir, "pair_level_excluded_%s.csv" % os.path.basename(files[m]).split("fit_")[1][:-4])) for m in margins}

out = []
for r in nmr.itertuples():
    row = dict(reaction_id=r.reaction, acyl_label=r.acyl_label, amine_label=r.amine_label,
               acyl_chloride=acyl_disp.get(r.acyl_label, ""), amine=amine_name.get(r.amine_label, ""),
               k_NMR_M1min1=r.k_NMR, in_train_PRISM_data=r.acyl_label in acyl_names)
    plates = acyl_names.get(r.acyl_label)
    for m in margins:
        t = ("%gx" % m).replace(".", "p")
        if not plates:
            row.update({f"k_PRISM_{t}": np.nan, f"n_wells_{t}": np.nan, f"PRISM_status_{t}": "no per-well PRISM data (test set)"}); continue
        f = fits[m]; f = f[f.acyl_chloride.isin(plates) & (f.amine == row["amine"])]
        e = excl[m]; e = e[e.acyl_chloride.isin(plates) & (e.amine == row["amine"])]
        if len(f):
            row[f"k_PRISM_{t}"] = 10 ** np.log10(f.k_M1min1).mean()
            row[f"n_wells_{t}"] = ";".join(str(int(x)) for x in f.n_wells_kept)
            row[f"PRISM_status_{t}"] = "fit (%d replicate plate%s)" % (len(f), "s" if len(f) > 1 else "")
        elif len(e):
            row[f"k_PRISM_{t}"] = np.nan; row[f"n_wells_{t}"] = ""
            row[f"PRISM_status_{t}"] = "no rate: " + " / ".join(sorted(set(e["bin"])))
        else:
            row[f"k_PRISM_{t}"] = np.nan; row[f"n_wells_{t}"] = ""; row[f"PRISM_status_{t}"] = "pair not run on PRISM plates"
    out.append(row)
paired = pd.DataFrame(out)
paired.to_csv(os.path.join(a.outdir, "paired_nmr_prism_rates.csv"), index=False)

# agreement by margin (log10 space), pairs with a PRISM rate
def matched(m, min_wells=2):
    t = ("%gx" % m).replace(".", "p")
    c = paired.dropna(subset=[f"k_PRISM_{t}"])
    c = c[(c.k_NMR_M1min1 > 0) & (c[f"k_PRISM_{t}"] > 0)]
    if min_wells > 2:   # every replicate plate fitted on >= min_wells wells
        c = c[c[f"n_wells_{t}"].astype(str).map(lambda v: all(int(x) >= min_wells for x in v.split(";")))]
    return c, t

summ = []
for m in margins:
    c, t = matched(m)
    x, y = np.log10(c.k_NMR_M1min1), np.log10(c[f"k_PRISM_{t}"]); d = y - x
    s, b = np.polyfit(x, y, 1)
    summ.append(dict(margin=m, n_pairs=len(c), slope=s, intercept=b, R2=np.corrcoef(x, y)[0, 1] ** 2,
                     median_PRISM_over_NMR=10 ** d.median(), rmse_log10=float(np.sqrt((d ** 2).mean())),
                     pct_within_2x=100 * (d.abs() <= np.log10(2)).mean(), pct_within_3x=100 * (d.abs() <= np.log10(3)).mean()))
pd.DataFrame(summ).round(3).to_csv(os.path.join(a.outdir, "agreement_by_margin.csv"), index=False)
print(pd.DataFrame(summ).round(3).to_string(index=False))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for w, colour in ((2, "#2F6DB5"), (3, "#C8622A")):
    rms, n = [], []
    for m in margins:
        c, t = matched(m, w)
        d = np.log10(c[f"k_PRISM_{t}"]) - np.log10(c.k_NMR_M1min1)
        rms.append(float(np.sqrt((d ** 2).mean())) if len(c) else np.nan); n.append(len(c))
    ax[0].plot(margins, rms, "o-", color=colour, label=f"pairs with >= {w} wells")
    ax[1].plot(margins, n, "o-", color=colour, label=f"pairs with >= {w} wells")
ax[0].set(xlabel="hydrolysis margin (x control)", ylabel="RMS error vs NMR (log10)")
ax[1].set(xlabel="hydrolysis margin (x control)", ylabel="NMR-matched pairs")
for a_ in ax[:2]:
    a_.legend(frameon=False, fontsize=8)
c, t = matched(a.scatter_margin)
ax[2].scatter(c.k_NMR_M1min1, c[f"k_PRISM_{t}"], s=18, color="#2F6DB5", edgecolor="white", linewidth=0.5,
              label=f"{a.scatter_margin:g}x margin (n = {len(c)})")
lim = np.array([0.1, 1000.0])
ax[2].plot(lim, lim, color="black", linewidth=1)
ax[2].fill_between(lim, lim / 2, lim * 2, color="0.9", zorder=0)
ax[2].set(xscale="log", yscale="log", xlim=lim, ylim=(1, 1000),
          xlabel="NMR k (M$^{-1}$ min$^{-1}$)", ylabel="PRISM k (M$^{-1}$ min$^{-1}$)")
ax[2].legend(frameon=False, fontsize=8, loc="upper left")
ax[2].set_title("line = 1:1, band = within 2x", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(a.outdir, "margin_vs_nmr.png"), dpi=300)
print("written:", os.path.join(a.outdir, "margin_vs_nmr.png"))
