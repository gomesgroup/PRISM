#!/usr/bin/env python3
"""Label the PRISM rate tables and combine replicate reactions.

    python scripts/build_rate_tables.py

Reads the rate tables as exported by the image-processing pipeline, which
identify each reaction only by reactant name:

    data/datasheets/unnamed_PRISM_rates.xlsx                 reactions with a rate constant
    data/datasheets/unnamed_PRISM_rates_unmeasurable.xlsx    reactions without one

Each acyl chloride and amine is given its label from PRISM_substrate_scope.xlsx
(`acyl_label` / `amine_label` by name), and each reaction its name rxn_AA_BB.
4-Fluorobenzoyl chloride (acyl chloride 1) was run as three replicate batches,
1A, 2B and 3C; their reactions are combined:

- rate constant: geometric mean of the usable (k > 0) replicates;
- a reaction with at least one usable replicate is measurable and appears only
  in PRISM_rates.xlsx; otherwise it takes the replicates' common class.

Writes, in data/datasheets/:

    PRISM_rates_all.xlsx             every row, labelled (sheets `rates`, `unmeasurable`),
                                     and `replicates`: each replicate reaction's 1A / 2B / 3C
                                     values next to the combined value
    PRISM_rates.xlsx                 one row per reaction; replicates combined (used by the models)
    PRISM_rates_unmeasurable.xlsx    one row per reaction without a rate constant
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

BATCH = r"-(\d[A-C])$"


def read_single(path: Path) -> tuple[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    if len(sheets) != 1:
        raise SystemExit(f"{path.name}: expected one sheet, found {list(sheets)}")
    return next(iter(sheets.items()))


def label_map(sheet: str, name_col: str, label_col: str) -> dict:
    scope = pd.read_excel(config.SCOPE_XLSX, sheet_name=sheet)
    scope = scope.dropna(subset=[name_col, label_col])
    names = scope[name_col].astype(str).str.strip()
    labels = pd.to_numeric(scope[label_col], errors="coerce")
    clash = names[names.duplicated(keep=False)]
    for name in set(clash):
        if labels[names == name].nunique() > 1:
            raise SystemExit(f"scope workbook: '{name}' has more than one {label_col}")
    return dict(zip(names, labels.astype("Int64")))


def label(df: pd.DataFrame, acyl: dict, amine: dict, what: str) -> pd.DataFrame:
    out = df.copy()
    out["acyl_chloride"] = out.acyl_chloride.astype(str).str.strip()
    out["amine"] = out.amine.astype(str).str.strip()
    out.insert(0, "acyl_label", out.acyl_chloride.map(acyl))
    out.insert(1, "amine_label", out.amine.map(amine))
    missing = out[out.acyl_label.isna() | out.amine_label.isna()]
    if len(missing):
        raise SystemExit(f"{what}: names not in the scope workbook: "
                         f"{sorted(set(missing.acyl_chloride) | set(missing.amine))}")
    out["acyl_label"] = out.acyl_label.astype(int)
    out["amine_label"] = out.amine_label.astype(int)
    out.insert(0, "reaction", [f"rxn_{a:02d}_{m:02d}" for a, m in
                               zip(out.acyl_label, out.amine_label)])
    out.insert(out.columns.get_loc("acyl_chloride") + 1, "batch",
               out.acyl_chloride.str.extract(BATCH, expand=False))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rates", type=Path, default=config.DATA_DIR / "unnamed_PRISM_rates.xlsx")
    p.add_argument("--unmeasurable", type=Path,
                   default=config.DATA_DIR / "unnamed_PRISM_rates_unmeasurable.xlsx")
    p.add_argument("--output-dir", type=Path, default=config.DATA_DIR)
    args = p.parse_args()

    acyl = label_map("acyls", "acyl_chloride", "acyl_label")
    amine = label_map("amine", "amine", "amine_label")
    rates_sheet, rates = read_single(args.rates)
    excl_sheet, excl = read_single(args.unmeasurable)
    rates = label(rates, acyl, amine, args.rates.name)
    excl = label(excl, acyl, amine, args.unmeasurable.name)
    if "rate_class" not in excl.columns:
        excl["rate_class"] = excl["bin"].map({
            "too_slow_indistinguishable_from_hydrolysis": "slow",
            "too_fast_even_when_diluted": "instant",
            "borderline_kept_out_by_fit_quality": "ambiguous"})
        if excl.rate_class.isna().any():
            raise SystemExit(f"unmapped bins: {sorted(set(excl.bin[excl.rate_class.isna()]))}")
    print(f"labelled: {len(rates)} rate rows, {len(excl)} unmeasurable rows")

    key = ["acyl_label", "amine_label"]
    for name, frame in (("rates", rates), ("unmeasurable", excl)):
        dup = frame[frame.duplicated(key + ["batch"], keep=False)]
        if len(dup):
            raise SystemExit(f"{name}: a reaction appears twice within one batch: "
                             f"{sorted(set(dup.reaction))}")

    # ---- replicate reactions: one row per reaction, all batches side by side
    rep_keys = (set(map(tuple, rates.loc[rates.batch.notna(), key].values))
                | set(map(tuple, excl.loc[excl.batch.notna(), key].values)))
    batches = sorted(set(rates.batch.dropna()) | set(excl.batch.dropna()))
    rep_rows, combined_rates, settled = [], [], set()
    for a, m in sorted(rep_keys):
        r = rates[(rates.acyl_label == a) & (rates.amine_label == m)]
        e = excl[(excl.acyl_label == a) & (excl.amine_label == m)]
        row = {"reaction": f"rxn_{a:02d}_{m:02d}", "acyl_label": a, "amine_label": m,
               "amine": (r.amine.iloc[0] if len(r) else e.amine.iloc[0])}
        for b in batches:
            rb, eb = r[r.batch == b], e[e.batch == b]
            row[f"k_{b}"] = float(rb.k_corrected.iloc[0]) if len(rb) else np.nan
            row[f"class_{b}"] = ("measurable" if len(rb) and rb.k_corrected.iloc[0] > 0
                                 else ("fit k <= 0" if len(rb) else
                                       (eb.rate_class.iloc[0] if len(eb) else "")))
        usable = r[r.k_corrected > 0]
        row["n_usable"] = len(usable)
        if len(usable):
            k = float(np.exp(np.log(usable.k_corrected).mean()))
            row["k_geometric_mean"], row["class"] = k, "measurable"
            best = usable.sort_values("n_wells_used", ascending=False, kind="stable").iloc[0]
            combined = best.copy()
            combined["acyl_chloride"] = best.acyl_chloride.rsplit("-", 1)[0]
            combined["batch"] = ",".join(sorted(usable.batch))
            combined["k_corrected"] = k
            combined["r2"] = np.nan
            combined["n_replicates_combined"] = len(usable)
            combined_rates.append(combined)
            settled.add((a, m))
        else:
            classes = sorted(set(e.rate_class))
            row["k_geometric_mean"] = np.nan
            row["class"] = classes[0] if len(classes) == 1 else "disagree: " + ",".join(classes)
            if len(classes) > 1:
                print(f"  WARNING {row['reaction']}: replicate classes disagree {classes}")
        rep_rows.append(row)
    replicates = pd.DataFrame(rep_rows)

    # ---- tables for the models
    singles = rates[rates.batch.isna()].copy()
    singles["n_replicates_combined"] = 1
    out_rates = pd.concat([singles, pd.DataFrame(combined_rates)], ignore_index=True)
    out_rates = out_rates.sort_values(key, kind="stable").reset_index(drop=True)

    excl_out = excl[excl.batch.isna()].copy()
    for (a, m), g in excl[excl.batch.notna()].groupby(key):
        if (a, m) in settled:
            continue                      # has a rate constant: lives in PRISM_rates
        first = g.iloc[0].copy()
        first["acyl_chloride"] = first.acyl_chloride.rsplit("-", 1)[0]
        first["batch"] = ",".join(sorted(g.batch))
        classes = sorted(set(g.rate_class))
        first["rate_class"] = classes[0] if len(classes) == 1 else "disagree"
        excl_out = pd.concat([excl_out, first.to_frame().T], ignore_index=True)
    excl_out = excl_out.sort_values(key, kind="stable").reset_index(drop=True)

    both = set(map(tuple, out_rates[key].values)) & set(map(tuple, excl_out[key].values))
    print(f"replicate reactions: {len(replicates)} ({len(settled)} measurable after combining)")
    print(f"PRISM_rates.xlsx: {len(out_rates)} reactions; "
          f"PRISM_rates_unmeasurable.xlsx: {len(excl_out)} reactions; in both: {len(both)}")

    out = args.output_dir
    with pd.ExcelWriter(out / "PRISM_rates_all.xlsx", engine="openpyxl") as w:
        rates.to_excel(w, sheet_name="rates", index=False)
        excl.to_excel(w, sheet_name="unmeasurable", index=False)
        replicates.to_excel(w, sheet_name="replicates", index=False)
    with pd.ExcelWriter(out / "PRISM_rates.xlsx", engine="openpyxl") as w:
        out_rates.to_excel(w, sheet_name=rates_sheet, index=False)
    with pd.ExcelWriter(out / "PRISM_rates_unmeasurable.xlsx", engine="openpyxl") as w:
        excl_out.to_excel(w, sheet_name=excl_sheet, index=False)
    print(f"written to {out}: PRISM_rates_all.xlsx, PRISM_rates.xlsx, "
          f"PRISM_rates_unmeasurable.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
