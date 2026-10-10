#!/usr/bin/env python3
"""PRISM per-well hydrolysis filtering and rate-constant refit.

Usage:
    python prism_perwell.py --plates <zip or folder of '* times_and_concs.xlsx'> --outdir <folder> \
                            [--margins 1.1 1.3 1.4 1.5 1.7] [--export-margin 1.5] \
                            [--scope <PRISM_substrate_scope.xlsx>] [--datasheets <folder>]

Input  : one workbook per (group, acyl chloride) plate. Each amine has two columns,
         '<amine>+<acyl> times'  (inflection time, min; the time to the indicator colour change)
         '<amine>+<acyl> 1/[A]'  (stored ordinate, = 2/[A]0 -> see note below)
         plus a 'blank+<acyl>' column pair (no-amine hydrolysis control) on the group1 plate of each
         acyl chloride. Rows are the 8 serial-dilution levels (first column = concentration, mM).

Method (per dilution row = per well):
  1. Control for a well = the blank's inflection time at the SAME dilution row. Plates without their own
     blank (group2/group3) use the blank from the same acyl chloride's group1 plate; the dilution grid is
     identical across plates, so rows are matched on concentration.
  2. A well is KEPT if   blank_time / amine_time >= margin   (the amine turns the indicator at least
     `margin`-fold faster than hydrolysis does at that concentration). Other wells are dropped
     individually; the pair is NOT discarded because one well fails.
  3. Rate constant k = slope of 1/[A]0 against inflection time over the kept wells, where
     1/[A]0 = stored ordinate / 2 (the stored value is the 50%-conversion value 2/[A]0).
     Needs >= 2 kept wells with distinct times.  Units: M^-1 min^-1.
  4. Confidence tier by number of kept wells: >=4 high, 3 medium, 2 low (order of magnitude only).
  5. Pairs without a rate are binned (excluded file):
       too_fast_even_when_diluted   - inflection <= 2 min at the most dilute concentration with a reading,
                                      or the amine is on the manual fast-amine list below
       too_slow_indistinguishable_from_hydrolysis - detected, but < 2 wells beat the control
       borderline_kept_out_by_fit_quality         - >=2 wells kept but times identical (no slope)
       set_aside_negative_fit                     - >=2 wells kept but the fitted slope is <= 0 (unphysical);
                                                    the value is kept in column k_negative_fit_value
  6. control_status flags pairs whose control is NOT from their own plate (group2/group3 plates borrow the
     blank from the group1 plate): 'indeterminate: control borrowed from other plate'. These are still fit
     (so nothing is silently dropped) but can be filtered out on this column.
       too_fast_no_inflection_detected            - no reading at any dilution (does not occur in this data)

Outputs (in outdir):
    inflection_times_long.csv           every time/ordinate in the input, controls included
    well_level_kept_excluded.csv        every amine well: times, control time, ratio, kept/excluded per margin
    pair_level_fit_<margin>.csv         pairs with a rate constant
    pair_level_excluded_<margin>.csv    pairs without one, binned, with the evidence for the bin
    unnamed_PRISM_rates.xlsx,           the export margin's pairs in the ML workbook layout, identified by
    unnamed_PRISM_rates_unmeasurable.xlsx   reactant name only (bins as above, no review)

Rate tables for the ML model (in --datasheets, export margin only):
    PRISM_rates.xlsx                    one row per reaction with a rate constant
    PRISM_rates_unmeasurable.xlsx       one row per reaction without one: rate_class and note
    PRISM_rates_all.xlsx                every plate row, labelled, and the replicate batches side by side
  Each reactant gets its number from the scope workbook (matched by name) and each reaction its name
  rxn_<acyl>_<amine>. 4-Fluorobenzoyl chloride was run on three plates (-1A, -2B, -3C): its rate constant
  is the geometric mean of the replicates with k > 0; with none, the replicates' common class.
  rate_class: slow (too_slow bin), instant (reactive, not measurable: too_fast, borderline and
  negative-fit bins, and reviewed slow pairs, below) or undetermined (not modelled). The note says why.

Control review (export margin). The too_slow bin does not separate wells that failed the control from
wells that could not be compared with it (no control reading, or amine time 0). Pairs binned too slow
with fewer than two comparable wells (kept or below margin) in every replicate are reviewed:
    instant       >= 2 wells beat the control at the same concentration (kept, or amine time 0 while the
                  control has a time), none failed, and every amine-time-0 well is at a higher
                  concentration than the kept wells; in every replicate. Not measurable: an amine-time-0
                  well supports the label but does not establish a 1.5x ratio.
    undetermined  otherwise.
A replicate pair that fails the control in another batch keeps the too_slow bin (slow).
"""
import argparse, glob, io, os, re, zipfile
import numpy as np
import pandas as pd

REF_CONC = np.array([50.0, 25.0, 12.5, 6.25, 3.125, 1.563, 0.781, 0.391])   # mM, dilution grid
CONTROL = "blank"
TOO_FAST_MIN = 2.0
# Secondary aliphatic amines that react too fast to detect (non-detection = too fast, not hydrolysis-limited)
FAST_AMINES = {"diisopropylamine", "2-methylpiperidine", "morpholine", "3 amino pentane"}


def tag(m):
    return ("%gx" % m).replace(".", "p")


def plate_key(path):
    base = re.sub(r"\s*times_and_concs\.xlsx$", "", os.path.basename(path))
    g, a = base.split(" + ", 1)
    return g.strip(), a.strip()


def list_plates(src):
    """(file name, readable source) for every plate workbook in a folder or a zip."""
    if src.endswith(".zip"):
        z = zipfile.ZipFile(src)
        names = sorted(n for n in z.namelist() if n.endswith("times_and_concs.xlsx"))
        return [(os.path.basename(n), io.BytesIO(z.read(n))) for n in names]
    return [(os.path.basename(p), p) for p in sorted(glob.glob(os.path.join(src, "*times_and_concs.xlsx")))]


def load_plate(path):
    if hasattr(path, "seek"):
        path.seek(0)
    df = pd.read_excel(path)
    conc = pd.to_numeric(df["Unnamed: 0"], errors="coerce").to_numpy(float)
    cols = {}
    for c in df.columns:
        if c.endswith(" times") and (c[:-6] + " 1/[A]") in df.columns:
            stem = c[:-6]
            cols[stem.split("+")[0].strip()] = (
                pd.to_numeric(df[c], errors="coerce").to_numpy(float),
                pd.to_numeric(df[stem + " 1/[A]"], errors="coerce").to_numpy(float))
    return conc, cols


def align(conc, arr):
    """Place a column on the standard 8-level grid by concentration (some plates lack a row)."""
    lk = dict(zip(conc, arr))
    return np.array([lk.get(c, np.nan) for c in REF_CONC], float)


def fit(t, y_stored):
    y = y_stored / 2.0
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    n = len(t)
    if n < 2 or np.ptp(t) == 0:
        return np.nan, np.nan, n
    s, b = np.polyfit(t, y, 1)
    r2 = np.nan
    if n >= 3:
        ss = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - (s * t + b)) ** 2).sum() / ss if ss > 0 else np.nan
    return s, r2, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1.1, 1.3, 1.4, 1.5, 1.7])
    ap.add_argument("--export-margin", type=float, default=1.5,
                    help="margin written as the ML rate tables (must be in --margins)")
    ap.add_argument("--scope", default=None, help="PRISM_substrate_scope.xlsx (default: in --datasheets)")
    ap.add_argument("--datasheets", default=None,
                    help="folder for PRISM_rates*.xlsx (default: ../3_ml_model/data/datasheets); 'none' to skip")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    files = list_plates(a.plates)
    blank_by_acyl, long_rows = {}, []
    for name, p in files:
        g, acyl = plate_key(name)
        conc, cols = load_plate(p)
        for amine, (t, y) in cols.items():
            for i in range(len(conc)):
                long_rows.append(dict(group=g, acyl_chloride=acyl, amine=amine, is_control=(amine == CONTROL),
                                      conc_mM=conc[i], inflection_time_min=t[i], stored_ordinate_2_over_A0=y[i],
                                      source_file=name))
        if CONTROL in cols and acyl not in blank_by_acyl:
            blank_by_acyl[acyl] = (align(conc, cols[CONTROL][0]), g)
    pd.DataFrame(long_rows).to_csv(os.path.join(a.outdir, "inflection_times_long.csv"), index=False)

    well_rows, pair_rows = [], []
    for name, p in files:
        group, acyl = plate_key(name)
        conc, cols = load_plate(p)
        own = CONTROL in cols
        if own:
            tb, ctl_plate = align(conc, cols[CONTROL][0]), group
        elif acyl in blank_by_acyl:
            tb, ctl_plate = blank_by_acyl[acyl]
        else:
            continue
        for amine, (tr, yr) in cols.items():
            if amine == CONTROL:
                continue
            t, y = align(conc, tr), align(conc, yr)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = tb / np.where(t == 0, np.nan, t)
            base = dict(group=group, acyl_chloride=acyl, amine=amine,
                        control_source="own_plate" if own else "same_acyl_%s_plate" % ctl_plate,
                        control_status="matched same-plate control" if own else "indeterminate: control borrowed from other plate")
            kept = {}
            for m in a.margins:
                kept[m] = np.isfinite(t) & np.isfinite(tb) & (ratio >= m)
            for i in range(len(REF_CONC)):
                r = dict(base, conc_mM=REF_CONC[i], amine_time_min=t[i], control_time_min=tb[i],
                         stored_ordinate_2_over_A0=y[i], control_over_amine_ratio=ratio[i])
                for m in a.margins:
                    if kept[m][i]:
                        s = "kept"
                    elif not np.isfinite(t[i]):
                        s = "excluded: no amine reading"
                    elif not np.isfinite(tb[i]):
                        s = "excluded: no control reading"
                    elif t[i] == 0:
                        s = "excluded: amine time 0 (done at first read; ratio undefined)"
                    else:
                        s = "excluded: below margin"
                    r["status_%s" % tag(m)] = s
                well_rows.append(r)
            n_raw = int(np.isfinite(t).sum())
            valid = np.where(np.isfinite(t))[0]
            t_dil = t[valid.max()] if len(valid) else np.nan
            c_dil = REF_CONC[valid.max()] if len(valid) else np.nan
            for m in a.margins:
                k, r2, n = fit(t[kept[m]], y[kept[m]])
                pair_rows.append(dict(base, margin=m, n_wells_kept=n, k_M1min1=k, r2=r2, n_raw_detections=n_raw,
                                      kept_conc_mM=";".join("%g" % c for c in REF_CONC[kept[m]]),
                                      time_at_most_dilute_min=t_dil, conc_at_most_dilute_mM=c_dil))
    pd.DataFrame(well_rows).to_csv(os.path.join(a.outdir, "well_level_kept_excluded.csv"), index=False)

    pairs = pd.DataFrame(pair_rows)
    tables = {}
    for m in a.margins:
        d = pairs[pairs["margin"] == m].copy()
        has = d["k_M1min1"].notna() & (d["k_M1min1"] > 0)
        d["k_negative_fit_value"] = np.where(d["k_M1min1"] <= 0, d["k_M1min1"], np.nan)
        d["confidence_tier"] = np.where(~has, "", np.where(d["n_wells_kept"] >= 4, "high_confidence",
                                np.where(d["n_wells_kept"] == 3, "medium_confidence", "low_confidence_order_of_magnitude")))
        fitd = d[has].drop(columns=["k_negative_fit_value"]).sort_values(["group", "acyl_chloride", "amine"])
        ex = d[~has].copy()

        def binify(r):
            if pd.notna(r["k_negative_fit_value"]):
                return "set_aside_negative_fit"
            if r["n_raw_detections"] == 0:
                return "too_fast_no_inflection_detected"
            if r["time_at_most_dilute_min"] <= TOO_FAST_MIN or r["amine"] in FAST_AMINES:
                return "too_fast_even_when_diluted"
            if r["n_wells_kept"] < 2:
                return "too_slow_indistinguishable_from_hydrolysis"
            return "borderline_kept_out_by_fit_quality"
        ex["bin"] = ex.apply(binify, axis=1)
        ex = ex.drop(columns=["k_M1min1", "r2", "confidence_tier"]).sort_values(["bin", "group", "acyl_chloride", "amine"])
        tables[m] = (fitd, ex)
        fitd.to_csv(os.path.join(a.outdir, "pair_level_fit_%s.csv" % tag(m)), index=False)
        ex.to_csv(os.path.join(a.outdir, "pair_level_excluded_%s.csv" % tag(m)), index=False)
        print("margin %.1fx: %d pairs fit (%s) | %d excluded (%s)" % (
            m, len(fitd), ", ".join("%s %d" % (k.split("_")[0], v) for k, v in fitd["confidence_tier"].value_counts().items()),
            len(ex), ", ".join("%s %d" % (k.split("_")[0] + "_" + k.split("_")[1], v) for k, v in ex["bin"].value_counts().items())))


    if a.export_margin not in tables:
        raise SystemExit("--export-margin %g is not in --margins" % a.export_margin)
    fitd, ex = tables[a.export_margin]
    rates, unmeas = export_tables(fitd, ex)
    write_sheet(rates, os.path.join(a.outdir, "unnamed_PRISM_rates.xlsx"), "PRISM_perwell_validated_dataset")
    write_sheet(unmeas, os.path.join(a.outdir, "unnamed_PRISM_rates_unmeasurable.xlsx"),
                "PRISM_excluded_binned_%s" % tag(a.export_margin))
    if (a.datasheets or "").lower() == "none":
        return
    here = os.path.dirname(os.path.abspath(__file__))
    datasheets = a.datasheets or os.path.join(here, "..", "..", "3_ml_model", "data", "datasheets")
    scope = a.scope or os.path.join(datasheets, "PRISM_substrate_scope.xlsx")
    wells = pd.DataFrame(well_rows)
    build_rate_tables(rates, unmeas, wells, "status_%s" % tag(a.export_margin), scope, datasheets,
                      "PRISM_excluded_binned_%s" % tag(a.export_margin))


# ---------------------------------------------------------------- export for the ML model

KEY = ["group", "acyl_chloride", "amine"]
BIN_CLASS = {"too_slow_indistinguishable_from_hydrolysis": "slow",
             "too_fast_even_when_diluted": "instant",
             "borderline_kept_out_by_fit_quality": "instant",
             "set_aside_negative_fit": "instant"}
BIN_NOTE = {
    "too_fast_even_when_diluted":
        "Instant (reactive, not measurable): too fast to resolve, so no rate constant could be fitted.",
    "borderline_kept_out_by_fit_quality":
        "Instant (reactive, not measurable): the wells beat the hydrolysis control, but with identical "
        "times there is no slope, so no rate constant could be fitted.",
    "set_aside_negative_fit":
        "Instant (reactive, not measurable): the wells beat the hydrolysis control, but the fitted slope "
        "is negative (unphysical; times at the reading limit), so no rate constant is used."}
NOTE_REVIEW_INSTANT = (
    "Instant (reactive, not measurable): at least two wells reacted faster than the hydrolysis control "
    "at the same concentration and none failed the control. This counts wells where the amine reaction "
    "was complete by the first reading (amine time 0) while the control was not, which the filter "
    "excludes because the ratio is undefined; such wells support the label but do not establish a 1.5x "
    "ratio, and too few timed wells remain to fit a rate constant.")
NOTE_REVIEW_UNDETERMINED = (
    "Undetermined: fewer than two wells could be compared with the hydrolysis control (no control "
    "reading at that concentration, or no control time for a first-read well), so neither a reactive nor "
    "a slow label is supported; left out of modelling.")


def bin_note(r):
    if r["bin"] == "borderline_kept_out_by_fit_quality":
        return "Beat control at %d dilutions but surviving times were identical (no slope)." % r["n_wells_kept"]
    if r["bin"] == "too_fast_even_when_diluted":
        if r["time_at_most_dilute_min"] <= TOO_FAST_MIN:
            return ("Even at the most dilute concentration tested (%g mM) the color change was complete by "
                    "the first reading (%d min)." % (r["conc_at_most_dilute_mM"], r["time_at_most_dilute_min"]))
        return ("Reclassified per domain knowledge: highly reactive secondary aliphatic amine; non-detection "
                "reflects a reaction too fast to capture, not loss to hydrolysis.")
    if r["bin"] == "too_slow_indistinguishable_from_hydrolysis":
        return ("Detected at %d dilution(s) but at fewer than 2 was the amine faster than hydrolysis "
                "(>= %gx)." % (r["n_raw_detections"], r["margin"]))
    if r["bin"] == "set_aside_negative_fit":
        return ("Beat control at %d dilutions but the fitted slope was negative (k = %g); set aside as "
                "unphysical." % (r["n_wells_kept"], r["k_negative_fit_value"]))
    return ""


def export_tables(fitd, ex):
    """The export margin's pairs in the workbook layout the ML model reads (bins, no review)."""
    cs = lambda s: np.where(s == "own_plate", "own_plate", "same_acyl_other_plate")
    rates = pd.DataFrame({
        "group": fitd.group, "acyl_chloride": fitd.acyl_chloride, "amine": fitd.amine,
        "n_wells_used": fitd.n_wells_kept.astype(int), "control_source": cs(fitd.control_source),
        "k_corrected": fitd.k_M1min1, "r2": fitd.r2, "confidence_tier": fitd.confidence_tier,
        "hydrolysis_margin_used": fitd.margin}).sort_values(KEY).reset_index(drop=True)
    unmeas = pd.DataFrame({
        "group": ex.group, "acyl_chloride": ex.acyl_chloride, "amine": ex.amine,
        "control_source": cs(ex.control_source), "n_raw_detections": ex.n_raw_detections.astype(int),
        "n_beats_control": ex.n_wells_kept.astype(int),
        "time_at_most_dilute_tested_min": ex.time_at_most_dilute_min.round().astype(int),
        "conc_at_most_dilute_tested_mM": ex.conc_at_most_dilute_mM, "hydrolysis_margin_used": ex.margin,
        "bin": ex["bin"], "note": [bin_note(r) for _, r in ex.iterrows()]}
    ).sort_values(["bin"] + KEY).reset_index(drop=True)
    return rates, unmeas


def write_sheet(frame, path, sheet):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        frame.to_excel(w, sheet_name=sheet, index=False)


def label(df, scope):
    """Reactant numbers from the scope workbook (by name), reaction name and replicate batch."""
    maps = {}
    for sheet, name_col, label_col in (("acyls", "acyl_chloride", "acyl_label"), ("amine", "amine", "amine_label")):
        sc = pd.read_excel(scope, sheet_name=sheet).dropna(subset=[name_col, label_col])
        names = sc[name_col].astype(str).str.strip()
        labels = pd.to_numeric(sc[label_col], errors="coerce")
        for n in set(names[names.duplicated(keep=False)]):
            if labels[names == n].nunique() > 1:
                raise SystemExit("scope workbook: '%s' has more than one %s" % (n, label_col))
        maps[name_col] = dict(zip(names, labels.astype("Int64")))
    out = df.copy()
    out["acyl_chloride"] = out.acyl_chloride.astype(str).str.strip()
    out["amine"] = out.amine.astype(str).str.strip()
    out.insert(0, "acyl_label", out.acyl_chloride.map(maps["acyl_chloride"]))
    out.insert(1, "amine_label", out.amine.map(maps["amine"]))
    missing = out[out.acyl_label.isna() | out.amine_label.isna()]
    if len(missing):
        raise SystemExit("names not in the scope workbook: %s"
                         % sorted(set(missing.acyl_chloride) | set(missing.amine)))
    out["acyl_label"] = out.acyl_label.astype(int)
    out["amine_label"] = out.amine_label.astype(int)
    out.insert(0, "reaction", ["rxn_%02d_%02d" % (x, y) for x, y in zip(out.acyl_label, out.amine_label)])
    out.insert(out.columns.get_loc("acyl_chloride") + 1, "batch",
               out.acyl_chloride.str.extract(r"-(\d[A-C])$", expand=False))
    return out


def review_wells(w, status):
    """Evidence counts and verdict for one plate pair from its wells."""
    s = w[status]
    first = s.str.startswith("excluded: amine time 0")
    first_vs = first & w.control_time_min.notna() & (w.control_time_min > 0)
    kept_c, first_c = w.conc_mM[s == "kept"], w.conc_mM[first_vs]
    n = {"passed": int((s == "kept").sum()), "first_read": int(first_vs.sum()),
         "failed": int((s == "excluded: below margin").sum()),
         "no_control": int((s == "excluded: no control reading").sum() + (first & ~first_vs).sum())}
    n["comparable"] = n["passed"] + n["failed"]
    ordered = len(kept_c) == 0 or len(first_c) == 0 or first_c.min() > kept_c.max()
    n["qualifies"] = n["passed"] + n["first_read"] >= 2 and n["failed"] == 0 and ordered
    return n


def build_rate_tables(rates, unmeas, wells, status, scope, outdir, unmeas_sheet):
    rates = label(rates, scope)
    excl = label(unmeas, scope)
    excl["rate_class"] = excl["bin"].map(BIN_CLASS)
    if excl.rate_class.isna().any():
        raise SystemExit("unmapped bins: %s" % sorted(set(excl["bin"][excl.rate_class.isna()])))
    for b, extra in BIN_NOTE.items():
        excl.loc[excl["bin"] == b, "note"] = excl.loc[excl["bin"] == b, "note"] + " " + extra
    key = ["acyl_label", "amine_label"]

    # control review of the too_slow bin
    slow = excl[excl["bin"] == "too_slow_indistinguishable_from_hydrolysis"]
    ev = {}
    for i, r in slow.iterrows():
        w = wells[(wells.group == r["group"]) & (wells.acyl_chloride == r["acyl_chloride"]) & (wells.amine == r["amine"])]
        ev[i] = review_wells(w, status)
    measured = set(map(tuple, rates.loc[rates.k_corrected > 0, key].values))
    other = set(map(tuple, excl.loc[excl["bin"] != "too_slow_indistinguishable_from_hydrolysis", key].values))
    reviewed = {"instant": 0, "undetermined": 0}
    for (x, y), g in slow.groupby(key):
        if (x, y) in measured or (x, y) in other:
            continue
        insufficient = [ev[i]["comparable"] < 2 for i in g.index]
        if not all(insufficient):
            continue                       # enough comparable wells somewhere: the slow bin stands
        verdict = "instant" if all(ev[i]["qualifies"] for i in g.index) else "undetermined"
        parts = []
        for i in g.index:
            e, r = ev[i], excl.loc[i]
            parts.append("%s%d passed, %d first-read vs control, %d failed, %d without a control reading (%s)"
                         % (r["batch"] + ": " if isinstance(r["batch"], str) else "", e["passed"], e["first_read"],
                            e["failed"], e["no_control"],
                            "own plate" if r["control_source"] == "own_plate" else "borrowed from the group1 plate"))
        note = (NOTE_REVIEW_INSTANT if verdict == "instant" else NOTE_REVIEW_UNDETERMINED) + \
            " Evidence: " + "; ".join(parts) + "."
        excl.loc[g.index, "rate_class"] = verdict
        excl.loc[g.index, "note"] = note
        reviewed[verdict] += 1

    # replicate reactions: one row per reaction, batches side by side
    rep_keys = (set(map(tuple, rates.loc[rates.batch.notna(), key].values))
                | set(map(tuple, excl.loc[excl.batch.notna(), key].values)))
    batches = sorted(set(rates.batch.dropna()) | set(excl.batch.dropna()))
    rep_rows, combined, settled = [], [], set()
    for x, y in sorted(rep_keys):
        r = rates[(rates.acyl_label == x) & (rates.amine_label == y)]
        e = excl[(excl.acyl_label == x) & (excl.amine_label == y)]
        row = {"reaction": "rxn_%02d_%02d" % (x, y), "acyl_label": x, "amine_label": y,
               "amine": r.amine.iloc[0] if len(r) else e.amine.iloc[0]}
        for b in batches:
            rb, eb = r[r.batch == b], e[e.batch == b]
            row["k_" + b] = float(rb.k_corrected.iloc[0]) if len(rb) else np.nan
            row["class_" + b] = "measurable" if len(rb) else (eb.rate_class.iloc[0] if len(eb) else "")
        usable = r[r.k_corrected > 0]
        row["n_usable"] = len(usable)
        if len(usable):
            k = float(np.exp(np.log(usable.k_corrected).mean()))
            row["k_geometric_mean"], row["class"] = k, "measurable"
            best = usable.sort_values("n_wells_used", ascending=False, kind="stable").iloc[0].copy()
            best["acyl_chloride"] = best.acyl_chloride.rsplit("-", 1)[0]
            best["batch"] = ",".join(sorted(usable.batch))
            best["k_corrected"], best["r2"], best["n_replicates_combined"] = k, np.nan, len(usable)
            combined.append(best)
            settled.add((x, y))
        else:
            classes = sorted(set(e.rate_class))
            row["k_geometric_mean"] = np.nan
            row["class"] = classes[0] if len(classes) == 1 else "disagree: " + ",".join(classes)
        rep_rows.append(row)

    singles = rates[rates.batch.isna()].copy()
    singles["n_replicates_combined"] = 1
    out_rates = pd.concat([singles, pd.DataFrame(combined)], ignore_index=True)
    out_rates = out_rates.sort_values(key, kind="stable").reset_index(drop=True)
    excl_out = excl[excl.batch.isna()].copy()
    for (x, y), g in excl[excl.batch.notna()].groupby(key):
        if (x, y) in settled:
            continue
        first = g.iloc[0].copy()
        first["acyl_chloride"] = first.acyl_chloride.rsplit("-", 1)[0]
        first["batch"] = ",".join(sorted(g.batch))
        classes = sorted(set(g.rate_class))
        if len(classes) > 1:
            raise SystemExit("replicates of %s disagree (%s) and none has a rate" % (first["reaction"], classes))
        excl_out = pd.concat([excl_out, first.to_frame().T], ignore_index=True)
    excl_out = excl_out.sort_values(key, kind="stable").reset_index(drop=True)

    os.makedirs(outdir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(outdir, "PRISM_rates_all.xlsx"), engine="openpyxl") as w:
        rates.to_excel(w, sheet_name="rates", index=False)
        excl.to_excel(w, sheet_name="unmeasurable", index=False)
        pd.DataFrame(rep_rows).to_excel(w, sheet_name="replicates", index=False)
    write_sheet(out_rates, os.path.join(outdir, "PRISM_rates.xlsx"), "PRISM_perwell_validated_dataset")
    write_sheet(excl_out, os.path.join(outdir, "PRISM_rates_unmeasurable.xlsx"), unmeas_sheet)
    counts = excl_out.rate_class.value_counts().to_dict()
    print("rate tables: %d reactions with a rate constant; without one: %s; control review: %d instant, "
          "%d undetermined -> %s" % (len(out_rates), counts, reviewed["instant"], reviewed["undetermined"], outdir))


if __name__ == "__main__":
    main()
