"""Scoring a fitted model on data it never saw.

Three reserved sets, each used once and only after the model was chosen on
cross-validation:

**held-out test** -- pairs whose acyl chloride, amine, or both were withheld
from training, so every prediction is a cold start.

**excluded substrate scope** -- substrates outside the training set whose
outcome is recorded as a note rather than a rate constant (the `excluded`
sheet of the substrate-scope workbook). The notes map onto the classes:

    too fast          -> reactive
    too slow          -> un-reactive
    non-reactive      -> un-reactive
    multiple products -> un-reactive
    insoluble         -> no reactivity outcome; predicted but not scored

The three un-reactive notes are one outcome: the acyl chloride was consumed by
something other than amidation, hydrolysis included.

**drug scope** -- drug-like secondary amines paired with training acyl
chlorides, with measured outcomes.

Every scored call is also labelled with the regime it was predicted in -- unseen
acyl chloride, unseen amine, or both -- so the three sets can be pooled regime by
regime. A scope substrate is paired with a fixed partner from the pool, so it
counts as both-unseen only when that partner was itself a test reactant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config, features, metrics, splits

#: exclusion note -> (training class, reactive label)
NOTE_MAP = {"too fast": ("instant", 1), "too slow": ("slow", 0),
            "non-reactive": ("slow", 0), "multiple products": ("slow", 0)}
NOTE_ORDER = ["too fast", "too slow", "non-reactive", "multiple products", "insoluble"]

#: drug-scope measured class -> reactive label
DRUG_REACTIVE = {"measurable": 1, "instant": 1, "slow": 0}

#: The pool reactant each scope substrate is paired with.
PARTNER_ACYL, PARTNER_AMINE = 1, 3

REGIME_KIND = {"new_acyl": "acyl", "new_amine": "amine", "both_new": "pair"}


def reactant_classes(table: pd.DataFrame) -> Dict[int, str]:
    """Chemical class per reactant id, from a feature sheet."""
    return {int(i): str(c) for i, c in zip(table.index, table["class"])
            if isinstance(c, str) and c}


def build_split(reactions: pd.DataFrame, acyl: pd.DataFrame,
                amine: pd.DataFrame) -> splits.Split:
    """The study's train/test split: config.FIXED_TEST_SET when set, otherwise
    the class-stratified rule with the configured pinned reactants."""
    fixed = config.FIXED_TEST_SET
    if fixed:
        for kind in ("acyl", "amine"):
            absent = sorted(set(fixed[kind]) - set(reactions[kind]))
            if absent:
                raise ValueError(f"fixed test {kind}s {absent} have no reactions")
        return splits.Split(test_acyls=tuple(sorted(fixed["acyl"])),
                            test_amines=tuple(sorted(fixed["amine"])))
    return splits.build_split(reactions, reactant_classes(acyl), reactant_classes(amine),
                              force_train_acyls=config.FORCE_TRAIN_ACYLS,
                              force_train_amines=config.FORCE_TRAIN_AMINES)


# ---------------------------------------------------------------- test set --
def evaluate_test(model, design: features.Design, split: splits.Split) -> dict:
    """Classification and rate-constant metrics on the held-out test set."""
    rows = design.rows
    pred = model.predict(design.X)
    truth = rows.reactive.to_numpy(dtype=int)
    regime = split.label(rows)
    report = metrics.classification_report(truth, pred.reactive)

    measured = np.isfinite(rows.logk.to_numpy(dtype=float))
    y = rows.logk.to_numpy(dtype=float)[measured]
    p = pred.log_k[measured]
    out = {"classification": report,
           "rate": {"n": int(measured.sum()), "r2": metrics.r2(y, p),
                    "mae": metrics.mae(y, p)},
           "by_regime": {}}
    for g in ("new_acyl", "new_amine", "both_new"):
        mask = (regime == g).to_numpy()
        if not mask.any():
            continue
        entry = {"n": int(mask.sum()),
                 "accuracy": metrics.accuracy(truth[mask], pred.reactive[mask])}
        gm = mask & measured
        if gm.sum() > 2:
            y_g = rows.logk.to_numpy(dtype=float)[gm]
            entry.update(n_rate=int(gm.sum()), r2=metrics.r2(y_g, pred.log_k[gm]),
                         mae=metrics.mae(y_g, pred.log_k[gm]))
        out["by_regime"][g] = entry
    # A rate constant is only shown where the class head expects a reaction, so
    # coverage is the fraction of measurable reactions that get a number.
    emitted = measured & pred.emitted
    out["coverage"] = float(emitted.sum() / measured.sum()) if measured.sum() else float("nan")
    if emitted.any():
        out["gated_mae"] = metrics.mae(rows.logk.to_numpy(dtype=float)[emitted],
                                       pred.log_k[emitted])
        # What the two-stage model reports: rate constants only for the
        # measured reactions it classifies reactive ("rate" above scores the
        # regression head on every measured reaction).
        y_e = rows.logk.to_numpy(dtype=float)[emitted]
        out["rate_reactive_only"] = {"n": int(emitted.sum()),
                                     "r2": metrics.r2(y_e, pred.log_k[emitted]),
                                     "mae": metrics.mae(y_e, pred.log_k[emitted])}
    resid = p - y
    out["residual_90pc_band"] = float(np.percentile(resid, 95) - np.percentile(resid, 5))
    return out


def test_frame(model, design: features.Design, split: splits.Split) -> pd.DataFrame:
    """Per-pair held-out predictions with their regime."""
    rows = design.rows
    pred = model.predict(design.X)
    return pd.DataFrame({
        "reaction": rows.reaction.to_numpy(), "acyl": rows.acyl.to_numpy(),
        "amine": rows.amine.to_numpy(), "regime": split.label(rows).to_numpy(),
        "reactive": rows.reactive.to_numpy(dtype=int), "predicted": pred.reactive,
        "probability": pred.probability, "logk": rows.logk.to_numpy(dtype=float),
        "predicted_logk": pred.log_k})


# ------------------------------------------------------------ excluded scope --
def score_scope(model, acyl: pd.DataFrame, amine: pd.DataFrame, feature_set: str,
                split: splits.Split, scope_xlsx: Path = None,
                features_xlsx: Path = None) -> pd.DataFrame:
    """Predict every excluded-scope substrate; truth from its note.

    Each substrate gets a temporary reactant id and is paired with the fixed
    pool partner. Its descriptors come from the `excluded_acyl` /
    `excluded_amine` sheets of the features workbook.
    """
    scope_xlsx = Path(scope_xlsx or config.SCOPE_XLSX)
    features_xlsx = Path(features_xlsx or config.FEATURES_SMILES_XLSX)
    scope = pd.read_excel(scope_xlsx, sheet_name="excluded")
    if "note" not in scope.columns:
        raise ValueError(f"{scope_xlsx.name}: the `excluded` sheet has no `note` "
                         f"column, so the scope has no ground truth")
    desc = {}
    for kind in ("acyl", "amine"):
        df = pd.read_excel(features_xlsx, sheet_name=config.SCOPE_FEATURE_SHEETS[kind])
        desc[kind] = df.assign(key=df["id"].astype(str)).set_index("key")

    acyl_x, amine_x = acyl.copy(), amine.copy()
    rows, meta = [], {}
    for k, row in scope.iterrows():
        kind = "acyl" if str(row["substrate"]).strip().startswith("acyl") else "amine"
        src = desc[kind].reindex([str(row["name"])]).dropna(how="all")
        if src.empty:
            continue
        sid = 90000 + k
        block = src.iloc[0]
        target = acyl_x if kind == "acyl" else amine_x
        target.loc[sid] = pd.NA
        for col in target.columns:
            if col in block.index and pd.notna(block[col]):
                target.loc[sid, col] = block[col]
        struct = [c for c in target.columns
                  if c not in ("class", "smiles", "name", "split_type", "source")
                  and not c.startswith(("FCS_", "FREQS_", "pka", "idx_", "el_"))]
        target.loc[sid, struct] = target.loc[sid, struct].fillna(0.0)
        note = str(row["note"]).strip().lower()
        rows.append({"acyl": sid if kind == "acyl" else PARTNER_ACYL,
                     "amine": sid if kind == "amine" else PARTNER_AMINE,
                     "reaction": str(row["name"]), "rate_class": "", "logk": np.nan,
                     "reactive": NOTE_MAP.get(note, (None, -1))[1]})
        meta[str(row["name"])] = (row, kind, note)

    design = features.build_design(pd.DataFrame(rows), acyl_x, amine_x, feature_set)
    pred = model.predict(design.X)
    test_acyls, test_amines = set(split.test_acyls), set(split.test_amines)
    records = []
    for name, prob, call, log_k in zip(design.rows.reaction, pred.probability,
                                       pred.reactive, pred.log_k):
        row, kind, note = meta[name]
        cls, truth = NOTE_MAP.get(note, ("", None))
        if kind == "acyl":
            regime = "both_new" if PARTNER_AMINE in test_amines else "new_acyl"
        else:
            regime = "both_new" if PARTNER_ACYL in test_acyls else "new_amine"
        records.append({"name": name, "molecule": row["molecule"], "kind": kind,
                        "substrate_class": row["class"], "note": note,
                        "training_class": cls, "regime": regime,
                        "probability": round(float(prob), 4), "predicted": int(call),
                        "predicted_logk": round(float(log_k), 3),
                        "truth": truth})
    return pd.DataFrame(records)


# ----------------------------------------------------------------- drug scope --
def score_drug(model, acyl: pd.DataFrame, amine: pd.DataFrame, feature_set: str,
               split: splits.Split, drug_xlsx: Path = None) -> pd.DataFrame:
    """Predict the drug-scope pairs; truth from their measured class."""
    raw = pd.read_excel(Path(drug_xlsx or config.DRUG_SCOPE_XLSX))
    k = pd.to_numeric(raw.k_corrected, errors="coerce")
    frame = pd.DataFrame({
        "acyl": raw.acyl_label.astype(int), "amine": raw.amine_label.astype(int),
        "reaction": raw.reaction.astype(str),
        "rate_class": raw.rate_class.astype(str).str.strip(),
        # A non-positive fitted rate is censored, not zero, as in data.py.
        "logk": np.where(k > 0, np.log10(k.where(k > 0)), np.nan)})
    frame["reactive"] = frame.rate_class.map(DRUG_REACTIVE)
    if frame.reactive.isna().any():
        raise ValueError(f"unmapped drug-scope classes: "
                         f"{sorted(set(frame.rate_class[frame.reactive.isna()]))}")
    design = features.build_design(frame, acyl, amine, feature_set)
    pred = model.predict(design.X)
    rows = design.rows.reset_index(drop=True)
    test_acyls = set(split.test_acyls)
    truth = rows.reactive.astype(int).to_numpy()
    logk = rows.logk.to_numpy(dtype=float)
    # Rate constants in the units of k_corrected in the drug-scope workbook;
    # the measured one is blank where none was fitted (too fast to resolve).
    return pd.DataFrame({
        "reaction": rows.reaction, "acyl": rows.acyl, "amine": rows.amine,
        "rate_class": rows.rate_class,
        "regime": np.where(rows.acyl.isin(test_acyls), "both_new", "new_amine"),
        "truth": truth, "probability": np.round(pred.probability, 4),
        "predicted": pred.reactive,
        "predicted_class": np.where(pred.reactive == 1, "reactive", "un-reactive"),
        "correct": pred.reactive == truth,
        "k": [float(f"{v:.4g}") if np.isfinite(v) else np.nan for v in 10.0 ** logk],
        "predicted_k": [float(f"{v:.4g}") for v in 10.0 ** pred.log_k],
        "logk": logk, "predicted_logk": np.round(pred.log_k, 3)})


# ---------------------------------------------------------------- pooling --
def pooled_calls(test: pd.DataFrame, scope: pd.DataFrame,
                 drug: pd.DataFrame) -> pd.DataFrame:
    """Every scored reactivity call across the three sets, with its regime."""
    scored_scope = scope[scope.truth.notna()]
    return pd.concat([
        pd.DataFrame({"source": "held-out test", "regime": test.regime,
                      "truth": test.reactive, "pred": test.predicted}),
        pd.DataFrame({"source": "excluded scope", "regime": scored_scope.regime,
                      "truth": scored_scope.truth.astype(int),
                      "pred": scored_scope.predicted}),
        pd.DataFrame({"source": "drug scope", "regime": drug.regime,
                      "truth": drug.truth, "pred": drug.predicted}),
    ], ignore_index=True).astype({"truth": int, "pred": int})
