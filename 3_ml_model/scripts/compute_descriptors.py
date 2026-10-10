#!/usr/bin/env python3
"""Compute Morfeus / xTB descriptors from SMILES.

Two uses:

    # recompute the training reactants -> data/datasheets/PRISM_substrate_features_smiles.xlsx
    python scripts/compute_descriptors.py

    # any molecules: a CSV with columns id, smiles, kind (acyl or amine)
    python scripts/compute_descriptors.py --input molecules.csv --output descriptors.csv

The first rebuilds the table the saved model is trained on, keeping every
non-descriptor column (id, name, class, ...) of the stored feature workbook and
replacing each descriptor with its value computed from the reactant's SMILES.
The workbook's excluded-scope sheets (excluded_acyl, excluded_amine) are kept
unchanged.
Needs RDKit, morfeus and xtb-python (see README).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config, descriptors  # noqa: E402

#: Columns of the stored workbook that are not computed from structure here.
KEEP = {"id", "name", "smiles", "split_type", "class"}


def recompute_training(out: Path) -> None:
    sheets = {}
    for sheet in ("acyl", "amine"):
        stored = pd.read_excel(config.FEATURES_XLSX, sheet_name=sheet)
        rows = []
        for _, row in stored.iterrows():
            values = descriptors.compute(str(row["smiles"]), sheet)
            rows.append(values)
            print(f"  {sheet:5s} {int(row['id']):3d}  {row['smiles']}", flush=True)
        new = pd.DataFrame(rows)
        table = stored[[c for c in stored.columns if c in KEEP]].copy()
        for col in stored.columns:
            if col in KEEP:
                continue
            # Computed here -> recomputed value; not computed (pKa, Hessian
            # force constants) -> left blank rather than carried over, so the
            # table is honestly one pipeline.
            table[col] = pd.to_numeric(new[col], errors="coerce") if col in new else float("nan")
        table["source"] = "rdkit-mmff + morfeus + xtb (prism_ml.descriptors)"
        sheets[sheet] = table
    # The excluded-scope sheets are not recomputed here; keep them as they are.
    if out.exists():
        for sheet in config.SCOPE_FEATURE_SHEETS.values():
            try:
                sheets[sheet] = pd.read_excel(out, sheet_name=sheet)
            except ValueError:
                pass
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
    print(f"written: {out}  ({', '.join(f'{k} {v.shape[0]}' for k, v in sheets.items())})")


def compute_csv(path: Path, out: Path) -> None:
    molecules = pd.read_csv(path)
    missing = {"id", "smiles", "kind"} - set(molecules.columns)
    if missing:
        raise SystemExit(f"{path} lacks columns {sorted(missing)}")
    rows = []
    for _, m in molecules.iterrows():
        try:
            rows.append({"id": m["id"], "kind": m["kind"], "smiles": m["smiles"],
                         **descriptors.compute(str(m["smiles"]), str(m["kind"])), "error": ""})
        except Exception as exc:  # noqa: BLE001 - reported per molecule
            rows.append({"id": m["id"], "kind": m["kind"], "smiles": m["smiles"],
                         "error": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written: {out}  ({len(rows)} molecules)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="CSV with id, smiles, kind")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if args.input:
        compute_csv(args.input, args.output or args.input.with_name(
            args.input.stem + "_descriptors.csv"))
    else:
        recompute_training(args.output or config.FEATURES_SMILES_XLSX)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
