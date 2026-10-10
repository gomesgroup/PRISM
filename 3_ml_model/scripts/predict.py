#!/usr/bin/env python3
"""Predict reactivity and rate constant for unseen acyl chloride / amine pairs.

Descriptors are computed on the fly from SMILES (RDKit + morfeus + xTB), so any
pair can be scored, not only the reactants in the training set.

    # one pair
    python scripts/predict.py --acyl "O=C(Cl)c1ccccc1" --amine "Nc1ccccc1"

    # many pairs: a CSV with columns acyl_smiles, amine_smiles (others are kept)
    python scripts/predict.py --input pairs.csv --output predictions.csv

Output columns:
    p_reactive      probability that the pair reacts (amide forms)
    reactive        1 if p_reactive >= the model's threshold (0.5)
    log10_k         predicted log10 rate constant
    log10_k_gated   log10_k, left blank where the pair is predicted un-reactive
    error           why a pair could not be scored, if it could not
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config  # noqa: E402
from prism_ml.predictor import Predictor  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="CSV with acyl_smiles, amine_smiles")
    src.add_argument("--acyl", help="SMILES of one acyl chloride (use with --amine)")
    p.add_argument("--amine", help="SMILES of one amine (use with --acyl)")
    p.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--output", type=Path, default=None,
                   help="write predictions here (default: print them)")
    args = p.parse_args()

    if args.input is not None:
        pairs = pd.read_csv(args.input)
        missing = {"acyl_smiles", "amine_smiles"} - set(pairs.columns)
        if missing:
            p.error(f"{args.input} lacks columns {sorted(missing)}")
    else:
        if not args.amine:
            p.error("--acyl needs --amine")
        pairs = pd.DataFrame({"acyl_smiles": [args.acyl], "amine_smiles": [args.amine]})

    model = Predictor.load(args.model_dir)
    meta = model.metadata
    print(f"model: {meta['feature_set_label']}, classifier {meta['classifier']}, "
          f"regressor {meta['regressor']} (trained on {meta['training'].get('reactions')} reactions)")
    result = model.predict_smiles(pairs)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"written: {args.output}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(result.to_string(index=False))
    failed = result["error"].astype(bool).sum()
    if failed:
        print(f"{failed} of {len(result)} pairs could not be scored; see the error column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
