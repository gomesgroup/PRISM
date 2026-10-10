#!/usr/bin/env python3
"""Fit the selected model on every labelled reaction and save it to model/.

    python scripts/train_final.py

The saved model is trained on descriptors computed from SMILES by
prism_ml.descriptors (data/datasheets/PRISM_substrate_features_smiles.xlsx,
built by scripts/compute_descriptors.py) -- the same computation predict.py
runs for an unseen molecule. A model trained on the stored descriptor table does
not transfer to on-the-fly descriptors, which come from a different conformer
pipeline.

Before the final fit, the same pipeline is fitted on the training pool and
scored on the held-out test set; that score is written to model/metadata.json
so the deployed model carries its own validation. The deployed model is then
refitted on all labelled reactions -- pool and held-out test -- so predictions
use every measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config, data, features, holdout, predictor  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402


def checksum(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-set", default=config.SELECTED)
    p.add_argument("--family", default=config.SELECTED_FAMILY)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--features", type=Path, default=config.FEATURES_SMILES_XLSX,
                   help="descriptor workbook to train on (default: computed from SMILES)")
    p.add_argument("--output-dir", type=Path, default=config.MODEL_DIR)
    args = p.parse_args()

    reactions = data.load_reactions()
    acyl, amine = data.load_features(args.features)

    # validation of exactly this pipeline: fit the pool, score the held-out test
    split = holdout.build_split(reactions.table, acyl, amine)
    regime = split.label(reactions.table)
    pool = reactions.table[regime == "pool"].reset_index(drop=True)
    test = reactions.table[regime != "pool"].reset_index(drop=True)
    pool_design = features.build_design(pool, acyl, amine, args.feature_set)
    check = ReactivityRateModel(threshold=args.threshold, family=args.family).fit(
        pool_design.X, pool_design.rows.reactive, pool_design.rows.logk)
    scored = holdout.evaluate_test(
        check, features.build_design(test, acyl, amine, args.feature_set), split)
    validation = {"held_out_reactions": scored["classification"]["n"],
                  "accuracy": round(scored["classification"]["accuracy"], 4),
                  "precision": round(scored["classification"]["precision"], 4),
                  "recall": round(scored["classification"]["recall"], 4),
                  "rate_r2": round(scored["rate"]["r2"], 4),
                  "rate_mae": round(scored["rate"]["mae"], 4),
                  "rate_n": scored["rate"]["n"]}
    print(f"held-out check of this pipeline: accuracy {validation['accuracy']:.3f}  "
          f"rate R2 {validation['rate_r2']:+.3f}  MAE {validation['rate_mae']:.3f}")

    design = features.build_design(reactions.table, acyl, amine, args.feature_set)
    model = ReactivityRateModel(threshold=args.threshold, family=args.family).fit(
        design.X, design.rows.reactive, design.rows.logk)
    n_measured = int(np.isfinite(design.rows.logk.to_numpy(float)).sum())
    print(f"fitted {config.MODEL_FAMILIES[args.family]['label']} on "
          f"{len(design.rows)} reactions ({n_measured} with a rate constant), "
          f"{len(design.columns)} descriptors")

    training = {
        "reactions": len(design.rows), "with_rate_constant": n_measured,
        "acyl_chlorides": int(design.rows.acyl.nunique()),
        "amines": int(design.rows.amine.nunique()),
        "descriptors": args.features.name,
        "inputs": {p.name: checksum(p) for p in
                   (config.RATES_XLSX, config.EXCLUDED_XLSX, args.features)},
        "held_out_validation": validation,
    }
    out = predictor.save(model, design.columns, args.feature_set, args.output_dir, training)
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
