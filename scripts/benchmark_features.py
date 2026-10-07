#!/usr/bin/env python3
"""Rank the descriptor feature sets by leave-one-reactant-out cross-validation.

    python scripts/benchmark_features.py

The learner is held fixed (config.FEATURE_BENCHMARK_FAMILY) so the ranking
measures the descriptors alone. Nothing here reads the held-out test set or the
scopes. Outputs, in results/benchmark/feature_sets/:

    loro_benchmark.csv / .json     ranking of the feature sets
    out_of_fold_predictions.csv    cross-validated predictions, selected set
    parity_summary.csv             train and CV R2 / MAE per feature set
    session.json                   inputs, split and settings
    figures/
      candidate_ranking.png        joint score per feature set
      regime_breakdown.png         accuracy and R2 by unseen reactant
      parity_cv.png, confusion_cv.png       selected set, cross-validated
      parity_train_cv_<set>.png    train vs CV parity, one per feature set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import prism_ml  # noqa: E402
from prism_ml import benchmark, config, data, features, holdout, metrics, plotting, splits  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402


def checksum(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def parity_by_tier(pool, acyl, amine, tiers, family, threshold, figures) -> pd.DataFrame:
    """Train vs cross-validated parity for each feature set, same learner."""
    rows = []
    for tier in tiers:
        label = config.FEATURE_SETS[tier]["label"]
        design = features.build_design(pool, acyl, amine, tier)
        model = ReactivityRateModel(threshold=threshold, family=family).fit(
            design.X, design.rows.reactive, design.rows.logk)
        y = design.rows.logk.to_numpy(float)
        m = np.isfinite(y)
        oof = benchmark.collect_out_of_fold(design, splits.loro_folds(design.rows),
                                            threshold, family=family)
        series = {"train": (y[m], model.predict(design.X).log_k[m]),
                  "CV": (oof.logk[oof.measured], oof.predicted_logk[oof.measured])}
        plotting.plot_parity_splits(series, figures / f"parity_train_cv_{tier}.png")
        for name, (obs, pred) in series.items():
            rows.append(dict(feature_set=tier, label=label, family=family, split=name,
                             n=len(obs), r2=metrics.r2(obs, pred), mae=metrics.mae(obs, pred)))
        print(f"  {label:18} train R2 {rows[-2]['r2']:+.3f}  |  CV R2 {rows[-1]['r2']:+.3f}")
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=config.RESULTS_DIR / "benchmark" / "feature_sets")
    p.add_argument("--candidates", nargs="+", default=None,
                   help=f"default: {' '.join(config.BENCHMARK_ORDER)}")
    p.add_argument("--family", default=config.FEATURE_BENCHMARK_FAMILY)
    p.add_argument("--class-weight", type=float, default=0.5,
                   help="weight on classification in the joint score")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fold-seed", type=int, default=config.FOLD_SEED)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"feature-set benchmark (leave-one-reactant-out, learner: {args.family})")
    print("=" * 72)
    reactions = data.load_reactions()
    print(reactions.summary())
    acyl, amine = data.load_features()
    split = holdout.build_split(reactions.table, acyl, amine)
    print(f"held out: acyls {[int(i) for i in split.test_acyls]}, "
          f"amines {[int(i) for i in split.test_amines]}")

    ranking = benchmark.run_benchmark(
        reactions.table, acyl, amine, split, candidates=args.candidates,
        class_weight=args.class_weight, threshold=args.threshold,
        fold_seed=args.fold_seed, family=args.family)
    show = ["label", "cost", "n_features", "n_reactions",
            "accuracy_macro", "r2_macro", "mae_macro", "joint"]
    print("\n" + ranking[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = ranking.iloc[0]
    print(f"\nranked first: {best.label}  (selected in config: "
          f"{config.FEATURE_SETS[config.SELECTED]['label']})")

    metadata = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prism_ml_version": prism_ml.__version__,
        "python": platform.python_version(),
        "inputs": {name: checksum(path) for name, path in
                   (("rates", config.RATES_XLSX), ("excluded", config.EXCLUDED_XLSX),
                    ("features", config.FEATURES_XLSX))},
        "split": {"test_acyls": [int(i) for i in split.test_acyls],
                  "test_amines": [int(i) for i in split.test_amines],
                  "forced_train_amines": list(config.FORCE_TRAIN_AMINES)},
        "settings": {"family": args.family, "class_weight": args.class_weight,
                     "threshold": args.threshold, "fold_seed": args.fold_seed},
    }
    benchmark.write_results(ranking, out, metadata)
    (out / "session.json").write_text(json.dumps(metadata, indent=2))
    if args.no_figures:
        return 0

    figures = out / "figures"
    pool = reactions.table[split.label(reactions.table) == "pool"].reset_index(drop=True)
    design = features.build_design(pool, acyl, amine, config.SELECTED)
    oof = benchmark.collect_out_of_fold(
        design, splits.loro_folds(design.rows, seed=args.fold_seed), args.threshold,
        family=args.family)
    oof.frame().to_csv(out / "out_of_fold_predictions.csv", index=False)
    m = oof.measured
    plotting.plot_parity(oof.logk[m], oof.predicted_logk[m],
                         [k for k, keep in zip(oof.fold_kind, m) if keep],
                         figures / "parity_cv.png")
    plotting.plot_confusion(oof.reactive, oof.predicted, figures / "confusion_cv.png")
    plotting.plot_candidate_ranking(ranking, figures / "candidate_ranking.png",
                                    highlight=config.SELECTED)
    plotting.plot_regime_breakdown(ranking, figures / "regime_breakdown.png",
                                   highlight=config.SELECTED)

    print("\ntrain vs cross-validated parity per feature set:")
    tiers = args.candidates or list(config.BENCHMARK_ORDER)
    parity_by_tier(pool, acyl, amine, tiers, args.family, args.threshold,
                   figures).to_csv(out / "parity_summary.csv", index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
