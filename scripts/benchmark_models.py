#!/usr/bin/env python3
"""Compare every learner family on the selected feature set.

    python scripts/benchmark_models.py                 # all families
    python scripts/benchmark_models.py --skip-graph    # without the graph networks

The feature benchmark holds the learner fixed and varies the descriptors; this
does the reverse. Same folds, same reactions, same training pool, so a
difference is the learner and nothing else. It never reads the held-out test
set or the scopes.

Families (src/prism_ml/config.py, MODEL_FAMILIES): gradient-boosting variants
(scikit-learn histogram and exact, AdaBoost, XGBoost, LightGBM, CatBoost),
averaging ensembles (random forest, extremely randomised trees), linear,
k-nearest neighbours, two graph networks, an uninformative baseline, and the
selected split model. The graph networks need torch and are slow without a GPU.

Outputs, in results/benchmark/model_families/: model_families.csv and
figures/model_families.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import benchmark, config, data, holdout, plotting  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-set", default=config.SELECTED)
    p.add_argument("--families", nargs="+", default=None,
                   help="default: every family in config.MODEL_FAMILIES")
    p.add_argument("--skip-graph", action="store_true",
                   help="leave out the graph-network families (need torch)")
    p.add_argument("--class-weight", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fold-seed", type=int, default=config.FOLD_SEED)
    p.add_argument("--output-dir", type=Path,
                   default=config.RESULTS_DIR / "benchmark" / "model_families")
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    families = args.families or list(config.MODEL_FAMILIES)
    if args.skip_graph:
        families = [f for f in families if not config.MODEL_FAMILIES[f].get("needs_graphs")]

    print("=" * 72)
    print(f"learner comparison on {config.FEATURE_SETS[args.feature_set]['label']} "
          f"({len(families)} families)")
    print("=" * 72)
    reactions = data.load_reactions()
    acyl, amine = data.load_features()
    split = holdout.build_split(reactions.table, acyl, amine)
    frame = benchmark.compare_model_families(
        reactions.table, acyl, amine, split, feature_set=args.feature_set,
        families=families, class_weight=args.class_weight,
        threshold=args.threshold, fold_seed=args.fold_seed)

    show = ["label", "accuracy_macro", "r2_macro", "mae_macro", "joint"]
    print("\n" + frame[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nbest classifier (CV accuracy): "
          f"{frame.loc[frame.accuracy_macro.idxmax(), 'label']}")
    print(f"best regressor  (CV R2):       {frame.loc[frame.r2_macro.idxmax(), 'label']}")
    frame.to_csv(out / "model_families.csv", index=False)
    plotting.plot_candidate_ranking(frame, out / "figures" / "model_families.png",
                                    highlight=config.SELECTED_FAMILY)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
