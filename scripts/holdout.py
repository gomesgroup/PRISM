#!/usr/bin/env python3
"""Score the selected model on the held-out test, excluded scope and drug scope.

    python scripts/holdout.py                    # selected split model
    python scripts/holdout.py --family boosted_trees --output-dir results/holdout_hgb

The model is fitted once on the training pool and every reserved set is scored
with it. Outputs, in --output-dir:

    holdout_results.json          all metrics
    test_predictions.csv          held-out test, per pair; for pairs with a measured
                                  rate, the residual (measured - predicted log k) and
                                  whether residuals_test.png marks it an outlier (>2 SD)
                                  and the same on k itself: k, predicted_k, residual_k
    scope_predictions.csv         excluded substrate scope, per substrate
    drug_scope_predictions.csv    drug scope, per pair: class call (0/1 and reactive /
                                  un-reactive) and whether it is
                                  correct; rate constant k and log10 k, measured and
                                  predicted
    figures/
      confusion_test_scope_drug.png            three panels, one per set
      confusion_held-out-scope.png             all three sets pooled
      confusion_by_regime_held-out-scope.png   pooled, by which reactant is unseen
      confusion_by_novelty_held-out-scope.png  pooled, one vs two reactants unseen
      parity_test.png                          held-out rate constants
      parity_train_cv_test.png                 training, cross-validated, held-out
      residuals_test.png                       held-out residuals by regime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import benchmark, config, data, features, holdout, metrics, plotting, splits  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402

REGIME_TITLES = (("new_acyl", "Unseen acyl chloride"), ("new_amine", "Unseen amine"),
                 ("both_new", "Both unseen"))


def _report(truth, pred) -> dict:
    return metrics.classification_report(np.asarray(truth, int), np.asarray(pred, int))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-set", default=config.SELECTED)
    p.add_argument("--family", default=config.SELECTED_FAMILY,
                   help=f"learner family (default: {config.SELECTED_FAMILY})")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR / "holdout")
    p.add_argument("--dpi", type=int, default=600, help="figure resolution")
    args = p.parse_args()
    plotting.SAVE_DPI = args.dpi
    out, figures = args.output_dir, args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"held-out evaluation: {config.FEATURE_SETS[args.feature_set]['label']}, "
          f"{config.MODEL_FAMILIES[args.family]['label']}")
    print("=" * 72)
    reactions = data.load_reactions()
    acyl, amine = data.load_features()
    split = holdout.build_split(reactions.table, acyl, amine)
    regime = split.label(reactions.table)
    pool = reactions.table[regime == "pool"].reset_index(drop=True)
    test = reactions.table[regime != "pool"].reset_index(drop=True)
    print(f"pool {len(pool)} reactions; test {len(test)} "
          f"(acyls {[int(i) for i in split.test_acyls]}, "
          f"amines {[int(i) for i in split.test_amines]})")

    pool_design = features.build_design(pool, acyl, amine, args.feature_set)
    test_design = features.build_design(test, acyl, amine, args.feature_set)
    model = ReactivityRateModel(threshold=args.threshold, family=args.family).fit(
        pool_design.X, pool_design.rows.reactive, pool_design.rows.logk)

    # ---- the three reserved sets, one fitted model ----
    results = {"feature_set": args.feature_set, "family": args.family,
               "test": holdout.evaluate_test(model, test_design, split)}
    test_frame = holdout.test_frame(model, test_design, split)
    scope = holdout.score_scope(model, acyl, amine, args.feature_set, split)
    drug = holdout.score_drug(model, acyl, amine, args.feature_set, split)
    scored_scope = scope[scope.truth.notna()]

    t = results["test"]
    c = t["classification"]
    print(f"\nheld-out test   n={c['n']}  accuracy {c['accuracy']:.3f} "
          f"(majority {c['majority']:.3f})  precision {c['precision']:.3f}  "
          f"recall {c['recall']:.3f}")
    print(f"  rate constant n={t['rate']['n']}  R2 {t['rate']['r2']:+.3f}  "
          f"MAE {t['rate']['mae']:.3f}  90% band {t['residual_90pc_band']:.2f}  "
          f"coverage {t['coverage']:.3f}")

    scope_report = _report(scored_scope.truth, scored_scope.predicted)
    by_note = {}
    for note in holdout.NOTE_ORDER:
        sub = scope[scope.note == note]
        if sub.empty:
            continue
        scored = sub[sub.truth.notna()]
        by_note[note] = {"n": len(sub), "predicted_reactive": int(sub.predicted.sum()),
                         "correct": (f"{int((scored.predicted == scored.truth).sum())}/"
                                     f"{len(scored)}" if len(scored) else "not scored")}
    results["scope"] = {"report": scope_report, "by_note": by_note,
                        "n_substrates": len(scope), "n_scored": len(scored_scope)}
    print(f"excluded scope  n={scope_report['n']} scored of {len(scope)}  "
          f"accuracy {scope_report['accuracy']:.3f}")

    drug_report = _report(drug.truth, drug.predicted)
    measured = drug.logk.notna()
    results["drug"] = {
        "report": drug_report,
        "rate_mae_measured": (metrics.mae(drug.logk[measured], drug.predicted_logk[measured])
                              if measured.any() else None),
        "n_measured": int(measured.sum())}
    print(f"drug scope      n={drug_report['n']}  correct "
          f"{drug_report['tp'] + drug_report['tn']}/{drug_report['n']}"
          f"  (every drug pair is reactive, so this is recall)")

    # ---- pooled over all three ----
    calls = holdout.pooled_calls(test_frame, scope, drug)
    pooled = _report(calls.truth, calls.pred)
    by_regime = {}
    for key, title in REGIME_TITLES:
        m = calls.regime == key
        if m.any():
            by_regime[title] = (calls.truth[m].to_numpy(), calls.pred[m].to_numpy())
    single = calls.regime != "both_new"
    novelty = {"One reactant unseen": (calls.truth[single].to_numpy(), calls.pred[single].to_numpy())}
    if (~single).any():
        novelty["Two reactants unseen"] = (calls.truth[~single].to_numpy(),
                                        calls.pred[~single].to_numpy())
    results["pooled"] = {"report": pooled, "by_regime": {
        title: _report(tr, pr) for title, (tr, pr) in {**by_regime, **novelty}.items()}}
    print(f"pooled          n={pooled['n']}  accuracy {pooled['accuracy']:.3f}  "
          f"precision {pooled['precision']:.3f}  recall {pooled['recall']:.3f}")

    # ---- figures ----
    sizes = (f"held-out test {len(test_frame)} + excluded scope {len(scored_scope)} + "
             f"drug scope {len(drug)} = {len(calls)} reactivity calls on unseen reactions")
    plotting.plot_confusion_panels(
        {"Held-out test": (test_frame.reactive.to_numpy(), test_frame.predicted.to_numpy()),
         "Excluded scope": (scored_scope.truth.astype(int).to_numpy(),
                            scored_scope.predicted.to_numpy()),
         "Drug scope": (drug.truth.to_numpy(), drug.predicted.to_numpy())},
        figures / "confusion_test_scope_drug.png")
    plotting.plot_confusion(calls.truth.to_numpy(), calls.pred.to_numpy(),
                            figures / "confusion_held-out-scope.png", note=sizes)
    plotting.plot_confusion_panels(by_regime, figures / "confusion_by_regime_held-out-scope.png")
    plotting.plot_confusion_panels(novelty, figures / "confusion_by_novelty_held-out-scope.png")

    y = test_frame.logk.to_numpy(float)
    m = np.isfinite(y)
    kinds = test_frame.regime.map(holdout.REGIME_KIND).to_numpy()
    yhat = test_frame.predicted_logk.to_numpy(float)
    plotting.plot_parity(y[m], yhat[m], kinds[m], figures / "parity_test.png")
    plotting.plot_residuals(y[m], yhat[m], figures / "residuals_test.png",
                            names=test_frame.reaction.to_numpy()[m], kinds=kinds[m])
    # The residuals plotted above, per pair, in test_predictions.csv.
    residual, outlier = plotting.residual_outliers(y[m], yhat[m])
    test_frame["residual"] = np.nan
    test_frame.loc[m, "residual"] = residual
    test_frame["residual_outlier"] = pd.Series(pd.NA, index=test_frame.index, dtype="boolean")
    test_frame.loc[m, "residual_outlier"] = outlier
    # The same on the rate constant itself (not log): k, predicted k, and
    # k_measured - k_predicted, in the units of k_corrected.
    k, k_hat = 10.0 ** test_frame.logk, 10.0 ** test_frame.predicted_logk
    sig4 = lambda v: v.map(lambda x: float(f"{x:.4g}") if np.isfinite(x) else np.nan)
    test_frame["k"], test_frame["predicted_k"] = sig4(k), sig4(k_hat)
    test_frame["residual_k"] = sig4(k - k_hat)

    # Training, cross-validated and held-out on one panel. The CV predictions
    # are refitted per fold with the same learner, so all three describe it.
    pool_y = pool_design.rows.logk.to_numpy(float)
    pm = np.isfinite(pool_y)
    oof = benchmark.collect_out_of_fold(pool_design, splits.loro_folds(pool_design.rows),
                                        args.threshold, family=args.family)
    plotting.plot_parity_splits(
        {"train": (pool_y[pm], model.predict(pool_design.X).log_k[pm]),
         "CV": (oof.logk[oof.measured], oof.predicted_logk[oof.measured]),
         "test": (y[m], yhat[m])},
        figures / "parity_train_cv_test.png")

    # ---- tables ----
    results["split"] = {"test_acyls": [int(i) for i in split.test_acyls],
                        "test_amines": [int(i) for i in split.test_amines],
                        "partner_acyl": holdout.PARTNER_ACYL,
                        "partner_amine": holdout.PARTNER_AMINE}
    test_frame.to_csv(out / "test_predictions.csv", index=False)
    scope.to_csv(out / "scope_predictions.csv", index=False)
    drug.to_csv(out / "drug_scope_predictions.csv", index=False)
    with open(out / "holdout_results.json", "w") as handle:
        json.dump(results, handle, indent=2, default=float)
    print(f"\nwritten to {out}")
    for path in sorted(figures.glob("*.png")):
        print(f"  {path.relative_to(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
