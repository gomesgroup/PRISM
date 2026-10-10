#!/usr/bin/env python3
"""Chemical space of the reactants: all data, and the train / test split.

    python analysis/chemical_space.py

Covers every acyl chloride and amine in the labelled reactions. Test reactants
are the held-out ones of the study's split (scripts/holdout.py); the rest are
training. Fingerprints, scaffolds and metrics: prism_ml.chemical_space.
Acyl chloride fingerprints leave out the C(=O)Cl group they all share; the
summary also reports the acyl metrics with it kept, for comparison.
Needs RDKit (pip install -e ".[features]").

Outputs, in --output-dir:

    reactants.csv                    per reactant: split, functional class, scaffold,
                                     nearest training reactant and its similarity
    chemical_space_summary.json      all metrics
    chemical_space_summary.txt       the same, as text
    figures/
      heatmap_acyl_chlorides.png              all acyl chlorides, train | test
      heatmap_amines.png                      all amines, train | test
      diversity_distribution.png              pairwise diversity (1 - similarity),
                                              acyl chlorides and amines together
      similarity_distribution_all.png         all pairs, per reactant kind
      similarity_distribution_train_test.png  train-train, test-test, test-train
      nearest_training_similarity.png         test reactants against training
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

from prism_ml import chemical_space, data, holdout, plotting  # noqa: E402

KINDS = (("acyl", "Acyl chlorides", "acyl_chlorides"), ("amine", "Amines", "amines"))


def reactant_table(features: pd.DataFrame, ids, test_ids) -> pd.DataFrame:
    """Reactants in the reactions, training first then test, each by id."""
    table = features.loc[sorted(ids), ["name", "smiles"]].reset_index()
    table["split"] = np.where(table.id.isin(test_ids), "test", "train")
    return table.sort_values(["split", "id"], ascending=[False, True]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis" / "results" / "chemical_space")
    args = p.parse_args()
    out, figures = args.output_dir, args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plotting.set_plot_style()

    reactions = data.load_reactions().table
    acyl, amine = data.load_features()
    split = holdout.build_split(reactions, acyl, amine)
    tables = {"acyl": reactant_table(acyl, reactions.acyl.unique(), split.test_acyls),
              "amine": reactant_table(amine, reactions.amine.unique(), split.test_amines)}

    results = {kind: chemical_space.analyse(tables[kind], kind) for kind, _, _ in KINDS}

    for kind, title, stem in KINDS:
        r = results[kind]
        t, train, test = r["table"], r["train"], r["test"]
        labels = t.id.astype(str).tolist()
        is_test = (t.split == "test").tolist()
        plotting.plot_similarity_heatmap(
            r["similarity"], labels, labels, figures / f"heatmap_{stem}.png",
            title=f"{title}: all {len(t)} reactants "
                  f"(train {len(train)} | test {len(test)}, test in blue)",
            row_split=len(train), col_split=len(train), row_test=is_test, col_test=is_test)

    pairs = {title: {"all pairs": chemical_space.upper(results[kind]["similarity"])}
             for kind, title, _ in KINDS}
    plotting.plot_diversity_distribution(
        {title: 1 - pairs[title]["all pairs"] for _, title, _ in KINDS},
        {"Acyl chlorides": plotting.REGIME_STRONG["acyl"],
         "Amines": plotting.REGIME_STRONG["amine"]},
        figures / "diversity_distribution.png")
    plotting.plot_similarity_distributions(pairs, figures / "similarity_distribution_all.png")
    split_pairs = {}
    for kind, title, _ in KINDS:
        r = results[kind]
        s, train, test = r["similarity"], r["train"], r["test"]
        split_pairs[title] = {
            "train-train": chemical_space.upper(s[np.ix_(train, train)]),
            "test-test": chemical_space.upper(s[np.ix_(test, test)]),
            "test-train": s[np.ix_(test, train)].ravel()}
    plotting.plot_similarity_distributions(
        split_pairs, figures / "similarity_distribution_train_test.png")
    plotting.plot_nearest_training(
        {title: results[kind]["table"] for kind, title, _ in KINDS},
        figures / "nearest_training_similarity.png")

    frame = pd.concat([results[kind]["table"].assign(kind=kind) for kind, _, _ in KINDS])
    frame["nearest_train_similarity"] = frame.nearest_train_similarity.round(4)
    frame[["kind", "id", "name", "smiles", "split", "functional_class", "scaffold",
           "nearest_train_id", "nearest_train_similarity"]].to_csv(
        out / "reactants.csv", index=False)
    summary = {kind: results[kind]["summary"] for kind, _, _ in KINDS}
    # For reference: the acyl chlorides with their shared C(=O)Cl group kept.
    kept = chemical_space.analyse(tables["acyl"], "acyl", exclude_common=False)["summary"]
    summary["acyl_with_common_group"] = {
        k: kept[k] for k in ("similarity_all", "similarity_train_train",
                             "similarity_test_train", "nearest_train_test",
                             "nearest_train_train_loo", "identical_fingerprints")}
    summary["split"] = {"test_acyls": [int(i) for i in split.test_acyls],
                        "test_amines": [int(i) for i in split.test_amines]}
    (out / "chemical_space_summary.json").write_text(json.dumps(summary, indent=2))
    text = chemical_space.summary_text(results, n_reactions=len(reactions))
    ref, now = summary["acyl_with_common_group"], summary["acyl"]
    text += ("\n\nAcyl chlorides with the C(=O)Cl group kept, for comparison "
             "(kept -> left out):\n"
             + "\n".join(f"  {label}: {ref[k]['mean']:.3f} -> {now[k]['mean']:.3f}"
                         for label, k in (("mean pairwise similarity", "similarity_all"),
                                          ("train-train", "similarity_train_train"),
                                          ("test-train", "similarity_test_train"),
                                          ("nearest training reactant, test",
                                           "nearest_train_test"),
                                          ("nearest training reactant, training",
                                           "nearest_train_train_loo"))))
    (out / "chemical_space_summary.txt").write_text(text + "\n")
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
