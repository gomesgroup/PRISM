"""The cross-validated comparisons that select the feature set and the learner.

Every candidate is scored two ways on the training pool: leave-one-reactant-out
(each acyl chloride and amine in turn) and repeated class-stratified validation
(splits.stratified_folds: several reactants held out at once, as in the test
set). config.SELECTION_CV decides which ranks the candidates.

This module never touches the held-out test set or the excluded substrate
scope. Selection happens here, on the training pool alone; the reserved sets
are scored once, afterwards, by a separate entry point. Keeping them apart is
the point of the exercise — a benchmark that has seen the test set cannot
justify the choice it recommends.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config, features, metrics, splits
from .model import ReactivityRateModel

#: Below this many shared reactions the folds stop being meaningful and the
#: benchmark refuses rather than reporting NaN.
MIN_SCORABLE = 200


@dataclass
class CandidateResult:
    """Cross-validated performance of one feature set."""

    name: str
    label: str
    cost: str
    n_features: int
    n_acyl: int
    n_amine: int
    n_reactions: int
    n_dropped: int
    n_covered: int = 0
    accuracy_acyl: float = float("nan")
    accuracy_amine: float = float("nan")
    accuracy_pair: float = float("nan")
    accuracy_macro: float = float("nan")
    r2_acyl: float = float("nan")
    r2_amine: float = float("nan")
    r2_macro: float = float("nan")
    mae_acyl: float = float("nan")
    mae_amine: float = float("nan")
    mae_macro: float = float("nan")
    joint_loro: float = float("nan")
    accuracy_stratified: float = float("nan")
    r2_stratified: float = float("nan")
    joint_stratified: float = float("nan")
    joint_stratified_sd: float = float("nan")
    joint: float = float("nan")
    folds: Dict[str, int] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["folds"] = ",".join(f"{k}={v}" for k, v in sorted(self.folds.items()))
        return d


def _score_candidate(name: str, design: features.Design,
                     fold_list: Sequence[splits.Fold],
                     class_weight: float, threshold: float,
                     family: str = None, label: str = None,
                     cost: str = None, graphs: Optional[tuple] = None
                     ) -> CandidateResult:
    """Cross-validate one candidate.

    ``name`` identifies the row; ``label`` and ``cost`` describe it. They are
    passed in rather than looked up, because this serves both comparisons — one
    varying the feature set, one varying the learner — and only the caller
    knows which registry ``name`` belongs to.
    """
    rows = design.rows
    reactive = rows.reactive.to_numpy(dtype=int)
    logk = rows.logk.to_numpy(dtype=float)

    acc: Dict[str, List[float]] = {"acyl": [], "amine": [], "pair": []}
    strat: List[tuple] = []          # (accuracy, R2) per stratified draw
    rate: Dict[str, List[np.ndarray]] = {"acyl": [[], []], "amine": [[], []],
                                         "pair": [[], []]}
    for kind, train_idx, held_idx in fold_list:
        if len(train_idx) < 20 or not len(held_idx):
            continue
        model = ReactivityRateModel(threshold=threshold,
                                    family=family or config.SELECTED_FAMILY,
                                    graphs=graphs)
        train_rows = rows.iloc[train_idx].reset_index(drop=True)
        held_rows = rows.iloc[held_idx].reset_index(drop=True)
        model.fit(design.X[train_idx], reactive[train_idx], logk[train_idx],
                  rows=train_rows)
        pred = model.predict(design.X[held_idx], rows=held_rows)
        measured = np.isfinite(logk[held_idx])
        if kind == "stratified":
            strat.append((metrics.accuracy(reactive[held_idx], pred.reactive),
                          metrics.r2(logk[held_idx][measured], pred.log_k[measured])
                          if measured.sum() > 2 else float("nan")))
            continue
        acc[kind].append(metrics.accuracy(reactive[held_idx], pred.reactive))
        if measured.any():
            rate[kind][0].extend(pred.log_k[measured])
            rate[kind][1].extend(logk[held_idx][measured])

    result = CandidateResult(
        name=name, label=label or name, cost=cost or "",
        n_features=len(design.columns), n_acyl=design.n_acyl,
        n_amine=design.n_amine, n_reactions=len(rows), n_dropped=design.dropped,
        folds=splits.fold_summary(fold_list))
    for kind in ("acyl", "amine", "pair"):
        if acc[kind]:
            setattr(result, f"accuracy_{kind}", float(np.mean(acc[kind])))
        predicted, truth = np.array(rate[kind][0]), np.array(rate[kind][1])
        if kind != "pair":
            setattr(result, f"r2_{kind}", metrics.r2(truth, predicted))
            setattr(result, f"mae_{kind}", metrics.mae(truth, predicted))
    # The macro is over the two exhaustive fold families. Pair folds are
    # sampled and reported, but kept out of the ranking so that the score does
    # not depend on the seed.
    result.accuracy_macro = float(np.mean([result.accuracy_acyl, result.accuracy_amine]))
    result.r2_macro = float(np.mean([result.r2_acyl, result.r2_amine]))
    result.mae_macro = float(np.mean([result.mae_acyl, result.mae_amine]))
    result.joint_loro = metrics.joint_score(result.accuracy_macro, result.r2_macro,
                                            class_weight)
    if strat:
        a, r = np.array(strat).T
        per_draw = [metrics.joint_score(x, y, class_weight) for x, y in zip(a, r)]
        result.accuracy_stratified = float(np.mean(a))
        result.r2_stratified = float(np.nanmean(r))
        result.joint_stratified = float(np.mean(per_draw))
        result.joint_stratified_sd = float(np.std(per_draw, ddof=1))
    result.joint = (result.joint_stratified if config.SELECTION_CV == "stratified" and strat
                    else result.joint_loro)
    return result


def _reactant_classes(table: pd.DataFrame) -> Dict[int, str]:
    return {int(i): str(c) for i, c in zip(table.index, table["class"])
            if isinstance(c, str) and c}


def selection_folds(rows: pd.DataFrame, acyl: pd.DataFrame, amine: pd.DataFrame,
                    fold_seed: int = None) -> List[splits.Fold]:
    """Leave-one-reactant-out folds plus the repeated class-stratified draws."""
    return (splits.loro_folds(rows, seed=fold_seed)
            + splits.stratified_folds(rows, _reactant_classes(acyl), _reactant_classes(amine),
                                      force_train_acyls=config.FORCE_TRAIN_ACYLS,
                                      force_train_amines=config.FORCE_TRAIN_AMINES))


def run_benchmark(reactions: pd.DataFrame, acyl: pd.DataFrame, amine: pd.DataFrame,
                  split: splits.Split, candidates: Optional[Sequence[str]] = None,
                  class_weight: float = 0.5, threshold: float = 0.5,
                  with_pka: bool = False, fold_seed: int = None,
                  common_subset: bool = True, family: Optional[str] = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Rank feature sets by cross-validated performance on the training pool.

    The learner is held fixed (``family``, default
    :data:`config.FEATURE_BENCHMARK_FAMILY`) so that the ranking measures the
    descriptors alone.

    Args:
        common_subset: score every candidate on the reactions all of them
            cover. Leave this on. With it off, a feature set that cannot
            describe some reactant is scored on an easier problem than its
            rivals, and the ranking rewards that.
    """
    regime = split.label(reactions)
    pool = reactions[regime == "pool"].reset_index(drop=True)
    if verbose:
        print(f"training pool: {len(pool)} reactions "
              f"({int(pool.rate_class.eq('measurable').sum())} with a measured rate constant)")
        print(f"held out and untouched here: "
              f"acyls {list(split.test_acyls)}, amines {list(split.test_amines)}")

    names = list(candidates or config.BENCHMARK_ORDER)
    designs = {n: features.build_design(pool, acyl, amine, n, with_pka=with_pka)
               for n in names}
    for name, design in designs.items():
        if verbose and design.dropped:
            print(f"  {spec_label(name)}: covers {len(design.rows)} of {len(pool)} "
                  f"reactions ({design.dropped} lack a required descriptor)")
    if common_subset:
        shared = features.common_reactions(list(designs.values()))
        if len(shared) < len(pool) and verbose:
            print(f"  scoring all candidates on the {len(shared)} reactions every "
                  f"one covers, so the comparison is like for like")
        if len(shared) < MIN_SCORABLE:
            raise ValueError(
                f"only {len(shared)} of {len(pool)} reactions are covered by every "
                f"candidate, which is too few to cross-validate. The usual cause "
                f"is a sparse descriptor: check the coverage report above and drop "
                f"the candidate or the column responsible.")
        designs = {n: features.restrict(d, shared) for n, d in designs.items()}

    results = []
    for name in names:
        design = designs[name]
        spec = config.FEATURE_SETS[name]
        fold_list = selection_folds(design.rows, acyl, amine, fold_seed)
        result = _score_candidate(name, design, fold_list, class_weight, threshold,
                                  family=family or config.FEATURE_BENCHMARK_FAMILY,
                                  label=spec["label"], cost=spec["cost"])
        result.n_covered = len(features.build_design(
            pool, acyl, amine, name, with_pka=with_pka).rows)
        results.append(result)
        if verbose:
            print(f"  {spec_label(name):18s} {result.n_features:3d} features  "
                  f"n={result.n_reactions:4d}  joint {result.joint:.4f}")
    frame = pd.DataFrame([r.as_row() for r in results])
    return frame.sort_values("joint", ascending=False).reset_index(drop=True)


@dataclass
class OutOfFold:
    """Held-out predictions gathered across every fold, for plotting."""

    reaction: List[str]
    fold_kind: List[str]
    reactive: np.ndarray
    predicted: np.ndarray
    probability: np.ndarray
    logk: np.ndarray
    predicted_logk: np.ndarray

    @property
    def measured(self) -> np.ndarray:
        """Mask of rows carrying a measured rate constant to compare against."""
        return np.isfinite(self.logk)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "reaction": self.reaction, "fold_kind": self.fold_kind,
            "reactive": self.reactive, "predicted": self.predicted,
            "probability": self.probability, "logk": self.logk,
            "predicted_logk": self.predicted_logk})


def collect_out_of_fold(design: features.Design,
                        fold_list: Sequence[splits.Fold],
                        threshold: float = 0.5,
                        family: Optional[str] = None) -> OutOfFold:
    """Re-run the folds, keeping each held-out prediction.

    ``family`` must match the model whose training and test predictions the
    out-of-fold ones are plotted beside; left unset it is the configured
    default, which once silently put gradient-boosting CV next to a random
    forest's training and test points.

    A reaction appears once per fold that holds it out, so a pair withheld by
    both its acyl fold and its amine fold contributes two rows. That is what
    the cross-validated figures show, and it is why their point count exceeds
    the number of reactions.
    """
    rows = design.rows
    reactive = rows.reactive.to_numpy(dtype=int)
    logk = rows.logk.to_numpy(dtype=float)
    reaction = rows.reaction.tolist()

    parts: Dict[str, list] = {k: [] for k in
                              ("reaction", "fold_kind", "reactive", "predicted",
                               "probability", "logk", "predicted_logk")}
    for kind, train_idx, held_idx in fold_list:
        if len(train_idx) < 20 or not len(held_idx):
            continue
        model = ReactivityRateModel(threshold=threshold, family=family)
        model.fit(design.X[train_idx], reactive[train_idx], logk[train_idx],
                  rows=rows.iloc[train_idx].reset_index(drop=True))
        pred = model.predict(design.X[held_idx],
                             rows=rows.iloc[held_idx].reset_index(drop=True))
        parts["reaction"].extend(reaction[i] for i in held_idx)
        parts["fold_kind"].extend([kind] * len(held_idx))
        parts["reactive"].extend(reactive[held_idx])
        parts["predicted"].extend(pred.reactive)
        parts["probability"].extend(pred.probability)
        parts["logk"].extend(logk[held_idx])
        parts["predicted_logk"].extend(pred.log_k)
    return OutOfFold(reaction=parts["reaction"], fold_kind=parts["fold_kind"],
                     reactive=np.array(parts["reactive"], dtype=int),
                     predicted=np.array(parts["predicted"], dtype=int),
                     probability=np.array(parts["probability"], dtype=float),
                     logk=np.array(parts["logk"], dtype=float),
                     predicted_logk=np.array(parts["predicted_logk"], dtype=float))


def compare_model_families(reactions: pd.DataFrame, acyl: pd.DataFrame,
                           amine: pd.DataFrame, split: splits.Split,
                           feature_set: Optional[str] = None,
                           families: Optional[Sequence[str]] = None,
                           class_weight: float = 0.5, threshold: float = 0.5,
                           fold_seed: int = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Score several learner families on one feature set, under the same folds.

    Holding the descriptors and the folds fixed isolates the learner: any
    difference here is the model family and nothing else. The feature-set
    comparison does the reverse, holding the learner fixed.
    """
    feature_set = feature_set or config.SELECTED
    regime = split.label(reactions)
    pool = reactions[regime == "pool"].reset_index(drop=True)
    design = features.build_design(pool, acyl, amine, feature_set)
    fold_list = selection_folds(design.rows, acyl, amine, fold_seed)

    names = list(families or config.MODEL_FAMILIES)
    graph_sources = {config.MODEL_FAMILIES[n].get("graph_source")
                     for n in names if config.MODEL_FAMILIES[n].get("needs_graphs")}
    graphs_by_source = {}
    for source in sorted(s_ for s_ in graph_sources if s_):
        from . import graph as graph_mod
        if source == "aimnet2":
            store = graph_mod.load_aimnet_graphs(config.AIMNET_NPZ)
            graphs_by_source[source] = (store["acyl"], store["amine"])
        else:
            graphs_by_source[source] = (graph_mod.build_graphs(acyl, "acyl"),
                                        graph_mod.build_graphs(amine, "amine"))
        if verbose:
            a, m = graphs_by_source[source]
            print(f"  graphs [{source}]: {len(a)} acyl chlorides, {len(m)} amines")

    # A graph source that does not cover every reactant would otherwise score
    # its family on an easier subset than the rest. Restrict everything to the
    # reactions all the requested families can describe.
    if graphs_by_source:
        covered = np.ones(len(design.rows), dtype=bool)
        for a, m in graphs_by_source.values():
            covered &= np.array([int(x) in a and int(y) in m
                                 for x, y in zip(design.rows.acyl, design.rows.amine)])
        if not covered.all():
            kept = design.rows.reaction[covered].tolist()
            if verbose:
                print(f"  {int((~covered).sum())} reaction(s) lack a graph; scoring "
                      f"every family on the {len(kept)} all of them cover")
            design = features.restrict(design, kept)
            fold_list = selection_folds(design.rows, acyl, amine, fold_seed)
    if verbose:
        print(f"learner comparison on {config.FEATURE_SETS[feature_set]['label']} "
              f"({len(design.columns)} descriptors, {len(design.rows)} reactions)")

    out_rows = []
    for name in names:
        spec = config.MODEL_FAMILIES[name]
        result = _score_candidate(
            name, design, fold_list, class_weight, threshold, family=name,
            label=spec["label"], cost=spec["note"],
            graphs=graphs_by_source.get(spec.get("graph_source")))
        out_rows.append(result.as_row())
        if verbose:
            print(f"  {spec['label']:28s} accuracy {result.accuracy_macro:.4f}  "
                  f"R2 {result.r2_macro:+.4f}  score {result.joint:.4f}")
    frame = pd.DataFrame(out_rows)
    return frame.sort_values("joint", ascending=False).reset_index(drop=True)


def spec_label(name: str) -> str:
    return config.FEATURE_SETS[name]["label"]


def write_results(frame: pd.DataFrame, output_dir: Path,
                  metadata: Optional[dict] = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "loro_benchmark.csv", index=False)
    payload = {"results": frame.to_dict(orient="records"), **(metadata or {})}
    with open(output_dir / "loro_benchmark.json", "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
