"""The class-stratified train/test split, and leave-one-reactant-out folds.

Every test reactant is unseen, so the held-out set measures cold start rather
than interpolation. Model selection never sees it: candidates are ranked by
leave-one-reactant-out over the training pool alone.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

Fold = Tuple[str, np.ndarray, np.ndarray]   # kind, train positions, held positions


@dataclass(frozen=True)
class Split:
    test_acyls: Tuple[int, ...]
    test_amines: Tuple[int, ...]
    untested_classes: Dict[str, List[int]] = field(default_factory=dict)

    def regime(self, acyl: int, amine: int) -> str:
        a, m = acyl in self.test_acyls, amine in self.test_amines
        if a and m:
            return "both_new"
        if a:
            return "new_acyl"
        if m:
            return "new_amine"
        return "pool"

    def label(self, reactions: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [self.regime(a, m) for a, m in zip(reactions.acyl, reactions.amine)],
            index=reactions.index, name="regime")


def _picks_for_class(n: int) -> int:
    """How many test representatives a class of ``n`` reactants contributes."""
    if n <= config.SMALL_CLASS:
        return 0
    return 2 if n > config.LARGE_CLASS else 1


def _spaced(members: Sequence[int], measurable: Dict[int, int], k: int) -> List[int]:
    """``k`` evenly spaced order statistics of the measurable-reaction count.

    Representatives are typical members of their class, not the best- or
    worst-measured ones; ``k == 1`` is the median.
    """
    if k <= 0:
        return []
    ordered = sorted(members, key=lambda i: (measurable.get(i, 0), i))
    return [ordered[int(round((j + 1) * len(ordered) / (k + 1))) - 1] for j in range(k)]


def build_split(reactions: pd.DataFrame, acyl_class: Dict[int, str],
                amine_class: Dict[int, str],
                force_train_acyls: Sequence[int] = (),
                force_train_amines: Sequence[int] = ()) -> Split:
    """Choose held-out reactants, sized by how well populated each class is.

    A forced reactant still counts towards its class's population — it is in
    the pool, so leave-one-out can hold a sibling out with it still in
    training — but cannot itself be a representative.
    """
    picks: Dict[str, List[int]] = {}
    untested: Dict[str, List[int]] = {}
    measurable = {
        "acyl": reactions[reactions.rate_class == "measurable"].acyl.value_counts().to_dict(),
        "amine": reactions[reactions.rate_class == "measurable"].amine.value_counts().to_dict()}

    for kind, classes, forced in (("acyl", acyl_class, set(force_train_acyls)),
                                  ("amine", amine_class, set(force_train_amines))):
        present = sorted(reactions[kind].unique())
        by_class = defaultdict(list)
        for i in present:
            by_class[classes.get(i, "unknown")].append(i)
        chosen: List[int] = []
        for name, members in sorted(by_class.items()):
            eligible = [i for i in members if i not in forced]
            k = min(_picks_for_class(len(members)), len(eligible))
            if k == 0:
                untested[f"{kind}:{name}"] = members
                continue
            chosen.extend(_spaced(eligible, measurable[kind], k))
        picks[kind] = sorted(chosen)

    return Split(test_acyls=tuple(picks["acyl"]), test_amines=tuple(picks["amine"]),
                 untested_classes=untested)


def loro_folds(rows: pd.DataFrame, min_rows: int = None, pair_folds: int = None,
               seed: int = None) -> List[Fold]:
    """Leave-one-reactant-out folds over ``rows``, plus leave-one-pair-out.

    Three kinds:

    ``acyl`` / ``amine``
        every reactant with at least ``min_rows`` reactions, held out in turn.
        Exhaustive and therefore seed-independent.
    ``pair``
        an acyl and an amine withheld together, scoring only the reactions
        where both are novel — the in-pool analogue of the ``both_new`` test
        regime. Sampled, and the only part of the procedure that uses ``seed``.
    """
    min_rows = config.MIN_FOLD_ROWS if min_rows is None else min_rows
    pair_folds = config.PAIR_FOLDS if pair_folds is None else pair_folds
    seed = config.FOLD_SEED if seed is None else seed

    positions = np.arange(len(rows))
    acyls, amines = rows.acyl.to_numpy(), rows.amine.to_numpy()
    folds: List[Fold] = []
    for kind, values in (("acyl", acyls), ("amine", amines)):
        for i in sorted(np.unique(values)):
            held = positions[values == i]
            if len(held) >= min_rows:
                folds.append((kind, positions[values != i], held))
    if pair_folds:
        rng = np.random.default_rng(seed)
        uniq_a, uniq_m = sorted(np.unique(acyls)), sorted(np.unique(amines))
        for _ in range(pair_folds):
            a, m = int(rng.choice(uniq_a)), int(rng.choice(uniq_m))
            held = positions[(acyls == a) & (amines == m)]
            train = positions[(acyls != a) & (amines != m)]
            if len(held) and len(train) > 20:
                folds.append(("pair", train, held))
    return folds


def stratified_folds(rows: pd.DataFrame, acyl_class: Dict[int, str],
                     amine_class: Dict[int, str], draws: int = None,
                     first_seed: int = None, force_train_acyls: Sequence[int] = (),
                     force_train_amines: Sequence[int] = ()) -> List[Fold]:
    """Repeated class-stratified validation folds over ``rows``.

    Each draw holds out reactants by the same class rule as the test set
    (:func:`build_split`: classes of <= SMALL_CLASS members contribute none,
    larger ones one, classes above LARGE_CLASS two), but chooses the members at
    random, with seed ``first_seed + i``. Every reaction involving a held-out
    acyl chloride or amine is validation; the rest is training.
    """
    draws = config.STRATIFIED_DRAWS if draws is None else draws
    first_seed = config.STRATIFIED_FIRST_SEED if first_seed is None else first_seed
    positions = np.arange(len(rows))
    folds: List[Fold] = []
    for i in range(draws):
        rng = np.random.default_rng(first_seed + i)
        held_out = {}
        for kind, classes, forced in (("acyl", acyl_class, set(force_train_acyls)),
                                      ("amine", amine_class, set(force_train_amines))):
            by_class = defaultdict(list)
            for r in sorted(np.unique(rows[kind])):
                by_class[classes.get(int(r), "unknown")].append(int(r))
            chosen: List[int] = []
            for _, members in sorted(by_class.items()):
                eligible = [m for m in members if m not in forced]
                k = min(_picks_for_class(len(members)), len(eligible))
                if k:
                    chosen += [int(v) for v in rng.choice(eligible, size=k, replace=False)]
            held_out[kind] = chosen
        held = (rows.acyl.isin(held_out["acyl"]) | rows.amine.isin(held_out["amine"])).to_numpy()
        folds.append(("stratified", positions[~held], positions[held]))
    return folds


def fold_summary(folds: Sequence[Fold]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for kind, _, _ in folds:
        out[kind] += 1
    return dict(out)
