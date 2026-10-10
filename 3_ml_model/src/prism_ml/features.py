"""Design matrices, one per feature set.

A reaction is described by concatenating its acyl chloride's descriptors with
its amine's. There is no pair-level term; that is a known limitation of this
representation rather than an oversight, and it is where the classifier's
remaining errors concentrate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config

#: Columns that identify or annotate a reactant rather than describe it.
NON_FEATURE = ("id", "name", "class", "smiles", "split_type", "source")
#: Provenance columns emitted by the feature generator, useful for auditing a
#: reaction centre assignment and meaningless as model input.
PROVENANCE_PREFIX = ("idx_", "el_")


@dataclass(frozen=True)
class Design:
    """A fitted-ready matrix plus everything needed to interpret it."""

    X: np.ndarray
    rows: pd.DataFrame          #: the reactions actually covered
    columns: List[str]          #: feature names, acyl block then amine block
    n_acyl: int                 #: width of the acyl block
    dropped: int                #: reactions excluded for incomplete features

    @property
    def n_amine(self) -> int:
        return len(self.columns) - self.n_acyl


def select_columns(df: pd.DataFrame, include: str,
                   with_pka: bool = False) -> List[str]:
    """Descriptor columns of ``df`` matching ``include``, in table order."""
    pattern = re.compile(include)
    cols = [c for c in df.columns
            if c not in NON_FEATURE
            and not c.startswith(PROVENANCE_PREFIX)
            and pattern.match(c)]
    if not with_pka:
        pka = re.compile(config.PKA_PATTERN)
        cols = [c for c in cols if not pka.match(c)]
    elif not any(re.match(config.PKA_PATTERN, c) for c in cols):
        cols += [c for c in df.columns
                 if re.match(config.PKA_PATTERN, c) and c not in NON_FEATURE]
    return cols


def build_design(reactions: pd.DataFrame, acyl: pd.DataFrame, amine: pd.DataFrame,
                 feature_set: str, with_pka: bool = False) -> Design:
    """Assemble the design matrix for one feature set.

    Reactions whose reactants lack any selected column are dropped, not
    imputed: a blank can mean "this atom does not exist" or "this quantity was
    never computed", and silently filling either with zero puts the model on a
    scale it did not train on.
    """
    spec = config.FEATURE_SETS[feature_set]
    acyl_cols = select_columns(acyl, spec["include"], with_pka)
    amine_cols = select_columns(amine, spec["include"], with_pka)
    if not acyl_cols or not amine_cols:
        raise ValueError(f"feature set '{feature_set}' selected no columns")

    a_ok = acyl[acyl_cols].notna().all(axis=1)
    m_ok = amine[amine_cols].notna().all(axis=1)
    keep = (reactions.acyl.map(a_ok).fillna(False)
            & reactions.amine.map(m_ok).fillna(False))
    rows = reactions[keep].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"feature set '{feature_set}' covers no reactions")

    X = np.hstack([acyl.loc[rows.acyl, acyl_cols].to_numpy(dtype=float),
                   amine.loc[rows.amine, amine_cols].to_numpy(dtype=float)])
    columns = [f"acyl:{c}" for c in acyl_cols] + [f"amine:{c}" for c in amine_cols]
    return Design(X=X, rows=rows, columns=columns, n_acyl=len(acyl_cols),
                  dropped=int((~keep).sum()))


def restrict(design: Design, reactions: Sequence[str]) -> Design:
    """Cut a design down to a given set of reaction ids, preserving order."""
    keep = design.rows.reaction.isin(set(reactions)).to_numpy()
    return Design(X=design.X[keep], rows=design.rows[keep].reset_index(drop=True),
                  columns=design.columns, n_acyl=design.n_acyl,
                  dropped=design.dropped + int((~keep).sum()))


def common_reactions(designs: Sequence[Design]) -> List[str]:
    """Reactions every design covers, so candidates can be compared like for like.

    Feature sets differ in which reactants they can describe — anything needing
    a Hessian cannot cover a reactant whose Hessian was never run. Scoring each
    candidate on whatever it happens to cover rewards the ones that quietly
    dropped the hardest rows, so the benchmark intersects first.
    """
    shared = set(designs[0].rows.reaction)
    for design in designs[1:]:
        shared &= set(design.rows.reaction)
    return sorted(shared)


def available_sets(acyl: pd.DataFrame, amine: pd.DataFrame,
                   names: Optional[Sequence[str]] = None) -> List[str]:
    """Feature sets that select at least one column on both sides."""
    out = []
    for name in (names or config.BENCHMARK_ORDER):
        spec = config.FEATURE_SETS[name]
        if (select_columns(acyl, spec["include"])
                and select_columns(amine, spec["include"])):
            out.append(name)
    return out
