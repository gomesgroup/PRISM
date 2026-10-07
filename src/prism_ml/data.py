"""Loading the Excel sources and assembling the reaction table.

Both rate workbooks carry one row per plate, so a reactant pair can appear more
than once, and ten pairs appear in both. The rules below are not arbitrary:
they were recovered by reproducing the study's existing reaction table from
these two workbooks, and :func:`load_reactions` re-checks them on every call
through ``expected_rows``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

#: `rate_class` values in the excluded workbook that map onto the three-way
#: scheme unchanged. `ambiguous` does not, and is handled explicitly.
KNOWN_CLASSES = ("slow", "instant")


@dataclass(frozen=True)
class Reactions:
    """One row per acyl/amine pair."""

    table: pd.DataFrame

    def __len__(self) -> int:
        return len(self.table)

    @property
    def measurable(self) -> pd.DataFrame:
        return self.table[self.table.rate_class == "measurable"]

    def summary(self) -> str:
        counts = self.table.rate_class.value_counts().to_dict()
        return (f"{len(self.table)} pairs "
                f"({self.table.acyl.nunique()} acyl chlorides x "
                f"{self.table.amine.nunique()} amines); {counts}")


def _pairs(path: Path) -> pd.DataFrame:
    """Read a rate workbook and collapse replicate plate rows."""
    df = pd.read_excel(path)
    df = df.assign(
        acyl=pd.to_numeric(df.acyl_label, errors="coerce"),
        amine=pd.to_numeric(df.amine_label, errors="coerce"),
    ).dropna(subset=["acyl", "amine"])
    df["acyl"] = df.acyl.astype(int)
    df["amine"] = df.amine.astype(int)
    # First row per pair, not the mean: matches the study's table on every one
    # of its duplicated pairs, where averaging does not.
    return df.drop_duplicates(subset=["acyl", "amine"], keep="first")


def load_reactions(rates_xlsx: Path = None, excluded_xlsx: Path = None,
                   ambiguous: str = "instant",
                   expected_rows: Optional[int] = None) -> Reactions:
    """Assemble the reaction table from the two rate workbooks.

    Args:
        ambiguous: how to treat the `ambiguous` class, which the three-way
            scheme has no slot for. ``"instant"`` reproduces the study's table;
            ``"drop"`` removes those pairs.
        expected_rows: if given, raise unless exactly this many pairs result.
            Use it to pin the table against silent upstream changes.
    """
    rates = _pairs(Path(rates_xlsx or config.RATES_XLSX)).set_index(["acyl", "amine"])
    excl = _pairs(Path(excluded_xlsx or config.EXCLUDED_XLSX)).set_index(["acyl", "amine"])

    records = []
    for key in sorted(set(rates.index) | set(excl.index)):
        k = logk = None
        rate_class = None
        nonpositive = False
        if key in rates.index:
            k = float(rates.loc[key, "k_corrected"])
            if k > 0:
                # A fitted rate wins over an excluded classification: every
                # pair appearing in both workbooks is measurable.
                rate_class, logk = "measurable", float(np.log10(k))
            else:
                nonpositive, k = True, None
        if rate_class is None:
            if key in excl.index:
                rate_class = str(excl.loc[key, "rate_class"])
                if rate_class == "ambiguous":
                    if ambiguous == "drop":
                        continue
                    rate_class = ambiguous
            elif nonpositive:
                # A non-positive fitted rate with no excluded row means the fit
                # ran away, not that nothing happened.
                rate_class = "instant"
            else:
                continue
        records.append(dict(acyl=key[0], amine=key[1],
                            reaction=f"rxn_{key[0]:02d}_{key[1]:02d}",
                            rate_class=rate_class, k=k, logk=logk))

    table = pd.DataFrame.from_records(records)
    unexpected = set(table.rate_class) - set(KNOWN_CLASSES) - {"measurable"}
    if unexpected:
        raise ValueError(f"unhandled rate_class values: {sorted(unexpected)}")
    # The classification target. `reactive` covers a measurable rate constant
    # and one too fast to resolve. `un-reactive` (recorded upstream as `slow`)
    # means the acyl chloride was consumed by a competing pathway faster than
    # it formed the amide, so no amide product appears — hydrolysis being the
    # ever-present competitor the assay measures against, and any other
    # favoured side reaction giving the same outcome.
    table["reactive"] = (table.rate_class != "slow").astype(int)
    if expected_rows is not None and len(table) != expected_rows:
        raise ValueError(f"expected {expected_rows} pairs, assembled {len(table)}")
    return Reactions(table.sort_values(["acyl", "amine"]).reset_index(drop=True))


def load_features(features_xlsx: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-reactant descriptors, indexed by reactant id.

    Returns the acyl and amine tables. Missing cells are left as NaN here; a
    feature set decides what to do with them, because "this atom does not
    exist" and "this quantity was not computed" both present as a blank and
    only the caller knows which matters.
    """
    path = Path(features_xlsx or config.FEATURES_XLSX)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It ships in data/datasheets; for unseen substrates use scripts/predict.py, which computes descriptors itself.")
    out = []
    for sheet in ("acyl", "amine"):
        df = pd.read_excel(path, sheet_name=sheet)
        if "id" not in df.columns:
            raise ValueError(f"sheet '{sheet}' has no 'id' column")
        out.append(df.set_index("id").sort_index())
    return out[0], out[1]


def describe_coverage(acyl: pd.DataFrame, amine: pd.DataFrame,
                      pattern: str) -> Dict[str, str]:
    """How many reactants have every column matching ``pattern``."""
    report = {}
    for name, df in (("acyl", acyl), ("amine", amine)):
        cols = [c for c in df.columns if pd.Series([c]).str.match(pattern).iloc[0]]
        if not cols:
            report[name] = "no matching columns"
            continue
        complete = df[cols].notna().all(axis=1)
        missing = sorted(df.index[~complete].tolist())
        report[name] = (f"{int(complete.sum())}/{len(df)} complete"
                        + (f"; missing {missing}" if missing else ""))
    return report
