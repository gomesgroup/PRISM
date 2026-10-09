"""The reaction-table rules, checked against synthetic workbooks.

These encode decisions recovered by reproducing the study's existing table.
If one changes, a test here should fail rather than a number quietly moving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import data  # noqa: E402


def write_workbooks(tmp_path: Path, rates: list, excluded: list):
    rates_path, excluded_path = tmp_path / "r.xlsx", tmp_path / "e.xlsx"
    pd.DataFrame(rates or [{"acyl_label": 99, "amine_label": 99,
                            "k_corrected": 1.0}]).to_excel(rates_path, index=False)
    pd.DataFrame(excluded or [{"acyl_label": 98, "amine_label": 98,
                               "rate_class": "slow"}]).to_excel(excluded_path,
                                                                index=False)
    return rates_path, excluded_path


def test_replicate_rate_constants_are_geometrically_averaged(tmp_path):
    r, e = write_workbooks(tmp_path, [
        {"acyl_label": 1, "amine_label": 1, "k_corrected": 10.0},
        {"acyl_label": 1, "amine_label": 1, "k_corrected": 1000.0},
        {"acyl_label": 1, "amine_label": 1, "k_corrected": -3.0}], [])
    table = data.load_reactions(r, e).table
    row = table[(table.acyl == 1) & (table.amine == 1)].iloc[0]
    assert row.k == pytest.approx(100.0)          # sqrt(10 * 1000); k <= 0 ignored


def test_replicate_batches_are_assigned_to_their_reactant(tmp_path):
    """4-F benzoyl chloride batch 2B, recorded under acyl 2, belongs to acyl 1."""
    r, e = write_workbooks(tmp_path, [
        {"acyl_label": 1, "amine_label": 7, "acyl_chloride": "4-fluorobenzoyl chloride-1A",
         "k_corrected": 20.0},
        {"acyl_label": 2, "amine_label": 7, "acyl_chloride": "4-fluorobenzoyl chloride-2B",
         "k_corrected": 80.0},
        {"acyl_label": 2, "amine_label": 7, "acyl_chloride": "benzoyl chloride",
         "k_corrected": 5.0}], [])
    table = data.load_reactions(r, e).table.set_index(["acyl", "amine"])
    assert table.loc[(1, 7), "k"] == pytest.approx(40.0)
    assert table.loc[(2, 7), "k"] == pytest.approx(5.0)


def test_disagreeing_replicate_classes_need_a_rate_constant(tmp_path):
    rows = [{"acyl_label": 1, "amine_label": 1, "rate_class": "slow"},
            {"acyl_label": 1, "amine_label": 1, "rate_class": "instant"}]
    r, e = write_workbooks(tmp_path, [], rows)
    with pytest.raises(ValueError, match="disagree"):
        data.load_reactions(r, e)
    r, e = write_workbooks(tmp_path, [{"acyl_label": 1, "amine_label": 1,
                                       "k_corrected": 10.0}], rows)
    table = data.load_reactions(r, e).table
    assert table.iloc[0].rate_class == "measurable"


def test_a_fitted_rate_beats_an_excluded_classification(tmp_path):
    r, e = write_workbooks(tmp_path,
                           [{"acyl_label": 1, "amine_label": 1, "k_corrected": 10.0}],
                           [{"acyl_label": 1, "amine_label": 1, "rate_class": "slow"}])
    table = data.load_reactions(r, e).table
    assert table[(table.acyl == 1) & (table.amine == 1)].iloc[0].rate_class == "measurable"


def test_nonpositive_rate_without_an_excluded_row_is_instant(tmp_path):
    """A runaway fit means too fast to measure, not that nothing happened."""
    r, e = write_workbooks(tmp_path,
                           [{"acyl_label": 1, "amine_label": 1, "k_corrected": -5.0}], [])
    table = data.load_reactions(r, e).table
    assert table[(table.acyl == 1) & (table.amine == 1)].iloc[0].rate_class == "instant"


def test_ambiguous_can_be_dropped(tmp_path):
    r, e = write_workbooks(tmp_path, [],
                           [{"acyl_label": 5, "amine_label": 5, "rate_class": "ambiguous"}])
    kept = data.load_reactions(r, e, ambiguous="instant").table
    dropped = data.load_reactions(r, e, ambiguous="drop").table
    assert (kept.acyl == 5).sum() == 1
    assert (dropped.acyl == 5).sum() == 0


def test_reactive_target_groups_instant_with_measurable(tmp_path):
    r, e = write_workbooks(tmp_path,
                           [{"acyl_label": 1, "amine_label": 1, "k_corrected": 10.0}],
                           [{"acyl_label": 2, "amine_label": 2, "rate_class": "instant"},
                            {"acyl_label": 3, "amine_label": 3, "rate_class": "slow"}])
    table = data.load_reactions(r, e).table.set_index(["acyl", "amine"])
    assert table.loc[(1, 1), "reactive"] == 1
    assert table.loc[(2, 2), "reactive"] == 1
    assert table.loc[(3, 3), "reactive"] == 0


def test_expected_rows_guards_against_silent_input_changes(tmp_path):
    r, e = write_workbooks(tmp_path,
                           [{"acyl_label": 1, "amine_label": 1, "k_corrected": 10.0}], [])
    with pytest.raises(ValueError, match="expected 99"):
        data.load_reactions(r, e, expected_rows=99)
