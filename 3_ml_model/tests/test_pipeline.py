"""Correctness checks that run in seconds and need no cluster."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config, features, metrics, splits  # noqa: E402
from prism_ml.model import ReactivityRateModel  # noqa: E402


def toy_reactants(n: int, n_cols: int = 4, prefix: str = "BV_") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(n, n_cols)),
                        columns=[f"{prefix}{i}" for i in range(n_cols)],
                        index=pd.Index(range(1, n + 1), name="id"))


def toy_reactions(n_acyl: int = 6, n_amine: int = 6) -> pd.DataFrame:
    rows = []
    for a in range(1, n_acyl + 1):
        for m in range(1, n_amine + 1):
            reactive = int((a + m) % 3 != 0)
            rows.append(dict(acyl=a, amine=m, reaction=f"rxn_{a:02d}_{m:02d}",
                             rate_class="measurable" if reactive else "slow",
                             logk=float(a - m) if reactive else float("nan"),
                             reactive=reactive))
    return pd.DataFrame(rows)


class TestMetrics:
    def test_r2_needs_spread(self):
        assert np.isnan(metrics.r2([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]))

    def test_r2_perfect(self):
        assert metrics.r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_confusion_counts(self):
        c = metrics.confusion([1, 1, 0, 0], [1, 0, 1, 0])
        assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)

    def test_joint_floors_negative_r2(self):
        """A worse-than-mean fit must not be rewarded for being less negative."""
        assert metrics.joint_score(0.8, -5.0) == metrics.joint_score(0.8, -0.1)


class TestSplitRule:
    def test_small_classes_contribute_no_representative(self):
        assert splits._picks_for_class(config.SMALL_CLASS) == 0

    def test_large_classes_contribute_two(self):
        assert splits._picks_for_class(config.LARGE_CLASS + 1) == 2

    def test_middle_classes_contribute_one(self):
        assert splits._picks_for_class(config.LARGE_CLASS) == 1

    def test_forced_reactant_never_held_out(self):
        reactions = toy_reactions()
        classes = {i: "aromatic" for i in range(1, 7)}
        split = splits.build_split(reactions, classes, classes,
                                   force_train_amines=[3])
        assert 3 not in split.test_amines

    def test_forcing_still_counts_towards_class_population(self):
        """A pinned reactant keeps its class testable via its siblings."""
        reactions = toy_reactions()
        classes = {i: "aromatic" for i in range(1, 7)}
        pinned = splits.build_split(reactions, classes, classes,
                                    force_train_amines=[3])
        assert pinned.test_amines, "class should still contribute a representative"


class TestFolds:
    def test_every_reactant_is_held_out_once(self):
        reactions = toy_reactions()
        folds = splits.loro_folds(reactions, min_rows=1, pair_folds=0)
        acyl_folds = [f for f in folds if f[0] == "acyl"]
        assert len(acyl_folds) == reactions.acyl.nunique()

    def test_held_rows_are_absent_from_training(self):
        reactions = toy_reactions()
        for _, train_idx, held_idx in splits.loro_folds(reactions, min_rows=1,
                                                        pair_folds=5):
            assert not set(train_idx) & set(held_idx)

    def test_pair_folds_are_seed_dependent_and_the_rest_are_not(self):
        reactions = toy_reactions()
        a = splits.loro_folds(reactions, min_rows=1, pair_folds=0, seed=0)
        b = splits.loro_folds(reactions, min_rows=1, pair_folds=0, seed=99)
        assert [f[0] for f in a] == [f[0] for f in b]
        assert all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))


class TestFeatures:
    def test_provenance_columns_are_not_features(self):
        df = toy_reactants(3)
        df["idx_main_atom"] = 1
        df["el_main_atom"] = "C"
        cols = features.select_columns(df, r"^.")
        assert not any(c.startswith(("idx_", "el_")) for c in cols)

    def test_incomplete_reactants_drop_their_reactions(self):
        reactions = toy_reactions(3, 3)
        acyl, amine = toy_reactants(3), toy_reactants(3)
        amine.loc[2, "BV_0"] = np.nan
        design = features.build_design(reactions, acyl, amine, "tier1")
        assert design.dropped == (reactions.amine == 2).sum()
        assert 2 not in set(design.rows.amine)

    def test_acyl_block_precedes_amine_block(self):
        reactions = toy_reactions(3, 3)
        design = features.build_design(reactions, toy_reactants(3),
                                       toy_reactants(3), "tier1")
        assert all(c.startswith("acyl:") for c in design.columns[:design.n_acyl])
        assert all(c.startswith("amine:") for c in design.columns[design.n_acyl:])


class TestModel:
    def _fit(self):
        reactions = toy_reactions(8, 8)
        design = features.build_design(reactions, toy_reactants(8),
                                       toy_reactants(8), "tier1")
        model = ReactivityRateModel().fit(
            design.X, design.rows.reactive, design.rows.logk)
        return model, design

    def test_gate_suppresses_rate_constants_for_unreactive_calls(self):
        model, design = self._fit()
        pred = model.predict(design.X)
        gated = pred.gated_log_k()
        assert np.isnan(gated[~pred.emitted]).all()
        assert np.isfinite(gated[pred.emitted]).all()

    def test_rate_head_ignores_rows_without_a_measured_rate_constant(self):
        model, design = self._fit()
        assert model.n_rate_rows == int(np.isfinite(design.rows.logk).sum())
        assert model.n_rate_rows < len(design.rows)

    def test_refitting_is_deterministic(self):
        model_a, design = self._fit()
        model_b = ReactivityRateModel().fit(
            design.X, design.rows.reactive, design.rows.logk)
        np.testing.assert_array_equal(model_a.predict(design.X).probability,
                                      model_b.predict(design.X).probability)

    def test_family_is_read_at_call_time_not_import_time(self):
        """A module-level default would freeze the family at import."""
        from prism_ml import config as cfg
        original = cfg.SELECTED_FAMILY
        try:
            cfg.SELECTED_FAMILY = "random_forest"
            assert ReactivityRateModel().family == "random_forest"
        finally:
            cfg.SELECTED_FAMILY = original
        assert ReactivityRateModel().family == original

    def test_unknown_family_is_rejected(self):
        with pytest.raises(ValueError, match="unknown model family"):
            ReactivityRateModel(family="not_a_model")

    def test_families_produce_different_predictions(self):
        """Guards against a family argument that is silently ignored."""
        _, design = self._fit()
        a = ReactivityRateModel(family="boosted_trees").fit(
            design.X, design.rows.reactive, design.rows.logk).predict(design.X)
        b = ReactivityRateModel(family="uninformative").fit(
            design.X, design.rows.reactive, design.rows.logk).predict(design.X)
        assert not np.allclose(a.log_k, b.log_k)

    def test_too_few_rate_constants_is_an_error_not_a_silent_fit(self):
        reactions = toy_reactions(3, 3)
        design = features.build_design(reactions, toy_reactants(3),
                                       toy_reactants(3), "tier1")
        logk = pd.Series([np.nan] * len(design.rows))
        with pytest.raises(ValueError, match="need >= 10"):
            ReactivityRateModel().fit(design.X, design.rows.reactive, logk)


def test_fixed_test_set_overrides_the_rule(monkeypatch):
    import pandas as pd
    from prism_ml import config, holdout
    reactions = pd.DataFrame({"acyl": [1, 2, 3, 4], "amine": [5, 6, 7, 8],
                              "rate_class": ["measurable"] * 4})
    monkeypatch.setattr(config, "FIXED_TEST_SET", {"acyl": (3, 1), "amine": (8,)})
    split = holdout.build_split(reactions, None, None)
    assert split.test_acyls == (1, 3) and split.test_amines == (8,)
    monkeypatch.setattr(config, "FIXED_TEST_SET", {"acyl": (9,), "amine": (8,)})
    with pytest.raises(ValueError, match="no reactions"):
        holdout.build_split(reactions, None, None)


def test_stratified_folds_follow_the_class_rule():
    import pandas as pd
    acyls, amines = list(range(1, 15)), list(range(101, 115))
    rows = pd.DataFrame([(a, m) for a in acyls for m in amines], columns=["acyl", "amine"])
    acyl_class = {a: ("big" if a <= 11 else "small") for a in acyls}     # 11 + 3 members
    amine_class = {m: ("mid" if m <= 105 else "two") for m in amines}    # 5 members, then 9
    folds = splits.stratified_folds(rows, acyl_class, amine_class, draws=5, first_seed=1,
                                    force_train_amines=[101])
    assert len(folds) == 5 and all(kind == "stratified" for kind, _, _ in folds)
    for _, train, held in folds:
        assert len(set(train) & set(held)) == 0 and len(train) + len(held) == len(rows)
        h = rows.iloc[held]
        held_acyls = set(h.acyl) - set(rows.iloc[train].acyl)
        held_amines = set(h.amine) - set(rows.iloc[train].amine)
        assert sum(acyl_class[a] == "big" for a in held_acyls) == 2      # > 10 members
        assert sum(acyl_class[a] == "small" for a in held_acyls) == 1    # 3-10 members
        assert 101 not in held_amines                                    # forced into training
    again = splits.stratified_folds(rows, acyl_class, amine_class, draws=5, first_seed=1,
                                    force_train_amines=[101])
    assert all((a[2] == b[2]).all() for a, b in zip(folds, again))       # seeded
