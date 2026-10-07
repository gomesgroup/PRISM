"""Graph construction and the graph estimator's contract.

The torch-dependent tests skip cleanly where torch is absent, so the suite
still runs in an environment that only has the descriptor models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

rdkit = pytest.importorskip("rdkit")
from prism_ml import graph  # noqa: E402

BENZOYL_CHLORIDE = "ClC(=O)c1ccccc1"
ANILINE = "Nc1ccc(F)cc1"
BOC_PIPERIDINE = "O=C(N1CCC(NC2CC2)CC1)OC(C)(C)C"


class TestCentres:
    def test_acyl_centre_is_the_carbonyl_carbon(self):
        g = graph.build_graph(BENZOYL_CHLORIDE, "acyl")
        mol = rdkit.Chem.MolFromSmiles(BENZOYL_CHLORIDE)
        atom = mol.GetAtomWithIdx(g.centre)
        assert atom.GetSymbol() == "C"
        assert any(n.GetSymbol() == "Cl" for n in atom.GetNeighbors())

    def test_amine_centre_is_a_nitrogen(self):
        g = graph.build_graph(ANILINE, "amine")
        mol = rdkit.Chem.MolFromSmiles(ANILINE)
        assert mol.GetAtomWithIdx(g.centre).GetSymbol() == "N"

    def test_carbamate_nitrogen_is_rejected(self):
        """The Boc nitrogen is adjacent to a carbonyl and is not nucleophilic."""
        g = graph.build_graph(BOC_PIPERIDINE, "amine")
        mol = rdkit.Chem.MolFromSmiles(BOC_PIPERIDINE)
        atom = mol.GetAtomWithIdx(g.centre)
        assert atom.GetSymbol() == "N"
        adjacent_carbonyl = any(
            n.GetSymbol() == "C" and any(o.GetSymbol() == "O" for o in n.GetNeighbors())
            for n in atom.GetNeighbors())
        assert not adjacent_carbonyl, "picked the carbamate nitrogen"

    def test_centre_flag_is_set_on_exactly_one_atom(self):
        g = graph.build_graph(ANILINE, "amine")
        assert g.atoms[:, -1].sum() == pytest.approx(1.0)
        assert g.atoms[g.centre, -1] == pytest.approx(1.0)

    def test_unparseable_smiles_returns_none(self):
        assert graph.build_graph("not a molecule", "amine") is None


class TestGraphShape:
    def test_adjacency_is_symmetric_without_self_loops(self):
        g = graph.build_graph(BENZOYL_CHLORIDE, "acyl")
        np.testing.assert_array_equal(g.adjacency, g.adjacency.T)
        assert np.diag(g.adjacency).sum() == 0

    def test_padding_masks_the_absent_atoms(self):
        small = graph.build_graph("CN", "amine")
        large = graph.build_graph(BOC_PIPERIDINE, "amine")
        batch = graph.pad_batch([small, large])
        assert batch["atoms"].shape[1] == large.n_atoms
        assert batch["mask"][0].sum() == small.n_atoms
        assert batch["mask"][1].sum() == large.n_atoms
        # padded rows must carry no features, or they would pollute the pool
        assert batch["atoms"][0, small.n_atoms:].sum() == 0


class TestEstimator:
    def _setup(self):
        pytest.importorskip("torch")
        from prism_ml.gnn import GraphNetEstimator
        acyl = {1: graph.build_graph(BENZOYL_CHLORIDE, "acyl")}
        amine = {1: graph.build_graph(ANILINE, "amine"),
                 2: graph.build_graph(BOC_PIPERIDINE, "amine")}
        rows = pd.DataFrame({"acyl": [1] * 12, "amine": [1, 2] * 6})
        y = np.array([1, 0] * 6, dtype=float)
        X = np.zeros((12, 3), dtype=np.float32)
        return GraphNetEstimator(acyl, amine, epochs=3, hidden=8, layers=1), X, y, rows

    def test_requires_rows(self):
        est, X, y, _ = self._setup()
        with pytest.raises(ValueError, match="needs the reaction table"):
            est.fit(X, y)

    def test_declares_that_it_needs_rows(self):
        est, _, _, _ = self._setup()
        assert est.needs_rows is True

    def test_probabilities_are_valid(self):
        est, X, y, rows = self._setup()
        p = est.fit(X, y, rows=rows).predict_proba(X, rows=rows)
        assert p.shape == (12, 2)
        assert np.all((p >= 0) & (p <= 1))
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_coverage_reports_missing_graphs(self):
        est, _, _, _ = self._setup()
        rows = pd.DataFrame({"acyl": [1, 99], "amine": [1, 1]})
        np.testing.assert_array_equal(est.covers(rows), [True, False])


class TestAimnetFeatures:
    """The stored AIMNet2 features, if they have been extracted."""

    def _store(self):
        from prism_ml import config, graph as g
        if not config.AIMNET_NPZ.exists():
            pytest.skip("run scripts/extract_aimnet_features.py first")
        return g.load_aimnet_graphs(config.AIMNET_NPZ)

    def test_atom_width_is_513(self):
        """256 learned + 256 atoms-in-molecules + 1 partial charge."""
        store = self._store()
        any_graph = next(iter(store["amine"].values()))
        assert any_graph.n_features == 513

    def test_centre_indexes_the_stored_atoms(self):
        store = self._store()
        for kind in ("acyl", "amine"):
            for rid, mol in store[kind].items():
                assert 0 <= mol.centre < mol.n_atoms, f"{kind} {rid}"

    def test_connectivity_is_a_complete_graph_without_self_loops(self):
        store = self._store()
        mol = next(iter(store["acyl"].values()))
        n = mol.n_atoms
        assert mol.adjacency.sum() == pytest.approx(n * (n - 1))
        assert np.diag(mol.adjacency).sum() == 0

    def test_source_is_recorded(self):
        store = self._store()
        assert next(iter(store["acyl"].values())).source == "aimnet2"


class TestAimnetFeaturiser:
    """Recomputing features from a geometry."""

    def _model(self):
        pytest.importorskip("torch")
        from prism_ml import aimnet, config
        if not config.AIMNET_MODEL.exists():
            pytest.skip("AIMNet2 model not present")
        return aimnet

    def test_unsupported_element_is_rejected(self, tmp_path):
        aimnet = self._model()
        path = tmp_path / "bad.xyz"
        path.write_text("2\n\nXx 0.0 0.0 0.0\nH 0.0 0.0 1.0\n")
        with pytest.raises(ValueError, match="unsupported element"):
            aimnet.featurise(path, aimnet.load_model())

    def test_features_match_the_stored_array(self):
        """The featuriser must reproduce what the stored dataset holds."""
        aimnet = self._model()
        from prism_ml import config
        if not config.AIMNET_NPZ.exists():
            pytest.skip("features not extracted")
        xyz = config.XYZ_DIR / "amines" / "amine_03.xyz"
        if not xyz.exists():
            pytest.skip("geometry not available")
        store = np.load(config.AIMNET_NPZ)
        computed = aimnet.featurise(xyz, aimnet.load_model())
        np.testing.assert_allclose(computed, store["atoms/amine/3"], atol=1e-3)
