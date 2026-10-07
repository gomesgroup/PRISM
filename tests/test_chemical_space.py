"""Chemical-space rules: fingerprints, functional classes and split metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("rdkit")
from prism_ml import chemical_space as cs  # noqa: E402


@pytest.mark.parametrize("smiles, expected", [
    ("Nc1ccccc1", "aromatic primary"),
    ("CNc1ccccc1", "aromatic secondary"),
    ("NCc1ccccc1", "aliphatic primary"),
    ("C1CCNCC1", "aliphatic secondary"),
    ("c1cc[nH]c1", "heteroaromatic"),
])
def test_amine_classes(smiles, expected):
    assert cs.classify_amine(smiles) == expected


@pytest.mark.parametrize("smiles, expected", [
    ("O=C(Cl)c1ccccc1", "aromatic"),
    ("O=C(Cl)c1ccncc1", "heteroaromatic"),
    ("O=C(Cl)c1ccco1", "heteroaromatic"),
    ("C=CC(=O)Cl", "alkenyl"),
    ("CC(C)(C)C(=O)Cl", "aliphatic tertiary"),
    ("CC(C)C(=O)Cl", "aliphatic secondary"),
    ("CC(=O)Cl", "aliphatic primary"),
])
def test_acyl_classes(smiles, expected):
    assert cs.classify_acyl_chloride(smiles) == expected


def test_similarity_is_symmetric_with_unit_diagonal():
    fps = cs.fingerprints(["Nc1ccccc1", "NCc1ccccc1", "C1CCNCC1"])
    s = cs.similarity(fps)
    np.testing.assert_allclose(s, s.T)
    np.testing.assert_allclose(np.diag(s), 1.0)


def test_nearest_training_reactant():
    table = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "smiles": ["Nc1ccccc1", "Nc1ccc(F)cc1", "C1CCNCC1", "Nc1ccc(Cl)cc1"],
        "split": ["train", "train", "train", "test"]})
    result = cs.analyse(table, "amine")
    t = result["table"].set_index("id")
    # the chloroaniline's nearest training reactant is an aniline, not piperidine
    assert t.loc[4, "nearest_train_id"] in (1, 2)
    assert result["summary"]["n_test"] == 1
    assert 0 < t.loc[4, "nearest_train_similarity"] < 1


def test_unparseable_smiles_is_rejected():
    with pytest.raises(ValueError, match="unparseable"):
        cs.fingerprints(["not a smiles"])


def test_shared_acyl_group_is_left_out():
    """Without the C(=O)Cl bits, acyl chlorides compare by their R groups."""
    smiles = ["O=C(Cl)c1ccccc1", "CCCC(=O)Cl"]
    kept = cs.similarity(cs.fingerprints(smiles))[0, 1]
    left_out = cs.similarity(cs.fingerprints(smiles, cs.COMMON_GROUP["acyl"]))[0, 1]
    assert left_out < kept
    # some bits remain (the ring), fewer than with the group
    bits = cs.fingerprints(["O=C(Cl)c1ccccc1"], cs.COMMON_GROUP["acyl"])[0]
    assert 0 < bits.GetNumOnBits() < cs.fingerprints(["O=C(Cl)c1ccccc1"])[0].GetNumOnBits()
