"""Molecular graphs built from the SMILES in the feature workbook.

The descriptor models see a reactant as a fixed vector of physical quantities.
A graph model sees it as atoms and bonds, and has to learn what matters from
the reaction data alone. Building both from the same workbook keeps the
comparison honest about where each one's information comes from.

Reaction centres are located by rule rather than learned, matching how the
descriptors are computed:

    acyl chloride   the carbonyl carbon — the one bonded to the chlorine
    amine           the nucleophilic nitrogen — fewer than three heavy
                    neighbours, never adjacent to a carbonyl (which would make
                    it an amide or carbamate), preferring the most hydrogens

Marking the centre matters: pooling over every atom averages the two atoms the
mechanism turns on in with thirty others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Elements seen in this dataset; anything else falls into the last slot.
ELEMENTS = ("C", "N", "O", "F", "S", "Cl", "Br", "I", "P", "B", "Si")
#: Atom feature width: element one-hot + degree, charge, aromatic, in-ring,
#: hydrogen count, and the reaction-centre flag.
ATOM_FEATURES = len(ELEMENTS) + 1 + 6


@dataclass(frozen=True)
class MolGraph:
    """One molecule: atom features, connectivity, and its reacting atom.

    ``adjacency`` is bond connectivity for graphs built from SMILES. For graphs
    carrying AIMNet2 atom features it is a complete graph: those features
    already encode each atom's local environment, and the geometry they came
    from is not reducible to a bond table here, so every atom is allowed to
    attend to every other and the model learns the weighting.
    """

    atoms: np.ndarray          #: (n_atoms, n_features)
    adjacency: np.ndarray      #: (n_atoms, n_atoms), symmetric, no self-loops
    centre: int                #: index of the reacting atom
    source: str = "smiles"     #: "smiles" or "aimnet2"

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.atoms.shape[1])


def _atom_vector(atom, is_centre: bool) -> np.ndarray:
    v = np.zeros(ATOM_FEATURES, dtype=np.float32)
    symbol = atom.GetSymbol()
    v[ELEMENTS.index(symbol) if symbol in ELEMENTS else len(ELEMENTS)] = 1.0
    off = len(ELEMENTS) + 1
    v[off + 0] = atom.GetDegree() / 4.0
    v[off + 1] = atom.GetFormalCharge()
    v[off + 2] = float(atom.GetIsAromatic())
    v[off + 3] = float(atom.IsInRing())
    v[off + 4] = atom.GetTotalNumHs() / 4.0
    v[off + 5] = float(is_centre)
    return v


def _acyl_centre(mol) -> Optional[int]:
    """The carbonyl carbon: the carbon bearing the chlorine."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "Cl":
            continue
        for neighbour in atom.GetNeighbors():
            if neighbour.GetSymbol() == "C":
                return neighbour.GetIdx()
    return None


def _amine_centre(mol) -> Optional[int]:
    """The nucleophilic nitrogen, never an amide or carbamate nitrogen."""
    best, best_key = None, None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]
        amide_like = any(
            n.GetSymbol() == "C" and any(o.GetSymbol() == "O" for o in n.GetNeighbors())
            for n in heavy)
        # Sort key: prefer not amide-like, then most hydrogens, then fewest
        # heavy neighbours — the same preference the descriptor pipeline uses.
        key = (amide_like, -atom.GetTotalNumHs(), len(heavy), atom.GetIdx())
        if best_key is None or key < best_key:
            best, best_key = atom.GetIdx(), key
    return best


def build_graph(smiles: str, kind: str) -> Optional[MolGraph]:
    """Graph for one reactant, or None if the SMILES or centre is unusable."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    centre = _acyl_centre(mol) if kind == "acyl" else _amine_centre(mol)
    if centre is None:
        return None

    n = mol.GetNumAtoms()
    atoms = np.stack([_atom_vector(a, a.GetIdx() == centre) for a in mol.GetAtoms()])
    adjacency = np.zeros((n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i, j] = adjacency[j, i] = 1.0
    return MolGraph(atoms=atoms, adjacency=adjacency, centre=centre)


def build_graphs(table: pd.DataFrame, kind: str) -> Dict[int, MolGraph]:
    """Graphs for every reactant in a feature table that has a usable SMILES."""
    if "smiles" not in table.columns:
        raise ValueError("feature table has no 'smiles' column")
    graphs: Dict[int, MolGraph] = {}
    for rid, smiles in zip(table.index, table["smiles"]):
        if not isinstance(smiles, str) or not smiles:
            continue
        graph = build_graph(smiles, kind)
        if graph is not None:
            graphs[int(rid)] = graph
    return graphs


def load_aimnet_graphs(path) -> Dict[str, Dict[int, MolGraph]]:
    """Per-reactant AIMNet2 atom features, from `extract_aimnet_features.py`.

    Returns ``{"acyl": {id: MolGraph}, "amine": {id: MolGraph}}``. Each atom
    carries 513 values — 256 learned atomic features, 256 AIM features and the
    partial charge — and the stored centre index addresses them directly,
    because both came from the same geometry.

    Connectivity is a complete graph. These features are the output of a
    message-passing potential and already describe each atom in the context of
    its neighbours, so imposing a bond table on top would be both redundant and
    unreliable: the atom order is the geometry's, and a SMILES traversal does
    not reproduce it.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/extract_aimnet_features.py first.")
    store = np.load(path)
    out: Dict[str, Dict[int, MolGraph]] = {"acyl": {}, "amine": {}}
    for key in store.files:
        if not key.startswith("atoms/"):
            continue
        _, kind, rid = key.split("/")
        atoms = store[key].astype(np.float32)
        n = atoms.shape[0]
        adjacency = (np.ones((n, n), dtype=np.float32)
                     - np.eye(n, dtype=np.float32))
        out[kind][int(rid)] = MolGraph(
            atoms=atoms, adjacency=adjacency,
            centre=int(store[f"centre/{kind}/{rid}"]), source="aimnet2")
    return out


def pad_batch(graphs: Sequence[MolGraph]) -> Dict[str, np.ndarray]:
    """Pad a list of graphs to a common size, with a mask over the real atoms.

    Padding keeps the batch a fixed-shape tensor; the mask stops padded atoms
    contributing to message passing or to the pooled representation.
    """
    if not graphs:
        raise ValueError("empty batch")
    n_max = max(g.n_atoms for g in graphs)
    width = graphs[0].n_features
    if any(g.n_features != width for g in graphs):
        raise ValueError("graphs in a batch must have the same feature width")
    b = len(graphs)
    atoms = np.zeros((b, n_max, width), dtype=np.float32)
    adjacency = np.zeros((b, n_max, n_max), dtype=np.float32)
    mask = np.zeros((b, n_max), dtype=np.float32)
    centre = np.zeros(b, dtype=np.int64)
    for i, g in enumerate(graphs):
        n = g.n_atoms
        atoms[i, :n] = g.atoms
        adjacency[i, :n, :n] = g.adjacency
        mask[i, :n] = 1.0
        centre[i] = g.centre
    return {"atoms": atoms, "adjacency": adjacency, "mask": mask, "centre": centre}
