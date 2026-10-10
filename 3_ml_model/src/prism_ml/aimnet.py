"""Atom features from AIMNet2, computed from a geometry.

AIMNet2 is a neural network potential. Run over a molecule it returns, per
atom, a learned atomic feature vector, an atoms-in-molecules feature vector and
a partial charge — a description of each atom in the context of the whole
molecule, which is what the graph model consumes instead of the hand-chosen
descriptors the other models use.

Everything needed is local: the TorchScript model in `data/aimnet2/` (the
AIMNet2 checkpoint distributed with LoQI; see the README there) and the
geometries the caller supplies. There is no server and no network call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config

#: Element symbol to atomic number, for the elements in this dataset.
ATOMIC_NUMBER = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14,
                 "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}

#: Width of the concatenated atom vector: 256 learned + 256 AIM + 1 charge.
FEATURE_WIDTH = 513


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Elements and coordinates, in file order — which the features index."""
    lines = Path(path).read_text().splitlines()
    count = int(lines[0].split()[0])
    elements, coordinates = [], []
    for line in lines[2:2 + count]:
        fields = line.split()
        if len(fields) >= 4:
            elements.append(fields[0])
            coordinates.append([float(v) for v in fields[1:4]])
    return elements, np.asarray(coordinates, dtype=np.float64)


def load_model(path: Optional[Path] = None):
    """Load the TorchScript AIMNet2 model onto the CPU."""
    import torch
    path = Path(path or config.AIMNET_MODEL)
    if not path.exists():
        raise FileNotFoundError(
            f"AIMNet2 model not found at {path}; see data/aimnet2/README.md.")
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def featurise(path: Path, model, charge: int = 0, multiplicity: int = 1
              ) -> np.ndarray:
    """Per-atom features for one geometry, as ``(n_atoms, 513)``.

    Columns are the learned atomic features, then the AIM features, then the
    partial charge — the order the stored dataset uses, so features computed
    here and features read from it are interchangeable.
    """
    import torch

    elements, coordinates = read_xyz(path)
    unknown = sorted({e for e in elements if e not in ATOMIC_NUMBER})
    if unknown:
        raise ValueError(f"{Path(path).name}: unsupported element(s) {unknown}")
    numbers = np.array([ATOMIC_NUMBER[e] for e in elements])

    inputs = {"coord": torch.tensor(coordinates).reshape(1, -1, 3),
              "numbers": torch.tensor(numbers).reshape(1, -1),
              "charge": torch.full((1,), float(charge)),
              "mult": torch.full((1,), float(multiplicity))}
    with torch.no_grad():
        output = model(inputs)

    n = len(elements)
    a = output["a"].detach().numpy().reshape(n, -1)
    aim = output["aim"].detach().numpy().reshape(n, -1)
    # The stored charge is the total: point charge plus the spin contribution
    # where the checkpoint predicts one. The LoQI checkpoint has no spin head;
    # for these closed-shell molecules that contribution is zero anyway.
    q = output["charges"]
    if "spin_charges" in output:
        q = q + output["spin_charges"]
    q = q.detach().numpy().reshape(n, 1)
    return np.concatenate([a, aim, q], axis=1).astype(np.float32)


def featurise_directory(xyz_dir: Path, pattern: str, model,
                        ids: Optional[Sequence[int]] = None
                        ) -> Dict[int, np.ndarray]:
    """Featurise every matching geometry, keyed by the id in its filename."""
    import re
    out: Dict[int, np.ndarray] = {}
    for path in sorted(Path(xyz_dir).glob(pattern)):
        match = re.search(r"(\d+)", path.stem)
        if not match:
            continue
        rid = int(match.group(1))
        if ids is not None and rid not in ids:
            continue
        out[rid] = featurise(path, model)
    return out
