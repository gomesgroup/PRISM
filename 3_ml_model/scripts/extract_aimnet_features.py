#!/usr/bin/env python3
"""Compute AIMNet2 atom features for every reactant geometry.

Runs the TorchScript AIMNet2 model in `data/aimnet2/` over the XYZ files and
writes `data/aimnet2_atoms.npz`. Each atom gets 513 values: 256 learned atomic
features, 256 atoms-in-molecules features, and the partial charge.

The reaction centre is located from the same geometry, so its index addresses
the stored array directly. Atom order is the geometry's; a SMILES traversal
does not reproduce it, which is why the centre is found here rather than
looked up later.

The checkpoint is the AIMNet2 model distributed with LoQI; see
data/aimnet2/README.md for where to get it.

Run once; afterwards the package needs only the .npz it writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prism_ml import config  # noqa: E402

#: Cordero covalent radii in angstrom, and the tolerance used to perceive bonds.
COVALENT = {"H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
            "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39}
TOLERANCE = 1.3


def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    elements, coords = [], []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) >= 4:
            elements.append(parts[0])
            coords.append([float(v) for v in parts[1:4]])
    return elements, np.asarray(coords)


def bonds(elements: Sequence[str], coords: np.ndarray) -> np.ndarray:
    radii = np.array([COVALENT.get(e, 0.8) for e in elements])
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt((delta ** 2).sum(-1))
    cutoff = TOLERANCE * (radii[:, None] + radii[None, :])
    return (distance < cutoff) & (distance > 1e-6)


def acyl_centre(elements, adjacency) -> Optional[int]:
    """The carbonyl carbon: the carbon bearing the chlorine."""
    for i, element in enumerate(elements):
        if element != "Cl":
            continue
        for j in np.where(adjacency[i])[0]:
            if elements[j] == "C":
                return int(j)
    return None


def amine_centre(elements, adjacency) -> Optional[int]:
    """The nucleophilic nitrogen, never an amide or carbamate nitrogen."""
    best, best_key = None, None
    for i, element in enumerate(elements):
        if element != "N":
            continue
        attached = np.where(adjacency[i])[0]
        hydrogens = sum(1 for j in attached if elements[j] == "H")
        amide_like = any(
            elements[j] == "C" and any(elements[k] == "O" for k in np.where(adjacency[j])[0])
            for j in attached)
        key = (amide_like, -hydrogens, len(attached), i)
        if best_key is None or key < best_key:
            best, best_key = int(i), key
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xyz-dir", type=Path,
                   default=config.XYZ_DIR)
    p.add_argument("--model", type=Path, default=None,
                   help="TorchScript AIMNet2 model (default: the LoQI checkpoint in data/aimnet2/)")
    p.add_argument("--output", type=Path, default=REPO / "data" / "aimnet2_atoms.npz")
    args = p.parse_args()

    from prism_ml import aimnet

    model = aimnet.load_model(args.model)
    atoms: Dict[str, np.ndarray] = {}
    for kind, sub, pattern in (("acyl", "acyls", "acid_*.xyz"),
                               ("amine", "amines", "amine_*.xyz")):
        directory = args.xyz_dir / sub
        if not directory.is_dir():
            print(f"no {directory}", file=sys.stderr)
            return 1
        found = aimnet.featurise_directory(directory, pattern, model)
        for rid, block in found.items():
            atoms[f"{kind}/{rid}"] = block
        print(f"{kind:6s} featurised {len(found)} geometries from {directory.name}/")

    centres: Dict[str, np.ndarray] = {}
    mismatched: List[str] = []
    for name, block in atoms.items():
        kind, rid = name.split("/")
        stem = f"acid_{int(rid):02d}.xyz" if kind == "acyl" else f"amine_{int(rid):02d}.xyz"
        path = args.xyz_dir / ("acyls" if kind == "acyl" else "amines") / stem
        if not path.exists():
            mismatched.append(f"{name}: no {path.name}")
            continue
        elements, coords = read_xyz(path)
        if len(elements) != block.shape[0]:
            # A geometry with a different atom count cannot index these
            # features; better to drop the reactant than to mis-address it.
            mismatched.append(f"{name}: xyz has {len(elements)} atoms, "
                              f"features have {block.shape[0]}")
            continue
        adjacency = bonds(elements, coords)
        centre = acyl_centre(elements, adjacency) if kind == "acyl" \
            else amine_centre(elements, adjacency)
        if centre is None:
            mismatched.append(f"{name}: no reaction centre found")
            continue
        centres[name] = np.array(centre, dtype=np.int64)

    usable = {k: v for k, v in atoms.items() if k in centres}
    print(f"with a located reaction centre: {len(usable)} of {len(atoms)}")
    for line in mismatched[:10]:
        print(f"  dropped {line}")
    if len(mismatched) > 10:
        print(f"  ... and {len(mismatched) - 10} more")

    payload = {f"atoms/{k}": v for k, v in usable.items()}
    payload.update({f"centre/{k}": centres[k] for k in usable})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    size = args.output.stat().st_size / 1e6
    width = next(iter(usable.values())).shape[1]
    print(f"\nwritten: {args.output}  ({size:.1f} MB, {width} features per atom)")
    for kind in ("acyl", "amine"):
        ids = sorted(int(k.split("/")[1]) for k in usable if k.startswith(kind))
        print(f"  {kind:6s} {len(ids):3d}: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
