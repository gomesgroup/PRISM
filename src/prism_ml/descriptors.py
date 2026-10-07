"""Morfeus / xTB descriptors for unseen substrates, computed from SMILES.

A model can only score an unseen acyl chloride or amine if its descriptors are
computed the way the training descriptors were. This reproduces that pipeline:
RDKit embedding and MMFF optimisation to one conformer (fixed seed), morfeus
for the geometric descriptors, and morfeus's xTB interface for the electronic
ones.

Reaction centres:

    acyl chloride  main atom = the carbon bearing chlorine; secondary atoms =
                   its heavy non-oxygen neighbours (chlorine first, then the
                   alpha carbon); hydrogens = those on the alpha carbon
    amine          main atom = a primary or secondary, non-amide nitrogen,
                   preferring the most hydrogens; secondary atoms = its heavy
                   neighbours

Needs the optional dependencies: ``pip install -e ".[features]"`` (RDKit,
morfeus) plus xtb-python, most easily from conda-forge. They are imported only
when a descriptor is actually computed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Columns written per substrate that record which atoms were used. Provenance,
#: not features: the model never sees them.
INDEX_COLUMNS = ("idx_main_atom", "idx_secondary_1", "idx_secondary_2",
                 "el_main_atom", "el_secondary_1", "el_secondary_2")


def _require():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        import morfeus
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "computing descriptors needs RDKit, morfeus and xtb-python; install "
            "with `pip install -e \".[features]\"` and "
            "`conda install -c conda-forge xtb-python`") from exc
    RDLogger.DisableLog("rdApp.*")
    return Chem, AllChem, morfeus


def _substituent_key(atom):
    """Deterministic chemical ordering of the substituents at a centre.

    Heavier element first (chlorine ahead of carbon), then aromatic before
    aliphatic, then the bulkier substituent, then atom index -- never the order
    in which the SMILES happened to be written.
    """
    from rdkit import Chem
    reach = Chem.GetDistanceMatrix(atom.GetOwningMol())[atom.GetIdx()]
    return (atom.GetAtomicNum(), int(atom.GetIsAromatic()),
            int(sum(1 for d in reach if d < 90)), -atom.GetIdx())


def embed(smiles: str, seed: int = 42):
    """One MMFF-optimised conformer."""
    Chem, AllChem, _ = _require()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True) != 0:
            raise ValueError(f"embedding failed: {smiles}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # noqa: BLE001 - MMFF lacks parameters for some atoms
        AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    return mol, elements, coords


def acyl_centres(mol) -> Tuple[Optional[int], List[int], List[int], int]:
    """Carbonyl carbon, its heavy non-O neighbours, alpha-carbon H's, H count."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or not any(
                n.GetAtomicNum() == 17 for n in atom.GetNeighbors()):
            continue
        secondary = [n.GetIdx() for n in sorted(
            (n for n in atom.GetNeighbors() if n.GetSymbol() not in ("H", "O")),
            key=_substituent_key, reverse=True)]
        hydrogens: List[int] = []
        for neighbour in atom.GetNeighbors():
            if neighbour.GetAtomicNum() != 6:
                continue
            hs = sorted(h.GetIdx() for h in neighbour.GetNeighbors() if h.GetAtomicNum() == 1)
            if hs:
                hydrogens = hs
        return atom.GetIdx(), secondary, hydrogens, len(hydrogens)
    return None, [], [], 0


def amine_centres(mol) -> Tuple[Optional[int], List[int]]:
    """Primary or secondary nitrogen that is not part of an amide."""
    best, best_secondary, best_h = None, [], -1
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 1]
        if len(heavy) >= 3:
            continue
        if any(n.GetSymbol() == "C" and any(o.GetSymbol() == "O" for o in n.GetNeighbors())
               for n in heavy):
            continue
        n_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        if n_h > best_h:
            best, best_h = atom.GetIdx(), n_h
            best_secondary = [n.GetIdx() for n in sorted(heavy, key=_substituent_key,
                                                         reverse=True)]
    return best, best_secondary


def _mean(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def compute(smiles: str, kind: str, with_xtb: bool = True) -> Dict[str, float]:
    """Descriptors for one substrate. ``kind`` is "acyl" or "amine"."""
    _, _, morfeus = _require()
    kind = "acyl" if kind.lower().startswith("acyl") else "amine"
    mol, elements, coords = embed(smiles)
    if kind == "acyl":
        main, secondary, hydrogens, num_hs = acyl_centres(mol)
    else:
        (main, secondary), hydrogens, num_hs = amine_centres(mol), [], 0
    if main is None:
        raise ValueError(f"no reaction centre found for {kind}: {smiles}")
    m1, sec1, hyd1 = main + 1, [i + 1 for i in secondary], [i + 1 for i in hydrogens]

    out: Dict[str, float] = {
        "idx_main_atom": float(main),
        "idx_secondary_1": float(secondary[0]) if secondary else float("nan"),
        "idx_secondary_2": float(secondary[1]) if len(secondary) > 1 else float("nan"),
        "el_main_atom": elements[main],
        "el_secondary_1": elements[secondary[0]] if secondary else "",
        "el_secondary_2": elements[secondary[1]] if len(secondary) > 1 else "",
    }
    sasa = morfeus.SASA(elements, coords)
    disp = morfeus.Dispersion(elements, coords)
    out["SA"], out["SV"] = float(sasa.area), float(sasa.volume)
    out["PYR"] = float(morfeus.Pyramidalization(coords, m1).P)
    out["DISP_mol"], out["DISP_atom"] = float(disp.p_int), float(disp.atom_p_int[m1])
    bv = [float(morfeus.BuriedVolume(elements, coords, a).buried_volume) for a in [m1] + sec1]
    out["BV_main_1"] = bv[0]
    for j, v in enumerate(bv[1:], start=1):
        out[f"BV_secondary_{j}"] = v
    out["BV_secondary_avg"] = _mean(bv[1:])
    L, B1, B5 = [], [], []
    for a in sec1:
        st = morfeus.Sterimol(elements, coords, m1, a)
        L.append(float(st.L_value)); B1.append(float(st.B_1_value)); B5.append(float(st.B_5_value))
    for j, (l, b1, b5) in enumerate(zip(L, B1, B5), start=1):
        out[f"L_main_atom_{j}"], out[f"B1_main_atom_{j}"], out[f"B5_main_atom_{j}"] = l, b1, b5
    out["L_avg"], out["B1_avg"], out["B5_avg"] = _mean(L), _mean(B1), _mean(B5)
    if kind == "acyl":
        # An acyl chloride with no alpha hydrogen (every aryl one) records its
        # hydrogen averages as 0, as the training descriptors do; a blank here
        # would make the model refuse every aryl acyl chloride.
        out["num_Hs"] = float(num_hs)
        out["BV_Hs_avg"] = (_mean([float(morfeus.BuriedVolume(elements, coords, h).buried_volume)
                                   for h in hyd1]) if hyd1 else 0.0)

    if with_xtb:
        x = morfeus.XTB(elements, coords)
        ip = float(x.get_ip(corrected=True))
        # corrected=False restores the electron-affinity convention the training
        # descriptors used (morfeus later changed its default); EA and hardness
        # are not in the selected feature set but are kept for completeness.
        ea = float(x.get_ea(corrected=False))
        out.update(IP=ip, EA=ea, HOMO=float(x.get_homo()), LUMO=float(x.get_lumo()),
                   global_E=float(x.get_global_descriptor("electrophilicity", corrected=True)),
                   global_N=float(x.get_global_descriptor("nucleophilicity", corrected=True)),
                   hardness=ip - ea)
        charges = x.get_charges()
        fe, fn = x.get_fukui("electrophilicity"), x.get_fukui("nucleophilicity")
        out["Charges_main_atom"] = float(charges[m1])
        out["fukui_E_main_atom"], out["fukui_N_main_atom"] = float(fe[m1]), float(fn[m1])
        for j, a in enumerate(sec1, start=1):
            out[f"Charges_secondary_{j}"] = float(charges[a])
            out[f"fukui_E_secondary_{j}"] = float(fe[a])
            out[f"fukui_N_secondary_{j}"] = float(fn[a])
        out["Charges_secondary_avg"] = _mean([float(charges[a]) for a in sec1])
        out["fukui_E_secondary_avg"] = _mean([float(fe[a]) for a in sec1])
        out["fukui_N_secondary_avg"] = _mean([float(fn[a]) for a in sec1])
        if kind == "acyl":
            out["Charges_Hs_avg"] = _mean([float(charges[h]) for h in hyd1]) if hyd1 else 0.0
            out["fukui_E_Hs_avg"] = _mean([float(fe[h]) for h in hyd1]) if hyd1 else 0.0
            out["fukui_N_Hs_avg"] = _mean([float(fn[h]) for h in hyd1]) if hyd1 else 0.0
    return out
