"""Chemical space of the reactant library and of the train / test split.

Each reactant is a Morgan fingerprint (radius 2, 2048 bits) of its SMILES;
two reactants are compared by Tanimoto similarity. For acyl chlorides the
C(=O)Cl group every one of them shares is left out: only atom environments
centred outside it set bits, so two acyl chlorides are compared by their R
groups (in context) rather than scoring high on the group they all carry. On
that basis this module reports

- how diverse each library is (pairwise similarity; diversity = 1 - similarity),
- its Murcko scaffolds and functional classes (SMARTS rules below), and
- how far the held-out test reactants sit from the training reactants: each
  test reactant's similarity to its nearest training reactant, set against the
  same quantity for every training reactant measured against the rest of
  training.

Ported from the chemical-space analysis of the original PRISM repository
(data/chemical_space_analysis.py), with the same fingerprints, scaffold rules
and metrics; here it runs on the reactants of the labelled reactions and on
the study's actual held-out split. Needs RDKit.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

#: Fingerprint settings of the original analysis.
RADIUS, N_BITS = 2, 2048
#: Group common to a whole reactant kind, left out of its fingerprints.
COMMON_GROUP = {"acyl": "[CX3](=O)Cl"}

#: Amine type, first match wins (most specific first).
AMINE_SMARTS = {
    "heteroaromatic": "[n;H1]",
    "aromatic primary": "[NH2]c",
    "aromatic secondary": "[NH]c",
    "aliphatic primary": "[NH2;!$([NH2]c)]",
    "aliphatic secondary": "[NH;!$([NH]c);!$([NH]=*)]",
    "aliphatic tertiary": "[N;H0;!$(N=*);!$(Nc)]",
}
#: Acyl chloride type by the carbon on the carbonyl, first match wins.
#: Heteroaromatic is decided before these (see classify_acyl_chloride).
ACYL_SMARTS = {
    "aromatic": "[C](=O)(Cl)c",
    "alkenyl": "[C](=O)(Cl)[CX3]=[CX3]",
    "aliphatic tertiary": "[C](=O)(Cl)[CX4;H0]([CX4])([CX4])[CX4]",
    "aliphatic secondary": "[C](=O)(Cl)[CX4;H1]([C])[C]",
    "aliphatic primary": "[C](=O)(Cl)[CX4;H2,H3]",
}


def _chem():
    try:
        from rdkit import Chem, DataStructs
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("chemical-space analysis needs RDKit: "
                          "pip install -e '.[features]'") from exc
    return Chem, DataStructs


def fingerprints(smiles: Sequence[str], exclude_smarts: str = None) -> list:
    """Morgan fingerprints; raises on a SMILES RDKit cannot parse.

    With `exclude_smarts`, atoms matching it do not centre any environment, so
    the bits a shared group contributes are dropped. Environments centred on
    neighbouring atoms still see the group, which keeps how the rest of the
    molecule is attached to it.
    """
    Chem, DataStructs = _chem()
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    generator = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    pattern = Chem.MolFromSmarts(exclude_smarts) if exclude_smarts else None
    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"unparseable SMILES: {smi}")
        if pattern is None:
            out.append(generator.GetFingerprint(mol))
            continue
        excluded = {i for match in mol.GetSubstructMatches(pattern) for i in match}
        keep = [i for i in range(mol.GetNumAtoms()) if i not in excluded]
        # An empty fromAtoms means "all atoms" to RDKit; nothing left -> no bits.
        out.append(generator.GetFingerprint(mol, fromAtoms=keep) if keep
                   else DataStructs.ExplicitBitVect(N_BITS))
    return out


def similarity(rows: list, cols: list = None) -> np.ndarray:
    """Tanimoto similarity matrix, rows x cols (cols defaults to rows)."""
    _, DataStructs = _chem()
    cols = rows if cols is None else cols
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, cols) for fp in rows])


def upper(matrix: np.ndarray) -> np.ndarray:
    """Distinct pairs of a square similarity matrix (i < j)."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def murcko_scaffold(smiles: str) -> str:
    """Canonical Murcko scaffold; empty for an acyclic molecule."""
    Chem, _ = _chem()
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles))
    return Chem.MolToSmiles(scaffold) if scaffold else ""


def classify_amine(smiles: str) -> str:
    Chem, _ = _chem()
    mol = Chem.MolFromSmiles(smiles)
    for label, smarts in AMINE_SMARTS.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return label
    return "other"


def classify_acyl_chloride(smiles: str) -> str:
    """By the carbon on the carbonyl. Heteroaromatic: that carbon is in an
    aromatic ring holding N, O or S anywhere in the ring."""
    Chem, _ = _chem()
    mol = Chem.MolFromSmiles(smiles)
    match = mol.GetSubstructMatch(Chem.MolFromSmarts("[C](=O)(Cl)[#6]"))
    if match:
        alpha = mol.GetAtomWithIdx(match[3])
        if alpha.GetIsAromatic() and alpha.GetAtomicNum() == 6:
            for ring in mol.GetRingInfo().AtomRings():
                if match[3] in ring and any(
                        mol.GetAtomWithIdx(i).GetAtomicNum() in (7, 8, 16) for i in ring):
                    return "heteroaromatic"
    for label, smarts in ACYL_SMARTS.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return label
    return "other"


def describe(table: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Add functional class and Murcko scaffold to a table with `smiles`."""
    classify = classify_amine if kind == "amine" else classify_acyl_chloride
    out = table.copy()
    out["functional_class"] = out.smiles.map(classify)
    out["scaffold"] = out.smiles.map(murcko_scaffold)
    return out


def _stats(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std()), "median": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def analyse(table: pd.DataFrame, kind: str, exclude_common: bool = True
            ) -> Dict[str, object]:
    """Library and train/test metrics for one reactant kind.

    `table` has columns id, smiles and split ('train' or 'test'), ordered as
    the heatmaps should show them. `exclude_common` leaves the kind's shared
    group (COMMON_GROUP) out of the fingerprints.
    """
    table = describe(table, kind).reset_index(drop=True)
    fps = fingerprints(table.smiles, COMMON_GROUP.get(kind) if exclude_common else None)
    full = similarity(fps)
    train = np.flatnonzero(table.split.to_numpy() == "train")
    test = np.flatnonzero(table.split.to_numpy() == "test")

    test_x_train = full[np.ix_(test, train)]
    nearest_test = test_x_train.max(axis=1)
    within_train = full[np.ix_(train, train)].copy()
    np.fill_diagonal(within_train, -np.inf)
    nearest_train = within_train.max(axis=1)     # leave-one-out, inside training

    table["nearest_train_similarity"] = np.nan
    table.loc[test, "nearest_train_similarity"] = nearest_test
    table.loc[train, "nearest_train_similarity"] = nearest_train
    table["nearest_train_id"] = pd.NA
    train_ids = table.id.to_numpy()[train]
    table.loc[test, "nearest_train_id"] = train_ids[test_x_train.argmax(axis=1)]
    within_train[np.isinf(within_train)] = 0
    table.loc[train, "nearest_train_id"] = train_ids[within_train.argmax(axis=1)]

    train_scaffolds = set(table.scaffold.iloc[train])
    test_scaffolds = table.scaffold.iloc[test]
    pairs = upper(full)
    summary = {
        "n": len(table), "n_train": len(train), "n_test": len(test),
        "similarity_all": _stats(pairs),
        "diversity_all": _stats(1 - pairs),
        "distinct_scaffolds": int(table.scaffold.nunique()),
        "acyclic": int((table.scaffold == "").sum()),
        "functional_classes": table.functional_class.value_counts().to_dict(),
        "similarity_train_train": _stats(upper(full[np.ix_(train, train)])),
        "similarity_test_test": _stats(upper(full[np.ix_(test, test)])) if len(test) > 1 else None,
        "similarity_test_train": _stats(test_x_train.ravel()),
        "nearest_train_test": _stats(nearest_test),
        "nearest_train_train_loo": _stats(nearest_train),
        "test_scaffold_in_train": float(np.mean([s in train_scaffolds for s in test_scaffolds])),
        "test_scaffolds_not_in_train": int(len(set(test_scaffolds) - train_scaffolds)),
        "identical_fingerprints": [
            (int(table.id[i]), int(table.id[j]))
            for i, j in zip(*np.triu_indices_from(full, k=1)) if full[i, j] >= 1.0 - 1e-9],
    }
    return {"table": table, "similarity": full, "train": train, "test": test,
            "summary": summary}


def summary_text(results: Dict[str, Dict[str, object]], n_reactions: int) -> str:
    """Plain-text report of `analyse` for both kinds."""
    lines = ["Chemical space of the PRISM reactant library",
             f"Morgan fingerprints (radius {RADIUS}, {N_BITS} bits), Tanimoto similarity; "
             "diversity = 1 - similarity",
             "Acyl chlorides: the shared C(=O)Cl group is left out of the fingerprints "
             "(no environment centred on it).", ""]
    a, m = results["acyl"]["summary"], results["amine"]["summary"]
    lines += [f"{n_reactions} labelled reactions between {a['n']} acyl chlorides and "
              f"{m['n']} amines ({a['n'] * m['n']} possible pairs).", ""]
    for kind, title in (("acyl", "Acyl chlorides"), ("amine", "Amines")):
        s = results[kind]["summary"]
        classes = ", ".join(f"{v} {k}" for k, v in s["functional_classes"].items())
        d, tt, te, xt = (s["diversity_all"], s["similarity_train_train"],
                         s["similarity_test_test"], s["similarity_test_train"])
        nt, nl = s["nearest_train_test"], s["nearest_train_train_loo"]
        lines += [
            title, "-" * len(title),
            f"  all data: {s['n']} reactants, {len(s['functional_classes'])} functional "
            f"classes ({classes}), {s['distinct_scaffolds']} Murcko scaffolds "
            f"({s['acyclic']} acyclic)",
            f"  pairwise diversity: {d['mean']:.3f} +/- {d['sd']:.3f} "
            f"(range {d['min']:.3f}-{d['max']:.3f})",
            f"  split: train {s['n_train']}, test {s['n_test']}",
            f"  mean pairwise similarity: train-train {tt['mean']:.3f}; "
            + (f"test-test {te['mean']:.3f}; " if te else "")
            + f"test-train {xt['mean']:.3f}",
            f"  nearest training reactant, per test reactant: mean {nt['mean']:.3f} "
            f"(range {nt['min']:.3f}-{nt['max']:.3f})",
            f"  nearest other training reactant, per training reactant: mean {nl['mean']:.3f} "
            f"(range {nl['min']:.3f}-{nl['max']:.3f})",
            f"  test reactants whose scaffold occurs in training: "
            f"{s['test_scaffold_in_train']:.2f}; distinct test scaffolds not in training: "
            f"{s['test_scaffolds_not_in_train']}",
            ""]
    identical = [(title, pair) for kind, title in (("acyl", "acyl chlorides"), ("amine", "amines"))
                 for pair in results[kind]["summary"]["identical_fingerprints"]]
    if identical:
        lines += ["Identical fingerprints (similarity 1.0). Radius-2 Morgan fingerprints do not",
                  "distinguish alkyl chains beyond a few carbons, so homologues coincide:"]
        lines += [f"  {title}: {a} and {b}" for title, (a, b) in identical]
    return "\n".join(lines)
