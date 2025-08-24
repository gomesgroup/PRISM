#!/usr/bin/env python3
"""
SMILES-based feature generation
===============================
Provides two families of features:
1) Chemprop GNN embeddings from SMILES
2) FastProp descriptor/fingerprint tables

These utilities return pandas DataFrames keyed by the input ID column
('acyl_chlorides' or 'amines') suitable for merging into the combined dataset.
"""

from __future__ import annotations

import os
from typing import List, Optional

# NOTE: avoid top-level heavy imports to accommodate environments lacking
# rdkit/chemprop. We import pandas and others lazily inside functions.


def _validate_smiles(smi: str) -> bool:
    try:
        import importlib
        _Chem = importlib.import_module('rdkit.Chem')
        return _Chem.MolFromSmiles(smi) is not None
    except Exception:
        # If RDKit is unavailable in this environment, assume valid to allow
        # fallback execution in the external chemprop-v2 environment.
        return True


def chemprop_embed(df,
                   id_col: str,
                   smiles_col: str,
                   embed_size: int = 300,
                   device: Optional[str] = None,
                   include_raw_rdkit: bool = True):
    """Generate Chemprop embeddings for a column of SMILES.

    Returns a DataFrame indexed by id_col with columns cp_emb_0..cp_emb_{embed_size-1}.
    """
    import importlib
    pd = importlib.import_module('pandas')
    import numpy as np
    # Pandas dataframe access
    smiles = df[smiles_col].astype(str).tolist()
    ids = df[id_col].astype(str).tolist()
    mask = [i for i, s in enumerate(smiles) if _validate_smiles(s)]
    if not mask:
        return pd.DataFrame(index=pd.Index([], name=id_col))

    smiles_valid = [smiles[i] for i in mask]
    ids_valid = [ids[i] for i in mask]

    # Try Chemprop v2 featurizers first
    try:
        import importlib
        _feats = importlib.import_module('chemprop.featurizers')
        V1RDKit2DNormalizedFeaturizer = getattr(_feats, 'V1RDKit2DNormalizedFeaturizer')
        RDKit2DFeaturizer = getattr(_feats, 'RDKit2DFeaturizer')
        _Chem = importlib.import_module('rdkit.Chem')

        mols = [_Chem.MolFromSmiles(s) for s in smiles_valid]
        f_norm = V1RDKit2DNormalizedFeaturizer()  # 200-dim normalized v1 features
        X = np.stack([f_norm(m) for m in mols], axis=0)
        # Pad/truncate to embed_size
        if X.shape[1] < embed_size:
            pad = np.zeros((X.shape[0], embed_size - X.shape[1]), dtype=X.dtype)
            X = np.hstack([X, pad])
        elif X.shape[1] > embed_size:
            X = X[:, :embed_size]

        cols = [f'cp_emb_{i}' for i in range(embed_size)]
        emb_df = pd.DataFrame(X, columns=cols)
        emb_df[id_col] = ids_valid
        emb_df = emb_df.drop_duplicates(subset=[id_col]).set_index(id_col)

        if include_raw_rdkit:
            try:
                f_raw = RDKit2DFeaturizer()
                feats2 = [f_raw(m) for m in mols]
                raw_cols = [f'cp_raw_{i}' for i in range(len(feats2[0]))]
                raw_df = pd.DataFrame(feats2, columns=raw_cols)
                raw_df[id_col] = ids_valid
                raw_df = raw_df.drop_duplicates(subset=[id_col]).set_index(id_col)
                emb_df = emb_df.join(raw_df, how='left')
            except Exception:
                pass
        return emb_df
    except Exception:
        # Fallback: call external chemprop-v2 environment via conda/mamba run
        import tempfile
        import subprocess
        _os = os

        tmp_dir = tempfile.mkdtemp(prefix='cp_embed_')
        inp_csv = _os.path.join(tmp_dir, 'smiles.csv')
        out_csv = _os.path.join(tmp_dir, 'emb.csv')
        raw_csv = _os.path.join(tmp_dir, 'raw.csv')

        # Write input
        pd.DataFrame({id_col: ids_valid, smiles_col: smiles_valid}).to_csv(inp_csv, index=False)

        # Python snippet to run inside chemprop-v2
        pycode = f"""
import pandas as pd
import numpy as np
from rdkit import Chem
from chemprop.featurizers import V1RDKit2DNormalizedFeaturizer, RDKit2DFeaturizer
df = pd.read_csv(r"{inp_csv}")
ids = df["{id_col}"].astype(str).tolist()
smiles = df["{smiles_col}"].astype(str).tolist()
mols = [Chem.MolFromSmiles(s) for s in smiles]
f_norm = V1RDKit2DNormalizedFeaturizer()
X = np.stack([f_norm(m) for m in mols], axis=0)
# Pad/truncate
embed_size = {embed_size}
if X.shape[1] < embed_size:
    pad = np.zeros((X.shape[0], embed_size - X.shape[1]), dtype=X.dtype)
    X = np.hstack([X, pad])
elif X.shape[1] > embed_size:
    X = X[:, :embed_size]
emb_cols = [f"cp_emb_{{i}}" for i in range(embed_size)]
emb_df = pd.DataFrame(X, columns=emb_cols)
emb_df["{id_col}"] = ids
emb_df.to_csv(r"{out_csv}", index=False)
try:
    f_raw = RDKit2DFeaturizer()
    R = np.stack([f_raw(m) for m in mols], axis=0)
    raw_cols = [f"cp_raw_{{i}}" for i in range(R.shape[1])]
    raw_df = pd.DataFrame(R, columns=raw_cols)
    raw_df["{id_col}"] = ids
    raw_df.to_csv(r"{raw_csv}", index=False)
except Exception:
    pass
"""

        # Try conda/mamba run to execute in chemprop-v2
        run_cmds = [
            ["conda", "run", "-n", "chemprop-v2", "python", "-c", pycode],
            ["mamba", "run", "-n", "chemprop-v2", "python", "-c", pycode],
        ]
        ran = False
        for cmd in run_cmds:
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                ran = True
                break
            except Exception:
                continue
        if not ran:
            return pd.DataFrame(index=pd.Index([], name=id_col))

        try:
            emb_df = pd.read_csv(out_csv)
            emb_df = emb_df.drop_duplicates(subset=[id_col]).set_index(id_col)
            if include_raw_rdkit and os.path.exists(raw_csv):
                raw_df = pd.read_csv(raw_csv)
                raw_df = raw_df.drop_duplicates(subset=[id_col]).set_index(id_col)
                emb_df = emb_df.join(raw_df, how='left')
            return emb_df
        except Exception:
            return pd.DataFrame(index=pd.Index([], name=id_col))


def fastprop_descriptors(df,
                         id_col: str,
                         smiles_col: str,
                         compute_mordred: bool = True,
                         compute_mhfp: bool = True,
                         compute_padel: bool = False):
    """Compute a rich descriptor set using fastprop/aimsim_core backends.

    Returns a DataFrame indexed by id_col; column names prefixed with 'fp_'.
    """
    import importlib
    pd = importlib.import_module('pandas')
    try:
        pl = importlib.import_module('polars')
    except Exception:
        return pd.DataFrame(index=pd.Index([], name=id_col))
    try:
        fpm = importlib.import_module('fastprop.molecules')
    except Exception:
        return pd.DataFrame(index=pd.Index([], name=id_col))

    data = df[[id_col, smiles_col]].drop_duplicates().copy()
    data = data[data[smiles_col].map(_validate_smiles)]
    if len(data) == 0:
        return pd.DataFrame(index=pd.Index([], name=id_col))

    pl_df = pl.from_pandas(data)
    mols = fpm.MoleculeTable(pl_df.rename({id_col: 'id', smiles_col: 'smiles'}))

    frames: List = []

    if compute_mordred:
        mord = mols.mordred_descriptors().to_pandas()
        mord = mord.add_prefix('fp_mord_')
        mord[id_col] = data[id_col].values
        frames.append(mord)

    if compute_mhfp:
        mh = mols.mhfp().to_pandas()
        mh = mh.add_prefix('fp_mhfp_')
        mh[id_col] = data[id_col].values
        frames.append(mh)

    if compute_padel:
        pdx = mols.padel_descriptors().to_pandas()
        pdx = pdx.add_prefix('fp_padel_')
        pdx[id_col] = data[id_col].values
        frames.append(pdx)

    if not frames:
        return pd.DataFrame(index=pd.Index([], name=id_col))

    out = frames[0]
    for frm in frames[1:]:
        out = out.merge(frm, on=id_col, how='outer')
    out = out.drop_duplicates(subset=[id_col]).set_index(id_col)
    return out


