# AIMNet2 checkpoint

`graph_network_aimnet`, one of the graph networks in the learner benchmark,
reads per-atom features from a TorchScript AIMNet2 model. The model is the
ωB97M-D3 / implicit-solvent (CPCM) checkpoint distributed with
[LoQI](https://github.com/isayevlab/LoQI) (Isayev lab, MIT license):

| | |
|---|---|
| file | `wb97m_cpcms_v2_0.jpt` (9.3 MB) |
| md5 | `612d58aff24a8396cce21131ce7b7fcd` |
| source | `src/megalodon/metrics/aimnet2/cpcm_model/` in the LoQI repository |

The checkpoint is not redistributed here.

## You don't need it to run the package

The features it produces are tracked in `data/aimnet2_atoms.npz`, and the
benchmark reads that file. Only `scripts/extract_aimnet_features.py` loads
the checkpoint.

## Regenerating the features

You need torch (`pip install -e ".[gnn]"`) and the checkpoint:

```bash
curl -L -o data/aimnet2/wb97m_cpcms_v2_0.jpt \
  https://github.com/isayevlab/LoQI/raw/main/src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt
python scripts/extract_aimnet_features.py      # writes data/aimnet2_atoms.npz
python -m pytest -q tests/test_graph.py        # checks the file against the checkpoint
```

To use a checkpoint stored elsewhere, set `PRISM_AIMNET_MODEL=/path/to/file.jpt`.
To benchmark a different feature file, set `PRISM_AIMNET_NPZ=/path/to/file.npz`.

## What the stored file contains

Features for 46 acyl chlorides and 55 amines, computed from the optimised
geometries in `data/xyz` (neutral, singlet). Each atom has 513 values:

- 256 learned atomic features
- 256 atoms-in-molecules features
- the partial charge

The file also stores the reaction-centre atom of each reactant: the carbonyl
carbon for acyl chlorides, the nucleophilic nitrogen for amines.

## Earlier checkpoint

Before October 2026 the features came from a different AIMNet2 checkpoint,
`model_wb_nse_0.jpt`, whose source is not recorded. Compared with it:

- The reaction centres are identical for all 101 reactants.
- Partial charges agree closely (r 0.986, mean absolute difference 0.010 e).
- The learned feature vectors are unrelated element by element. That is
  expected: separately trained networks learn different internal
  representations.

Benchmark scores with the current checkpoint are in `docs/RESULTS.md`; the
AIMNet2 graph network ranks well below the descriptor-based models.

## Citation

If you use these features, cite LoQI and AIMNet2:

- F. Nikitin, D. M. Anstine, R. Zubatyuk, S. G. Paliwal, O. Isayev,
  *Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical
  Accuracy*, ChemRxiv (2025).
  <https://chemrxiv.org/engage/chemrxiv/article-details/689f756ea94eede154ec6217>
- D. M. Anstine, R. Zubatyuk, O. Isayev, *AIMNet2: a neural network potential
  to meet your neutral, charged, organic, and elemental-organic needs*,
  Chem. Sci. (2025).
