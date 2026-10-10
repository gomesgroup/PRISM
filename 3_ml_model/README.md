# PRISM ML

Machine-learning models that predict, for an **acyl chloride + amine** pair,

1. **reactivity** — does the amide form? (*reactive* vs *un-reactive*), and
2. the **rate constant** — log₁₀ *k*, for pairs predicted to react,

trained on PRISM plate-reader kinetics at the **1.5× hydrolysis control**.

*Reactive* means a rate constant was measured, or the reaction is instant
(reactive but not measurable: too fast to resolve, or the wells beat the
hydrolysis control without giving a usable rate; the data notes say which).
*Un-reactive* means the acyl chloride was consumed by a competing pathway —
hydrolysis or another side reaction — before forming the amide. Pairs with too
few wells to compare with the hydrolysis control are undetermined and not
modelled (`data/README.md`).

## Selected model

| | |
|---|---|
| descriptors | Tier 2 (reduced): 52 morfeus / xTB descriptors (one conformer + one xTB single point) |
| reactivity | exact gradient boosting classifier (`GradientBoostingClassifier`) |
| rate constant | histogram gradient boosting regressor (`HistGradientBoostingRegressor`), trained on the reactions with a rate constant |

The held-out test set (acyl chlorides 2, 18, 25 and amines 18, 21, 23;
`config.FIXED_TEST_SET`) was selected at random under a class-stratified rule,
to cover a diverse set of each reactant's chemical classes, and then fixed. The descriptors and learner were selected on the
training pool alone, by repeated class-stratified validation. Each of 20
draws holds out acyl chlorides and amines the same way the test set is built.
The selected model was then scored once on data it never saw:

| evaluation | reactivity accuracy | log₁₀ *k* R² / MAE |
|---|---|---|
| held-out test (186 pairs, unseen reactants) | 0.83 | all 69 measured: 0.83 / 0.26; classified reactive (46): 0.83 / 0.27 |
| excluded substrate scope (30 substrates) | 0.93 | — |
| drug-like amines (11 pairs) | 9 / 11 | — |
| all three pooled (227 calls) | 0.85 | — |

The rate constant is reported two ways: over every measured test reaction
(the regression model alone), and over the reactions the classifier calls
reactive (what the two-stage model returns).

Full benchmark and held-out results: [`docs/RESULTS.md`](docs/RESULTS.md).
The rate tables in `data/datasheets/` come from `../2_rate_fitting`.

## Installation

```bash
git clone https://github.com/gomesgroup/PRISM.git && cd PRISM/3_ml_model
conda env create -f environment.yml
conda activate prism-ml
```

This installs everything needed to predict from SMILES (RDKit, morfeus,
xtb-python) and to rerun the benchmark. To only load the saved model and score
precomputed descriptors, `pip install -e .` is enough. Optional extras:
`.[features]` (descriptors from SMILES; xtb-python comes from conda-forge),
`.[boosting]` (XGBoost and CatBoost, for the learner benchmark), `.[gnn]` (graph networks, torch).

Check the installation:

```bash
python -m pytest -q
```

## Predict

```bash
# one pair
python scripts/predict.py --acyl "O=C(Cl)c1ccccc1" --amine "Nc1ccccc1"

# many pairs: CSV with columns acyl_smiles, amine_smiles
python scripts/predict.py --input examples/pairs.csv --output predictions.csv
```

```python
import pandas as pd
from prism_ml.predictor import Predictor

model = Predictor.load()   # the model in model/
model.predict_smiles(pd.DataFrame({"acyl_smiles": ["O=C(Cl)c1ccccc1"],
                                   "amine_smiles": ["Nc1ccccc1"]}))
```

Output: `p_reactive`, `reactive` (0/1), `log10_k`, and `log10_k_gated`
(blank where the pair is predicted un-reactive). Descriptors are computed on
the fly (RDKit conformer → MMFF → morfeus + xTB).

The saved model is trained on descriptors computed by that same SMILES
pipeline, so predictions for unseen molecules are consistent with training. Its
own held-out check is in `model/metadata.json` (accuracy 0.88, log₁₀ *k*
R² 0.81).

## Reproduce

```bash
./run_all.sh                 # tests, both benchmarks, held-out scoring, chemical space, final model
./run_all.sh --skip-graph    # without the graph networks (no torch / GPU needed)
```

or step by step:

| step | script | output |
|---|---|---|
| feature-set benchmark | `scripts/benchmark_features.py` | `results/benchmark/feature_sets/` |
| learner benchmark | `scripts/benchmark_models.py` | `results/benchmark/model_families/` |
| held-out, scope, drug | `scripts/holdout.py` | `results/holdout/` |
| chemical space (all data, train vs test) | `analysis/chemical_space.py` | `analysis/results/chemical_space/` |
| data summary (class counts, rate constants) | `analysis/data_summary.py` | `analysis/results/data_summary/` |
| descriptors from SMILES | `scripts/compute_descriptors.py` | `data/datasheets/PRISM_substrate_features_smiles.xlsx` |
| final model | `scripts/train_final.py` | `model/` |

On a SLURM cluster: `sbatch slurm/workflow_cpu.slurm` and
`sbatch slurm/benchmark_models_gpu.slurm` (edit the partition first).

## Layout

```
src/prism_ml/     library (data, features, splits, models, scoring, descriptors, predictor)
scripts/          the workflow steps above, plus predict.py
model/            the saved model and its metadata
results/          benchmark and held-out outputs
data/datasheets/  rate constants, reactivity classes, descriptors, substrate scope
data/             reactant geometries, AIMNet2 atom features (data dictionary: data/README.md)
analysis/         chemical-space analysis; NMR and Hammett comparison data and figures
examples/         example input for predict.py
tests/            test suite
```

The graph-network benchmark uses AIMNet2 atom features
(`data/aimnet2_atoms.npz`, included). They were computed with the AIMNet2
checkpoint distributed with [LoQI](https://github.com/isayevlab/LoQI), which
is needed only to regenerate them (`data/aimnet2/README.md`).

## Use of AI tools

Claude Code (Anthropic, `claude-opus-5-5`, October 2026) was used as a tool to
help organize and run this workflow.

## License

MIT (see [`LICENSE`](LICENSE)). The AIMNet2 checkpoint is third-party, from
the LoQI repository (MIT), and is not redistributed here.
