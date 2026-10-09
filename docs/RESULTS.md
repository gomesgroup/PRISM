# Results

Model selection and held-out performance at the **1.5× hydrolysis control**.
Every number comes from `results/` and `analysis/results/` and is regenerated
by `./run_all.sh`.

## Data and split

1085 acyl chloride / amine pairs (31 acyl chlorides, 35 amines). Of these, 363
have a fitted rate constant, 117 were too fast to resolve, and 605 are
un-reactive. A pair is *reactive* if a rate constant was measured or the
reaction was too fast to resolve, and *un-reactive* if the acyl chloride was
consumed by a competing pathway first.

The rate tables are built from the image-processing exports by
`scripts/build_rate_tables.py`. Each reactant is labelled by name from
`PRISM_substrate_scope.xlsx`. The three replicate batches of
4-fluorobenzoyl chloride (acyl chloride 1: 1A, 2B, 3C) are combined by the
geometric mean of their usable rate constants (`data/README.md`).

### Held-out test set

Whole reactants are held out: **acyl chlorides 2, 18, 25 and amines 18, 21,
23** (`config.FIXED_TEST_SET`).

The test reactants were chosen with a class-stratified rule
(`prism_ml.splits.build_split`), with no random draw, and then fixed. The rule
picks typical members by their number of measurable reactions, so its exact
picks depend on the rate tables; fixing the set keeps the held-out reactants
independent of later processing changes. The rule is applied separately to the
acyl chlorides and to the amines:

1. **Group by chemical class.** Each reactant takes the `class` column of
   `PRISM_substrate_features.xlsx`.
2. **Decide how many to take from each class, by its size:**
   - 2 members or fewer: none. The class stays wholly in training, so
     leave-one-reactant-out can still hold one member out while a sibling
     remains in training.
   - 3–10 members: one.
   - More than 10: two.
3. **Pick typical members, not extremes.** Members are sorted by number of
   measurable reactions. One pick is the median; two picks are the members one
   third and two thirds of the way along.
4. **Exclusions.** Amine 34 is never held out, because its descriptors came
   from a different pipeline.

| reactant | class label (members) | held out | stays in training |
|---|---|---|---|
| acyl chloride | `aromatic` (19) | **2** benzoyl chloride, **18** 4-methylbenzoyl chloride | — |
| acyl chloride | `aliphatic` (8) | **25** dodecanoyl chloride | — |
| acyl chloride | `hetero` (2), `mixture` (2) | — | 7, 20; 21, 27 |
| amine | `1_aromatic` (22) | **18** 3-aminoacetophenone, **21** 2-(2-methoxyethoxy)aniline | — |
| amine | `2_aliphatic` (8) | **23** 2,6-dimethylpiperidine | — |
| amine | `1_aliphatic` (2), `1_hetero` (1), `1_mixture` (1), `2_mixture` (1) | — | 1, 22; 17; 29; 32 |

Amine class labels start with 1 for primary and 2 for secondary amines.

Every pair containing a held-out reactant goes to the test set. That gives
three regimes: unseen acyl chloride, unseen amine, and both unseen.

| set | pairs | with a rate constant | too fast | un-reactive |
|---|---|---|---|---|
| training pool | 896 | 294 | 104 | 498 |
| held-out test (unseen acyl / unseen amine / both unseen) | 189 (96 / 84 / 9) | 69 | 13 | 107 |

## Model selection

Selection uses only the training pool. Two cross-validation schemes are run on
it, on the same candidates:

- **Repeated class-stratified validation (used for selection).** In each of
  20 draws (seeds 1–20), 3 acyl chlorides and 3 amines are held out by the
  same class rule as the test set, with the members chosen at random. Every
  reaction with a held-out reactant (171) is validation; the model trains on
  the other 725. This mirrors how the test set is built: several unseen
  reactants at once, spread over the chemical classes.
- **Leave-one-reactant-out (reported alongside).** Each of the 28 acyl
  chlorides and 32 amines is held out in turn.

Each candidate is scored by **joint = ½ accuracy + ½ R²** (R² = 1 −
SS_res/SS_tot on log₁₀ *k*, floored at 0). In the stratified scheme the joint
score is computed per draw and averaged over the 20 draws; the candidate with
the highest mean is selected. The descriptor set is chosen first, with a fixed
learner (histogram gradient boosting); the learner is then chosen on that
descriptor set. The held-out test, excluded scope and drug scope are not used.

## 1. Descriptor set (`results/benchmark/feature_sets/`)

Histogram gradient boosting on each descriptor tier:

| tier | descriptors | stratified accuracy | stratified R² | **stratified joint** (mean ± SD) | leave-one-reactant-out joint |
|---|---|---|---|---|---|
| **Tier 2 (reduced), selected** | 52 | 0.731 | **0.482** | **0.620 ± 0.104** | 0.651 |
| Tier 2 | 56 | 0.730 | 0.459 | 0.608 ± 0.104 | 0.645 |
| Tier 3 (reduced) | 68 | 0.719 | 0.316 | 0.599 ± 0.125 | **0.657** |
| Tier 3 | 72 | 0.719 | 0.303 | 0.593 ± 0.120 | 0.641 |
| Tier 1 | 34 | **0.733** | 0.349 | 0.556 ± 0.129 | 0.619 |

- Tier 1 is geometry only (buried volume, Sterimol, SASA, pyramidalization,
  dispersion).
- Tier 2 adds one xTB single point.
- Tier 2 (reduced) drops electron affinity and hardness.
- Tier 3 adds Fukui indices, which need two more xTB calculations.

Leave-one-reactant-out puts Tier 3 (reduced) marginally ahead (by 0.006).
With several reactants held out at once, Tier 3 (reduced) predicts rate
constants less reliably (R² 0.32 vs 0.48), with a larger spread across draws.
Tier 1 classifies about as well as the others, but without the electronic
descriptors the rate constant is predicted much less well, as expected from
the Hammett analysis.

## 2. Learner (`results/benchmark/model_families/`)

All learners use Tier 2 (reduced). The graph networks read molecular graphs
instead of descriptors. One of them adds per-atom AIMNet2 features from the
ωB97M-D3/CPCM checkpoint distributed with
[LoQI](https://github.com/isayevlab/LoQI) (`data/aimnet2/README.md`).

| learner | stratified accuracy | stratified R² | **stratified joint** (mean ± SD) | leave-one-reactant-out joint |
|---|---|---|---|---|
| **LightGBM, selected** | 0.742 | **0.499** | **0.629 ± 0.105** | 0.651 |
| split: exact GB (class) + HistGB (rate) | 0.743 | 0.482 | 0.626 ± 0.100 | 0.654 |
| extremely randomised trees | **0.764** | 0.438 | 0.621 ± 0.130 | **0.659** |
| gradient-boosted trees (HistGB) | 0.731 | 0.482 | 0.620 ± 0.104 | 0.651 |
| random forest | 0.756 | 0.480 | 0.618 ± 0.088 | 0.655 |
| XGBoost | 0.737 | 0.490 | 0.613 ± 0.095 | 0.634 |
| CatBoost | 0.749 | 0.375 | 0.591 ± 0.129 | 0.636 |
| gradient boosting (exact) | 0.743 | 0.398 | 0.582 ± 0.125 | 0.634 |
| AdaBoost | 0.733 | 0.357 | 0.561 ± 0.102 | 0.605 |
| k-nearest neighbours | 0.709 | 0.320 | 0.528 ± 0.121 | 0.575 |
| graph network + AIMNet2 atoms (LoQI checkpoint) | 0.684 | 0.117 | 0.514 ± 0.164 | 0.471 |
| linear (logistic / ridge) | 0.710 | −0.256 | 0.474 ± 0.152 | 0.372 |
| graph network | 0.715 | −0.537 | 0.397 ± 0.088 | 0.350 |
| uninformative baseline | 0.507 | −0.228 | 0.254 ± 0.042 | 0.278 |

LightGBM is selected: it has the best mean joint score and the best rate
constant R² under stratified validation. The tree ensembles that follow are
close behind (within 0.011), so other boosted learners would perform
similarly. Both heads of the selected model are LightGBM, with 200 trees,
learning rate 0.1 and 15 leaves, without tuning.

## 3. Held-out evaluation (`results/holdout/`)

The selected model (Tier 2 reduced, LightGBM) is fitted once on the training
pool and then scored on data it never saw.

**Held-out test (189 pairs).**
- Reactivity: accuracy 0.83 (majority-class baseline 0.57), precision 0.86,
  recall 0.73.
- log₁₀ *k* (69 pairs with a measured rate constant): R² 0.79, MAE 0.29.
  90% of residuals fall within 0.93 log units.
- 71% of these 69 pairs are classified reactive and so receive a rate
  constant; their MAE is 0.28.

| regime | pairs | accuracy | rate R² (n) | rate MAE |
|---|---|---|---|---|
| unseen acyl chloride | 96 | 0.83 | 0.82 (28) | 0.28 |
| unseen amine | 84 | 0.85 | 0.76 (38) | 0.29 |
| both unseen | 9 | 0.67 | 0.84 (3) | 0.38 |

Two reactions fall outside ±2σ in the residuals plot (`residuals_test.png`),
both with amine 21 (2-(2-methoxyethoxy)aniline): rxn_19_21 and rxn_22_21.
Both are under-predicted, by factors of about 12 and 5. Amine 21 is
under-predicted in 15 of its 19 measured reactions (mean residual +0.25 log
units).

**Excluded substrate scope** (SI Section 7). 30 of the 41 excluded substrates
have a class label; the 11 insoluble ones are not scored. Accuracy is 0.87,
with no substrate wrongly called reactive.

| note | substrates | correct |
|---|---|---|
| too fast (reactive) | 18 | 14 |
| too slow | 4 | 4 |
| non-reactive | 3 | 3 |
| multiple products | 5 | 5 |

Each scope amine is paired with acyl chloride 1, and each scope acyl chloride
with amine 3.

**Drug-like amines (11 pairs, all reactive).** 7 of 11 are called reactive.
The two pairs with a measured rate constant have MAE 0.06 log units.

**All three pooled (230 calls).** Accuracy 0.83, precision 0.89, recall 0.73.

| regime | calls | accuracy |
|---|---|---|
| unseen acyl chloride | 96 | 0.83 |
| unseen amine | 124 | 0.83 |
| both unseen | 10 | 0.70 |

Figures in `results/holdout/figures/`:
- `confusion_test_scope_drug.png`: the three sets side by side
- `confusion_*held-out-scope.png`: pooled, by regime and by novelty
- `parity_test.png`, `parity_train_cv_test.png` (cross-validated points are
  leave-one-reactant-out)
- `residuals_test.png`

## Chemical space (`analysis/results/chemical_space/`)

`python analysis/chemical_space.py`. Each reactant is encoded as a Morgan
fingerprint (radius 2, 2048 bits) and compared by Tanimoto similarity. The
analysis covers all reactants in the labelled reactions and the held-out
split.

For acyl chlorides, the C(=O)Cl group they all share is left out of the
fingerprints, so they are compared by their R groups. With the group kept, it
inflates similarity: mean pairwise similarity is 0.292 with it and 0.130
without.

Figures:
- similarity heatmaps of all reactants, train | test;
- pairwise diversity (1 − similarity) of acyl chlorides and amines, overlaid;
- pairwise similarity distributions;
- each reactant's similarity to its nearest training reactant.

| | acyl chlorides | amines |
|---|---|---|
| reactants (train / test) | 31 (28 / 3) | 35 (32 / 3) |
| functional classes / Murcko scaffolds | 7 / 7 | 4 / 12 |
| mean pairwise diversity (1 − similarity) | 0.870 ± 0.129 | 0.855 ± 0.130 |
| mean similarity: train–train / test–train | 0.122 / 0.169 | 0.147 / 0.139 |
| nearest training reactant: test (mean) | 0.653 | 0.491 |
| nearest training reactant: training, leave-one-out (mean) | 0.336 | 0.430 |

The test reactants are at least as close to training as training reactants
are to each other. The held-out split therefore tests unseen reactants inside
the library's chemical space, not extrapolation beyond it.

Radius-2 fingerprints do not distinguish alkyl chain lengths beyond a few
carbons, so two homologue pairs have identical fingerprints:
- acyl chlorides 13 and 25 (heptanoyl and dodecanoyl chloride; 25 is a test
  reactant);
- amines 7 and 10 (4-heptyl- and 4-hexylaniline).

## Data summary (`analysis/results/data_summary/`)

`python analysis/data_summary.py` gives:
- class counts for all data, the training pool and each held-out regime;
- the distribution of measured log₁₀ *k*: 363 values from −0.03 to 3.41,
  mean 1.50 ± 0.70;
- a map of the class of every acyl chloride × amine pair.

## 4. Deployed model (`model/`)

The model in `model/` is trained on **all 1085 reactions**. Its descriptors are
computed from SMILES by `prism_ml.descriptors`, the same pipeline that
`scripts/predict.py` runs on an unseen molecule.

The stored descriptor table (`PRISM_substrate_features.xlsx`, used for
sections 1–3) came from a different conformer workflow. For several acyl
chloride descriptors it does not agree with the SMILES pipeline. For example,
SMILES-computed C=O carbon charge correlates only r = 0.28 with the stored
value, so a model trained on the table would not transfer to unseen molecules.

Before the final fit, the same pipeline is validated on the held-out split:

| | accuracy | precision | recall | rate R² | rate MAE |
|---|---|---|---|---|---|
| SMILES-descriptor pipeline, held-out test | 0.78 | 0.72 | 0.83 | 0.81 | 0.24 |

These numbers are also stored in `model/metadata.json`.

## Reproducibility

- The histogram learners (HistGB, LightGBM, run deterministically) give
  identical scores on every machine tried.
- Exact-split learners (exact gradient boosting, AdaBoost) can differ in the
  third decimal between CPU types.
- The graph networks train on a GPU. They reproduce exactly on the same GPU
  (NVIDIA A30), but can differ by about 0.015 in joint score on other
  hardware.
