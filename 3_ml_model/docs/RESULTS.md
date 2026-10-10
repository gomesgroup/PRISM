# Results

Model selection and held-out performance at the **1.5× hydrolysis control**.
Every number comes from `results/` and `analysis/results/` and is regenerated
by `./run_all.sh`.

## Data and split

1085 acyl chloride / amine pairs (31 acyl chlorides, 35 amines). Of these, 363
have a fitted rate constant, 138 are instant (reactive but not measurable),
543 are un-reactive, and 41 are undetermined. The 1044 pairs with a class are
modelled. A pair is *reactive* if a rate constant was measured or it is
instant, and *un-reactive* if the acyl chloride was consumed by a competing
pathway first.

Instant pairs have no rate constant, and the `note` column of
`PRISM_rates_unmeasurable.xlsx` says why: too fast to resolve (111); the wells
beat the hydrolysis control but with identical times there is no slope (4);
the fitted slope is negative (2); or the control review below (21).

**Control review.** The per-well filter bins a pair as too slow when fewer
than two wells beat the no-amine control by 1.5×, without separating wells
that failed the control from wells that could not be compared with it. 62
slow-binned pairs had fewer than two comparable wells in every replicate. Of
these, 21 (12 with own-plate controls, 9 with borrowed controls) are instant:
each has at least two wells where the amine reacted faster than the hydrolysis
control at the same concentration, and none failed the control. This counts
wells where the amine reaction was complete by the first reading while the
control was not; such wells support the label but do not establish a 1.5×
ratio. The other 41 are undetermined. Replicate pairs that fail the control in
other batches (rxn_01_01, rxn_01_12, rxn_01_34) stay un-reactive.

The rate tables are written by `2_rate_fitting/scripts/prism_perwell.py`
(details in that folder's README). Each reactant is labelled by name from
`PRISM_substrate_scope.xlsx`. The three replicate batches of
4-fluorobenzoyl chloride (acyl chloride 1: 1A, 2B, 3C) are combined by the
geometric mean of their usable rate constants (`data/README.md`).

### Held-out test set

Whole reactants are held out: **acyl chlorides 2, 18, 25 and amines 18, 21,
23** (`config.FIXED_TEST_SET`).

The test reactants were selected at random under a class-stratified rule, so
that the test set covers a diverse set of each reactant's chemical classes,
and the set was then fixed. Fixing it keeps the held-out reactants the same
through later processing changes. The rule is applied separately to the acyl
chlorides and to the amines:

1. **Group by chemical class.** Each reactant takes the `class` column of
   `PRISM_substrate_features.xlsx`.
2. **Decide how many to take from each class, by its size:**
   - 2 members or fewer: none. The class stays wholly in training, so
     leave-one-reactant-out can still hold one member out while a sibling
     remains in training.
   - 3–10 members: one.
   - More than 10: two.
3. **Draw the members at random** within each class.
4. **Exclusions.** Amine 34 is never held out, because its descriptors came
   from a different pipeline.

The same rule, with random members, defines the validation draws used for
model selection (below). With `config.FIXED_TEST_SET = None`, the code instead
applies a deterministic version of the rule (`prism_ml.splits.build_split`,
which takes the members with a median number of measurable reactions).

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

| set | pairs modelled | with a rate constant | instant | un-reactive | undetermined (not modelled) |
|---|---|---|---|---|---|
| training pool | 858 | 294 | 120 | 444 | 38 |
| held-out test (unseen acyl / unseen amine / both unseen) | 186 (96 / 82 / 8) | 69 | 18 | 99 | 3 |

## Model selection

Selection uses only the training pool. Two cross-validation schemes are run on
it, on the same candidates:

- **Repeated class-stratified validation (used for selection).** In each of
  20 draws (seeds 1–20), 3 acyl chlorides and 3 amines are held out by the
  same class rule as the test set, with the members chosen at random. Every
  reaction with a held-out reactant (158–170 per draw) is validation; the
  model trains on the other 688–700. This mirrors how the test set is built: several unseen
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
| **Tier 2 (reduced), selected** | 52 | 0.774 | **0.482** | **0.642 ± 0.108** | 0.668 |
| Tier 2 | 56 | **0.777** | 0.459 | 0.631 ± 0.111 | 0.669 |
| Tier 3 (reduced) | 68 | 0.760 | 0.316 | 0.619 ± 0.133 | **0.677** |
| Tier 3 | 72 | 0.752 | 0.303 | 0.610 ± 0.127 | 0.662 |
| Tier 1 | 34 | 0.775 | 0.349 | 0.577 ± 0.128 | 0.638 |

- Tier 1 is geometry only (buried volume, Sterimol, SASA, pyramidalization,
  dispersion).
- Tier 2 adds one xTB single point.
- Tier 2 (reduced) drops electron affinity and hardness.
- Tier 3 adds Fukui indices, which need two more xTB calculations.

Leave-one-reactant-out puts Tier 3 (reduced) marginally ahead (by 0.009).
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
| **split: exact GB (class) + HistGB (rate), selected** | 0.790 | 0.482 | **0.650 ± 0.099** | 0.676 |
| LightGBM | 0.780 | **0.497** | 0.648 ± 0.104 | 0.678 |
| random forest | 0.812 | 0.476 | 0.644 ± 0.088 | **0.679** |
| gradient-boosted trees (HistGB) | 0.775 | 0.482 | 0.642 ± 0.108 | 0.668 |
| extremely randomised trees | 0.799 | 0.437 | 0.637 ± 0.123 | 0.679 |
| XGBoost | 0.782 | 0.490 | 0.636 ± 0.095 | 0.656 |
| CatBoost | **0.816** | 0.375 | 0.625 ± 0.119 | 0.663 |
| gradient boosting (exact) | 0.790 | 0.398 | 0.613 ± 0.121 | 0.641 |
| AdaBoost | 0.802 | 0.355 | 0.600 ± 0.105 | 0.634 |
| graph network + AIMNet2 atoms (LoQI checkpoint) | 0.749 | 0.194 | 0.570 ± 0.182 | 0.558 |
| k-nearest neighbours | 0.733 | 0.315 | 0.538 ± 0.118 | 0.591 |
| linear (logistic / ridge) | 0.769 | −0.254 | 0.503 ± 0.149 | 0.390 |
| graph network | 0.725 | −0.537 | 0.402 ± 0.098 | 0.356 |
| uninformative baseline | 0.447 | −0.228 | 0.224 ± 0.044 | 0.256 |

The split model is selected: it has the highest mean joint score. LightGBM
(0.001 behind) and random forest (0.006 behind) are effectively tied with it,
since the spread across draws is about 0.1, so other tree ensembles would
perform similarly. The selected model decides reactivity with exact gradient
boosting (200 trees, learning rate 0.1, depth 3) and predicts the rate
constant with histogram gradient boosting (200 iterations, learning rate 0.1,
15 leaves), without tuning.

## 3. Held-out evaluation (`results/holdout/`)

The selected model (Tier 2 reduced, split exact GB + HistGB) is fitted once on
the training pool and then scored on data it never saw.

**Held-out test (186 pairs; the 3 undetermined test pairs are not scored).**
- Reactivity: accuracy 0.83 (majority-class baseline 0.53), precision 0.90,
  recall 0.72.
- log₁₀ *k*, reported two ways:
  - **All measured test reactions** (69): R² 0.83, MAE 0.26; 90% of residuals
    fall within 0.87 log units. This scores the regression model on every
    reaction with a measured rate constant, whatever the classifier says.
  - **Classified reactive** (46 of the 69, 67% coverage): R² 0.83, MAE 0.27.
    This is what the two-stage model reports, since it gives a rate constant
    only where the classifier calls the reaction reactive. It shows how well
    the two models agree.
- The 23 measured reactions the classifier calls un-reactive are the slower
  ones (median log₁₀ *k* 1.00, against 1.32 for those it reports). The model
  also reports rate constants for 24 test reactions without a measured one:
  17 instant and 7 un-reactive.

Per regime (rate R² and MAE over all measured test reactions):

| regime | pairs | accuracy | rate R² (n) | rate MAE |
|---|---|---|---|---|
| unseen acyl chloride | 96 | 0.84 | 0.88 (28) | 0.22 |
| unseen amine | 82 | 0.84 | 0.78 (38) | 0.27 |
| both unseen | 8 | 0.63 | 0.87 (3) | 0.33 |

Two reactions fall outside ±2σ in the residuals plot (`residuals_test.png`),
both with amine 21 (2-(2-methoxyethoxy)aniline): rxn_19_21 and rxn_22_21.
Both are under-predicted, by factors of about 11 and 5. Amine 21 is
under-predicted in 13 of its 19 measured reactions (mean residual +0.18 log
units).

**Excluded substrate scope** (SI Section 7). 30 of the 41 excluded substrates
have a class label; the 11 insoluble ones are not scored. Accuracy is 0.93
(28 of 30), with no substrate wrongly called reactive.

| note | substrates | correct |
|---|---|---|
| too fast (reactive) | 18 | 16 |
| too slow | 4 | 4 |
| non-reactive | 3 | 3 |
| multiple products | 5 | 5 |

Each scope amine is paired with acyl chloride 1, and each scope acyl chloride
with amine 3.

**Drug-like amines (11 pairs, all reactive).** 9 of 11 are called reactive.
The two pairs with a measured rate constant have MAE 0.10 log units.

**All three pooled (227 calls).** Accuracy 0.85, precision 0.93, recall 0.76.

| regime | calls | accuracy |
|---|---|---|
| unseen acyl chloride | 96 | 0.84 |
| unseen amine | 122 | 0.86 |
| both unseen | 9 | 0.67 |

Figures in `results/holdout/figures/`:
- `confusion_test_scope_drug.png`: the three sets side by side
- `confusion_*held-out-scope.png`: pooled, by regime and by novelty
- `parity_test.png`, `parity_train_cv_test.png`: every measured reaction
  (the regression model alone); cross-validated points are
  leave-one-reactant-out
- `parity_test_reactive-only.png`, `parity_train_cv_test_reactive-only.png`:
  only the reactions the classifier calls reactive (the two-stage model's
  output, i.e. the agreement of the two models)
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
- class counts for all data, the training pool and each held-out regime:
  measurable, instant (reactive, not measurable), un-reactive, and
  undetermined (counted, not modelled);
- the distribution of measured log₁₀ *k*: 363 values from −0.03 to 3.41,
  mean 1.50 ± 0.70;
- a map of the class of every acyl chloride × amine pair.

## 4. Deployed model (`model/`)

The model in `model/` is trained on **all 1044 modelled reactions**. Its
descriptors are computed from SMILES by `prism_ml.descriptors`, the same
pipeline that `scripts/predict.py` runs on an unseen molecule.

The stored descriptor table (`PRISM_substrate_features.xlsx`, used for
sections 1–3) was computed on the DFT-optimised structures in `data/xyz`
(recomputing on them reproduces it; `data/README.md`). The SMILES pipeline
uses a single MMFF conformer instead, and for the conformation-sensitive acyl
chloride descriptors the two disagree. For example, SMILES-computed C=O carbon
charge correlates only r = 0.28 with the stored value, so a model trained on
the table would not transfer to unseen molecules.

Before the final fit, the same pipeline is validated on the held-out split:

| | accuracy | precision | recall | rate R² | rate MAE |
|---|---|---|---|---|---|
| SMILES-descriptor pipeline, held-out test | 0.88 | 0.86 | 0.89 | 0.81 | 0.24 |

These numbers are also stored in `model/metadata.json`.

## Reproducibility

- The histogram learners (HistGB, LightGBM, run deterministically) give
  identical scores on every machine tried.
- Exact-split learners (exact gradient boosting, AdaBoost) can differ in the
  third decimal between CPU types. This includes the selected model's
  classifier.
- The graph networks train on a GPU. They reproduce exactly on the same GPU
  (NVIDIA A30), but can differ by about 0.015 in joint score on other
  hardware.
