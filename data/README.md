# Data dictionary

Everything the workflow reads is in this directory. The tables are Excel
workbooks in `datasheets/`. Reactions are keyed by `acyl_label` and
`amine_label`, the reactant numbers used throughout the manuscript and SI. The
reaction name `rxn_AA_BB` combines them as acyl chloride AA + amine BB.

| file | contents |
|---|---|
| `datasheets/PRISM_rates.xlsx` | reactions with a PRISM rate constant, one row per reaction (used by the models) |
| `datasheets/PRISM_rates_unmeasurable.xlsx` | reactions without a rate constant, with the reason |
| `datasheets/PRISM_rates_all.xlsx` | every row before replicates are combined, and the replicate values side by side |
| `datasheets/unnamed_PRISM_rates.xlsx`, `datasheets/unnamed_PRISM_rates_unmeasurable.xlsx` | the same tables as exported by the image processing, identified by reactant name only |
| `datasheets/PRISM_rates_drug_scope.xlsx` | drug-like amines (PRISM and NMR) |
| `datasheets/PRISM_substrate_scope.xlsx` | names, SMILES and classes of all substrates, including the excluded scope |
| `datasheets/PRISM_substrate_features.xlsx` | descriptors used for the reported results |
| `datasheets/PRISM_substrate_features_smiles.xlsx` | descriptors recomputed from SMILES (deployed model) and excluded-scope descriptors |
| `xyz/acyls/acid_NN.xyz`, `xyz/amines/amine_NN.xyz` | 3D structures of the reactants (46 acyl chlorides, 55 amines) |
| `aimnet2_atoms.npz` | AIMNet2 atom features for the graph-network benchmark (see `aimnet2/README.md`) |

## How the rate tables are built

`scripts/build_rate_tables.py` builds the three `PRISM_rates*` workbooks from
the two `unnamed_*` exports. Each acyl chloride and amine gets its label from
`PRISM_substrate_scope.xlsx` (`acyl_label` / `amine_label`, matched by name),
and each reaction its name `rxn_AA_BB`.

4-Fluorobenzoyl chloride (acyl chloride 1) was run as three replicate batches,
1A, 2B and 3C (names ending `-1A`, `-2B`, `-3C`). Their reactions are combined:
the rate constant is the geometric mean of the usable (k > 0) replicates, and a
reaction with at least one usable replicate is measurable. Reactions with no
usable replicate take the replicates' common class. `PRISM_rates_all.xlsx`
keeps every individual value.

## How the two rate workbooks become the modeling table

Each acyl chloride / amine pair gets one class (`prism_ml.data.load_reactions`):

- **measurable**: a positive `k_corrected` in `PRISM_rates.xlsx`. This takes
  priority if the pair also appears in the unmeasurable workbook.
- **instant**: `rate_class = instant` in `PRISM_rates_unmeasurable.xlsx` (too
  fast to resolve). The 4 `ambiguous` pairs, and fitted pairs with a
  non-positive `k_corrected`, are also counted as instant.
- **slow**: `rate_class = slow` (not distinguishable from hydrolysis).

For classification, *reactive* = measurable or instant, and *un-reactive* =
slow. Regression uses log₁₀ `k_corrected` of the measurable pairs.

## PRISM_rates.xlsx

Sheet `PRISM_perwell_validated_dataset`: 365 reactions whose wells passed the
per-well hydrolysis control (1.5×), one row per reaction.

| column | meaning |
|---|---|
| `reaction` | reaction name, `rxn_AA_BB` |
| `acyl_label`, `amine_label` | reactant numbers |
| `group` | plate group the reaction was run in (`group1`–`group3`) |
| `acyl_chloride`, `amine` | reactant names |
| `batch` | for 4-fluorobenzoyl chloride, the replicate batches combined in this row (e.g. `1A,2B,3C`); blank otherwise |
| `n_replicates_combined` | number of replicate rate constants in the geometric mean (1 for other reactions) |
| `n_wells_used` | number of dilution wells that passed the control and were used in the fit (2–5) |
| `control_source` | where the hydrolysis control came from: `own_plate`, or `same_acyl_other_plate` (the same acyl chloride's control on another plate) |
| `k_corrected` | PRISM rate constant from the fit of 1/[A]₀ against t½ over the passing wells; for replicate reactions, the geometric mean of the replicates |
| `r2` | R² of that fit (blank when only 2 wells were used, and for combined replicates) |
| `confidence_tier` | set by the number of wells used: `high_confidence` (4–5 wells), `medium_confidence` (3), `low_confidence_order_of_magnitude` (2); for combined replicates, that of the replicate fitted on the most wells |
| `hydrolysis_margin_used` | per-well control threshold: a well passes when the blank's inflection time exceeds the reaction's by this factor (1.5) |

## PRISM_rates_unmeasurable.xlsx

Sheet `PRISM_excluded_binned_1p5x`: 720 reactions for which no rate constant
could be fitted, one row per reaction (replicate reactions with a usable rate
constant appear only in `PRISM_rates.xlsx`).

| column | meaning |
|---|---|
| `reaction`, `acyl_label`, `amine_label`, `group`, `acyl_chloride`, `batch`, `amine`, `control_source`, `hydrolysis_margin_used` | as in `PRISM_rates.xlsx` |
| `n_raw_detections` | number of dilution wells in which a color change was detected |
| `n_beats_control` | number of those wells that passed the hydrolysis control |
| `time_at_most_dilute_tested_min` | inflection time (min) at the most dilute concentration tested |
| `conc_at_most_dilute_tested_mM` | that concentration (mM) |
| `rate_class` | `slow`, `instant` or `ambiguous` |
| `bin` | reason, matching `rate_class`: `too_slow_indistinguishable_from_hydrolysis` (slow, 605), `too_fast_even_when_diluted` (instant, 111), `borderline_kept_out_by_fit_quality` (ambiguous, 4) |
| `note` | the reason in words, e.g. "Even at the most dilute concentration tested (0.391 mM) the color change was complete by the first reading" |

## PRISM_rates_all.xlsx

| sheet | contents |
|---|---|
| `rates` | every row of `unnamed_PRISM_rates.xlsx`, labelled, with `batch` (`1A`, `2B`, `3C` or blank) |
| `unmeasurable` | every row of `unnamed_PRISM_rates_unmeasurable.xlsx`, labelled, with `batch` and `rate_class` |
| `replicates` | one row per 4-fluorobenzoyl chloride reaction: `k_1A`, `k_2B`, `k_3C` and `class_1A`, `class_2B`, `class_3C` for each batch, `n_usable` (replicates with k > 0), `k_geometric_mean`, and the combined `class` |

## PRISM_rates_drug_scope.xlsx

Sheet `Sheet1`: 11 reactions of three drug-like amines (54–56).

| column | meaning |
|---|---|
| `reaction`, `acyl_label`, `amine_label` | as above |
| `k_corrected` | PRISM rate constant; 0 where the reaction was too fast to measure |
| `k_NMR`, `logk_NMR` | NMR rate constant and its log₁₀, where measured |
| `Notes` | comments on the NMR measurement |
| `rate_class` | `measurable` or `instant` |

## PRISM_substrate_scope.xlsx

| sheet | contents |
|---|---|
| `acyls` | acyl chlorides: `name` (structure file), `acyl_label`, `acyl_chloride` (name), `smiles`, `split_type`, `split_type_cluster` and `cluster_assignment` (from an earlier clustering-based split, not used here), `class` |
| `amine` | amines, same columns (`amine_label`, `amine`) |
| `drug` | the three drug-like amines (54–56) |
| `excluded` | the excluded substrate scope (SI Section 7): `name` (e.g. `57_amine`), `molecule`, `CAS`, `smiles`, `class`, `substrate` (amine or acyl chloride), `note` (reason for exclusion: too fast, too slow, non-reactive, multiple products, insoluble) |

`class` gives the chemical class used for the train/test split. For amines,
the prefix 1 or 2 means primary or secondary.

The `split_type` columns in these workbooks come from an earlier split and
are not used. The held-out split is computed by `prism_ml.splits.build_split`
(`docs/RESULTS.md`).

## PRISM_substrate_features.xlsx and PRISM_substrate_features_smiles.xlsx

Sheets `acyl` (46 rows) and `amine` (55 rows) in both workbooks, one row per
reactant (`id` = reactant number):

- `PRISM_substrate_features.xlsx`: descriptors used for all reported results.
- `PRISM_substrate_features_smiles.xlsx`: the same descriptors recomputed
  from SMILES by `scripts/compute_descriptors.py`. The deployed model in
  `model/` uses these.

The SMILES workbook has two extra sheets, `excluded_acyl` (3 rows) and
`excluded_amine` (38 rows). These hold descriptors for the excluded substrate
scope, with `id` matching `name` on the `excluded` sheet of
`PRISM_substrate_scope.xlsx`.

Identifier columns: `id`, `name`, `smiles`, `split_type`, `class`, `source`
(how the row was produced). The excluded sheets also have `idx_*` and `el_*`,
the atom indices and elements of the atoms the descriptors refer to.

Descriptors are computed at a **main atom** and at its **secondary atoms**:

- Acyl chlorides: the main atom is the carbonyl carbon; secondary atom 1 is
  the chlorine and secondary atom 2 is the α-carbon.
- Amines: the main atom is the reacting nitrogen; secondary atom 1 is its
  first heavy-atom substituent.

Column suffixes: `_main_atom` or `_main_1` for the main atom, `_secondary_N`
for secondary atom N, `_avg` for the average over the secondary atoms, and
`_Hs_avg` for the average over the α-hydrogens (acyl chlorides only; 0 when
there are none).

| columns | descriptor | tool |
|---|---|---|
| `IP`, `EA` | ionization potential and electron affinity (eV) | xTB |
| `HOMO`, `LUMO` | frontier orbital energies (Hartree) | xTB |
| `global_E`, `global_N`, `hardness` | global electrophilicity, nucleophilicity and hardness | xTB |
| `Charges_*` | atomic partial charges | xTB |
| `fukui_E_*`, `fukui_N_*` | electrophilic and nucleophilic Fukui indices | xTB |
| `BV_*` | percent buried volume | Morfeus |
| `L_*`, `B1_*`, `B5_*` | Sterimol parameters along the main–secondary atom bond (Å) | Morfeus |
| `SA`, `SV` | solvent-accessible surface area (Å²) and volume (Å³) | Morfeus |
| `PYR` | pyramidalization of the main atom | Morfeus |
| `DISP_mol`, `DISP_atom` | dispersion descriptor P_int of the molecule and of the main atom | Morfeus |
| `num_Hs` | number of α-hydrogens (acyl chlorides) | — |
| `pka_aHs`, `pka_lowest` (acyl), `pka_basic` (amine) | pKa values (α-hydrogens, most acidic site, conjugate acid of the amine) | stored workbook only |
| `FCS_*`, `FREQS_*` | force constants and frequencies of the main–secondary bonds | stored workbook only |

The columns used by the selected model (Tier 2, reduced; 52 descriptors) are
listed in `model/metadata.json`. The descriptor tiers are defined in
`src/prism_ml/config.py`. pKa, force-constant and frequency columns are not
computed from SMILES; they are blank in the SMILES workbook and are not used
by the selected model.
