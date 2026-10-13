# Rate fitting

Turns the PRISM plate data (inflection times from `1_image_analysis`) into rate
constants and reaction classes, and writes the rate tables used by
`3_ml_model`.

## Workflow

```bash
cd 2_rate_fitting
python scripts/prism_perwell.py --plates inputs/original_plate_files_99.zip --outdir outputs
python scripts/nmr_vs_prism.py --nmr inputs/PRISM_nmr_rates.xlsx --key inputs/label_key_acyl_amine.csv \
                               --prism-dir outputs --outdir validation/margin_vs_nmr
```

`prism_perwell.py` does the following.

1. **Reads the plates.** There are 99 workbooks, one per group and acyl
   chloride (33 acyl chlorides, counting the three 4-fluorobenzoyl chloride
   batches, × groups 1–3). Each amine column gives the inflection time at 8
   dilutions (stock concentrations 50 to 0.391 mM; the initial concentration in
   the well, [A]₀, is a quarter of the stock). The `blank` column (no amine) is the
   hydrolysis control.
2. **Applies a per-well control.** A well is kept if the control is at least
   1.5× slower than the amine at the same concentration
   (`control time / amine time ≥ margin`). Other wells are dropped one at a
   time; the pair is not discarded.
3. **Fits the rate constant.** k is the slope of 1/[A]₀ against inflection
   time over the kept wells. It needs at least 2 wells; 4 or more are high
   confidence, 3 medium and 2 low.
4. **Bins pairs without a rate constant:** too fast, too slow, borderline (no
   slope) or negative fit.
5. **Writes the ML rate tables** at the 1.5× margin, to
   `../3_ml_model/data/datasheets` (or `--datasheets`):
   - each reactant is numbered from `PRISM_substrate_scope.xlsx`;
   - the 4-fluorobenzoyl chloride batches are combined;
   - each pair gets a class and a note saying why.

`nmr_vs_prism.py` pairs PRISM with NMR rate constants at each margin.

## Outputs

| file | contents |
|---|---|
| `outputs/inflection_times_long.csv` | every inflection time in the plates, controls included |
| `outputs/well_level_kept_excluded.csv` | every amine well: amine and control times, ratio, kept or why excluded, per margin |
| `outputs/pair_level_fit_<margin>.csv` | pairs with a rate constant |
| `outputs/pair_level_excluded_<margin>.csv` | pairs without one, with the bin and its evidence |
| `outputs/unnamed_PRISM_rates*.xlsx` | the 1.5× pairs in the ML workbook layout, by reactant name |
| `3_ml_model/data/datasheets/PRISM_rates.xlsx` | reactions with a rate constant |
| `3_ml_model/data/datasheets/PRISM_rates_unmeasurable.xlsx` | reactions without one: `rate_class` and `note` |
| `3_ml_model/data/datasheets/PRISM_rates_all.xlsx` | every plate row, labelled, and the replicate batches side by side |
| `validation/margin_vs_nmr/` | `paired_nmr_prism_rates.csv`, `agreement_by_margin.csv`, `margin_vs_nmr.png` |

## Classes in the rate tables

| class | meaning |
|---|---|
| measurable | rate constant fitted (k > 0) |
| instant | reactive but not measurable; the note says why (too fast, no slope, negative slope, or the control review) |
| slow | not distinguishable from hydrolysis (un-reactive) |
| undetermined | too few wells to compare with the control; not modelled |

At 1.5×: 363 measurable, 138 instant, 543 slow, 41 undetermined (1085
reactions).

## Notes and caveats

- **Borrowed controls.** Only the group1 plate of each acyl chloride has a
  blank. The group2 and group3 plates use that blank, matched by
  concentration. These pairs are flagged as indeterminate but not removed.
  Across the three 4-fluorobenzoyl chloride runs, the blank varies by up to
  about 1.5× at the same concentration.
- **Missing times.** A blank time in a plate workbook can mean that no colour
  change was seen, or that the well was outside the range picked by hand in
  the inflection finder (`1_image_analysis`). The filter cannot tell these
  apart; both count as "no reading".
- **Ordinate.** The stored `1/[A]` column is 2/[A]₀ (the value at 50%
  conversion), so k is fitted on half of it: 1/[A]₀ = k t₁/₂. The `conc_mM`
  column of the outputs is the stock concentration, not [A]₀.
- **Too fast.** The inflection is ≤ 2 min at the most dilute concentration
  with a reading. Four amines that react too fast to detect are also binned
  too fast when they have no rate: diisopropylamine, 2-methylpiperidine,
  morpholine and 3-aminopentane.
- **Negative fits.** Two pairs (2-furoyl chloride + p-anisidine;
  4-tert-butylbenzoyl chloride + 3-aminopentane) beat the control, but their
  slope is negative, with times at the reading limit. They are instant, with
  no rate constant.
- **Control review.** The too-slow bin does not separate wells that failed the
  control from wells that could not be compared with it (no control reading,
  or an amine time of 0, where the ratio is undefined). 62 too-slow pairs had
  fewer than two comparable wells in every replicate.
  - 21 (12 own-plate, 9 borrowed controls) are instant. Each has at least two
    wells that beat the control and none that failed; this counts wells where
    the amine was done by the first reading while the control was not. These
    wells support the label but do not establish a 1.5× ratio.
  - The other 41 are undetermined.
  - Replicate pairs that fail the control in another batch stay slow
    (rxn_01_01, rxn_01_12, rxn_01_34).
- **Margin.** Agreement with NMR improves sharply at 1.4× and is flat from 1.4×
  to 1.5× (RMS about 0.43 log units); higher margins improve it further only
  by losing pairs (`margin_vs_nmr.png`). 1.5× is a conservative choice within
  that range, not an optimum. The margin was examined on the same NMR pairs
  used to compare PRISM with NMR, so that comparison is not an independent
  check.
- **PRISM vs NMR offset.** PRISM rate constants are about 2× the NMR values at
  every margin (median ratio 1.8–2.1). The margin reduces the scatter, not
  this offset.
- **NMR pairs.** Only reactions with per-well plate data can be paired: 47 at
  1.5×. In the figure, the "≥ 3 wells" curve counts pairs whose every
  replicate plate kept at least 3 wells.
