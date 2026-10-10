# PRISM

Code and data for PRISM amide-coupling kinetics, from plate images to the
machine-learning models.

| folder | step | output |
|---|---|---|
| [`1_image_analysis`](1_image_analysis/Amide_Code) | plate images → inflection time of every well | one workbook per plate (`times_and_concs.xlsx`) |
| [`2_rate_fitting`](2_rate_fitting) | per-well hydrolysis control (1.5×), rate constants, reaction classes | `PRISM_rates*.xlsx` in `3_ml_model/data/datasheets` |
| [`3_ml_model`](3_ml_model) | reactivity classifier and rate-constant regressor; held-out evaluation; NMR and Hammett comparisons | models, results, `docs/RESULTS.md` |

Each folder has its own README. To rebuild the rate tables and rerun the
models:

```bash
cd 2_rate_fitting
python scripts/prism_perwell.py --plates inputs/original_plate_files_99.zip --outdir outputs
cd ../3_ml_model
./run_all.sh
```

The plate images are not included. The 99 plate workbooks they produced are
in `2_rate_fitting/inputs`.
