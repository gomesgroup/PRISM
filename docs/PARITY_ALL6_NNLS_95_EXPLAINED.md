# Parity plot (ln k): CheMeleon → big-FFN Ensemble-5

This document fully explains how the figure `plots/parity_all6_nnls_95.png` is created: the dataset, train/test split, model configuration, prediction pipeline, plotting inputs, styling, and how metrics are computed. All relevant file paths are listed explicitly.

---

## Figure artifacts

- Primary figure (TRAIN + TEST):
  - `plots/parity_all6_nnls_95.png`
- Optional figure (TRAIN + VAL + TEST, if VAL present or synthesized):
  - `plots/parity_all6_nnls_95_with_val.png`

Plotting script:
- `src/plot_gpg_style_parity_all6.py`

---

## Dataset

- Source CSV (cleaned for finite ln k):
  - `data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv`
- Input columns used by the model:
  - `amine_smiles`, `acid_smiles`
- Target column (measured ln k):
  - `HTE_lnk`
- Split column:
  - A "split" column is normalized to `TRAIN`/`VAL`/`TEST` if present. In our combined plotting CSV (see below), typical counts are:
    - TRAIN: 958
    - TEST: 163
    - VAL: present only if supplied by the dataset; otherwise not present.

Note: A previous step removed non‑finite ln k rows to create the finite dataset above.

---

## Model and training

We use Chemprop with the CheMeleon foundation encoder and a larger FFN head ("big‑FFN").

- Baseline single‑model training output:
  - Directory: `runs/chemeleon_lnk_bigffn/`
  - Config (exact training parameters): `runs/chemeleon_lnk_bigffn/config.toml`
    - Key parameters (subset):
      - `from-foundation = CheMeleon`
      - `smiles-columns = [amine_smiles, acid_smiles]`
      - `target-columns = [HTE_lnk]`
      - `rxn-mode = REAC_DIFF`
      - `multi-hot-atom-featurizer-mode = V2`
      - `ffn-hidden-dim = 1024`, `ffn-num-layers = 3`, `batch-norm = true`, `dropout = 0.2`
      - `epochs = 100`, `warmup-epochs = 5`, `patience = 10`, `max-lr = 0.001`
      - `split = RANDOM`, `split-sizes = [0.8, 0.1, 0.1]`

- Ensemble‑5 (used for the figure):
  - Directory: `runs/chemeleon_lnk_bigffn_ens5/`
  - Member checkpoints (examples):
    - `runs/chemeleon_lnk_bigffn_ens5/model_0/best.pt`
    - `runs/chemeleon_lnk_bigffn_ens5/model_1/best.pt`
    - `runs/chemeleon_lnk_bigffn_ens5/model_2/best.pt`
    - `runs/chemeleon_lnk_bigffn_ens5/model_3/best.pt`
    - `runs/chemeleon_lnk_bigffn_ens5/model_4/best.pt`

- Optional: Ensemble‑10 DDP (for experiments):
  - Rerun directory: `runs/chemeleon_lnk_bigffn_ens10_ddp_rerun/`

---

## Generating predictions for plotting

We compute predictions across the entire dataset (TRAIN/VAL/TEST rows) using the ensemble‑5 checkpoints, then compute an ensemble average and an ensemble standard deviation per row to capture uncertainty.

1) Chemprop prediction over the full CSV with all five models:

- Averaged predictions (CSV):
  - `results/bigffn_ens5_all_predictions.csv`
- Individual model predictions (CSV):
  - `results/bigffn_ens5_all_predictions_individual.csv` with columns `HTE_lnk_model_0..4`

2) Construct the plotting input CSV by joining predictions and ground truth and computing uncertainty:

- Combined plotting input (used by the script):
  - `results/ensemble_all6_predictions_full.csv`
  - Columns:
    - `y_true`: from dataset `HTE_lnk`
    - `pred_nnls`: averaged prediction from the ensemble (`results/bigffn_ens5_all_predictions.csv`)
    - `std_unweighted`: per‑row ensemble standard deviation across individual model outputs `HTE_lnk_model_0..4`
    - `split`: TRAIN/VAL/TEST (normalized if the dataset provides a split column)

- Optional (when VAL split is available or explicitly synthesized for visualization):
  - `results/ensemble_all6_predictions_full_with_val.csv`

---

## Plot configuration and styling

Script: `src/plot_gpg_style_parity_all6.py`

- Inputs expected by the script:
  - `results/ensemble_all6_predictions_full.csv`
    - Required columns: `y_true`, `pred_nnls`, `std_weighted_nnls` or `std_unweighted`, `split`
- Outputs produced:
  - `plots/parity_all6_nnls_95.png` (TRAIN/TEST; will include VAL if present in the CSV)
  - `plots/parity_all6_nnls_95_with_val.png` (explicit VAL rendering if the `_with_val.csv` is present)

- Style:
  - Font: Helvetica (fallbacks: Nimbus Sans L / Arial / DejaVu Sans)
  - Markers: filled squares (`s=6`)
  - Colors: TRAIN black `#000000`; TEST red `#d62728`; VAL gold `#FDB515` (CMU “Gold Thread”)
  - Uncertainty: `std_*` mapped to alpha in [0.2, 1.0]; lighter = higher uncertainty
  - Lines: black 1:1 diagonal; light gray (TEST‑only) linear fit line
  - Grid: subtle light gray, behind points (points use `zorder=3.0`)
  - Ticks: both x and y identical: `[-1, 0, 1, 2, 3, 4]` with limits expanded to include full tick range
  - Axis labels:
    - x: “measured ln k”
    - y: “predicted ln k (big‑FFN ens‑5)”

---

## Metric computation (legend contents)

Metrics are computed inside the plotting script from the exact points being plotted. No external cache is used.

- R²: 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
- MAE: mean(|y_true - y_pred|)
- Split‑specific: computed for `TRAIN`, `VAL`, and `TEST` subsets separately (if present)
- The TEST linear fit line is computed from TEST points only.

---

## Repro and update steps

1) Predict across the entire dataset using ensemble‑5 checkpoints (GPU 3 example):

```bash
# Average predictions and per-model predictions
eval "$(conda shell.zsh hook)" && conda activate chemprop-v2
CUDA_VISIBLE_DEVICES=3 chemprop predict \
  -i data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv \
  -s amine_smiles acid_smiles \
  --model-paths \
    runs/chemeleon_lnk_bigffn_ens5/model_0/best.pt \
    runs/chemeleon_lnk_bigffn_ens5/model_1/best.pt \
    runs/chemeleon_lnk_bigffn_ens5/model_2/best.pt \
    runs/chemeleon_lnk_bigffn_ens5/model_3/best.pt \
    runs/chemeleon_lnk_bigffn_ens5/model_4/best.pt \
  -o results/bigffn_ens5_all_predictions.csv \
  --accelerator gpu --devices 1
```

2) Build plotting CSV (join averaged predictions, compute ensemble std from individual predictions, attach `y_true` and `split`):

- Final combined CSV used by the plot:
  - `results/ensemble_all6_predictions_full.csv`

3) Generate figures:

```bash
python src/plot_gpg_style_parity_all6.py
# Produces: plots/parity_all6_nnls_95.png
# If results/ensemble_all6_predictions_full_with_val.csv exists:
# Produces: plots/parity_all6_nnls_95_with_val.png
```

---

## File path summary (all relevant artifacts)

- Dataset and splits
  - `data/rates/corrected_hte_rates_each_8_optuna_finite_lnk.csv`
- Training configuration and outputs
  - `runs/chemeleon_lnk_bigffn/config.toml`
  - `runs/chemeleon_lnk_bigffn_ens5/model_0/best.pt`
  - `runs/chemeleon_lnk_bigffn_ens5/model_1/best.pt`
  - `runs/chemeleon_lnk_bigffn_ens5/model_2/best.pt`
  - `runs/chemeleon_lnk_bigffn_ens5/model_3/best.pt`
  - `runs/chemeleon_lnk_bigffn_ens5/model_4/best.pt`
- Predictions
  - `results/bigffn_ens5_all_predictions.csv`
  - `results/bigffn_ens5_all_predictions_individual.csv`
- Plotting CSV(s)
  - `results/ensemble_all6_predictions_full.csv`
  - `results/ensemble_all6_predictions_full_with_val.csv` (optional)
- Plotting script and outputs
  - `src/plot_gpg_style_parity_all6.py`
  - `plots/parity_all6_nnls_95.png`
  - `plots/parity_all6_nnls_95_with_val.png` (optional)

---

## Color reference (VAL)

- CMU Secondary “Gold Thread” for VAL: `#FDB515`
  - Reference: https://www.cmu.edu/brand/brand-guidelines/visual-identity/colors.html
