# GNN Model Building for Amide Reaction PRISM Rate Prediction

This directory contains the workflow for building and optimizing a Graph Neural Network (GNN) model to predict amide coupling PRISM HTE rates using molecular features and graph representations.

## Overview

The workflow consists of three main stages:
1. **Data preparation**: Converting raw CSV data to structured splits with molecular features
2. **Data verification**: Inspecting H5 data structure and contents
3. **Model optimization**: Hyperparameter tuning and final model training

## Workflow Scripts

### 1. Data Preparation: `assign_splits.py`

Prepares training data by creating train/val/test splits from raw CSV files with molecular descriptors.

**Purpose:**
- Creates train/validation/test splits from CSV data containing 'test splits' column (TRAIN/TEST1/TEST2)
- Adds molecular features (acyl chlorides and amines descriptors) from MORFEUS and xTB calculations
- Creates reaction keys in format `rxn_ACYL_AMINE` paired with target PRISM HTE rate values
- Includes control reaction data for each acyl chloride
- Outputs structured JSON file with all features embedded

**Key Features:**
- Automatically loads molecular descriptors from `data/features/` directory
- Handles categorical features (amine/acyl classes) by one-hot encoding
- Processes pKa features with binary flags for acidic protons
- Creates validation split (default 10%) from training data
- Saves combined JSON with all molecular features at reaction level

**Usage:**
```bash
cd utils
python assign_splits.py /path/to/input.csv \
    --output-file ../data/hte-all-corrected_splits_train_val_tests_lnk.json \
    --val-ratio 0.1 \
    --y-column corrected_HTE_rate_all \
    --add-features
```

**Outputs:**
- JSON file with train/val/test splits containing:
  - Reaction keys (`rxn_ACYL_AMINE`)
  - Target PRISM HTE rate values (`hte_lnk`)
  - Control rates
  - Molecular features (acyl and amine descriptors)
- Optional: Separate CSV with features for inspection

**Input Requirements:**
- CSV file with columns: `acyl_chlorides`, `amines`, `corrected_HTE_rate_all`, `test splits`, `Controls`
- Feature files in `data/features/`:
  - `descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv`
  - `descriptors_acyl_chlorides.csv`
  - `descriptors_amines_morfeus_addn_w_xtb.csv`
  - `descriptors_amines.csv`

---

### 2. Data Verification: `read_h5.py`

Analyzes and summarizes H5 data files to verify structure and contents.

**Purpose:**
- Inspects H5 files in the `splits/` directory
- Reports data shapes, types, and sample values for each dataset
- Compares structure across train/val/test splits
- Generates human-readable reports for data validation

**Key Features:**
- Recursively analyzes H5 group structure
- Shows dataset shapes, dtypes, and sample values
- Cross-split comparison to ensure consistency
- Saves text report for documentation

**Usage:**
```bash
cd utils
python read_h5.py
```

**Outputs:**
- Terminal output with detailed H5 structure
- `splits/h5_structure_report.txt` - saved report file

**What to verify:**
- All splits (train/val/test) contain the same dataset keys
- Dataset shapes are consistent with molecular representations:
  - `amine_a`, `acid_a`, `int_a` - adjacency matrices
  - `amine_q`, `acid_q`, `int_q` - atomic charges
  - `amine_aim`, `acid_aim`, `int_aim` - AIM features (256-dim)
  - Additional features from molecular descriptors
  - `rate` - target PRISM HTE rate values

---

### 3. Hyperparameter Optimization: `hyperparameter_optimization.py`

Main workflow for optimizing GNN model hyperparameters and training the final model.

**Purpose:**
- Uses Optuna to find optimal hyperparameters for AmidePredictor model
- Trains final model with best parameters
- Generates comprehensive evaluation metrics and visualizations
- Saves optimized model and predictions

**Optimized Hyperparameters:**
- `learning_rate`: 1e-6 to 1e-2 (log scale)
- `n_graph_layers`: 2 to 5 layers
- `n_output_layers`: 1 to 3 layers  
- `samples_per_epoch`: 50 to 300 (step: 50)

**Model Architecture:**
- **AmidePredictor** GNN model with:
  - Graph convolution layers processing molecular graphs
  - Integration of adjacency matrices, atomic charges, and AIM features
  - Additional molecular descriptors as auxiliary features
  - Control reaction rate normalization
  - Output: predicted PRISM HTE ln(k) values

**Input Features:**
- Molecular graph features:
  - `amine_a`, `amine_q`, `amine_aim` - amine molecular features
  - `acid_a`, `acid_q`, `acid_aim` - acid molecular features  
  - `int_a`, `int_q`, `int_aim` - intermediate features
- Additional features: Dynamically loaded from JSON (MORFEUS, xTB descriptors)
- Control rates for normalization

**Usage:**
```bash
python hyperparameter_optimization.py
```

**Configuration:**
Modify in `__main__` section:
```python
# Number of Optuna trials
study, best_trial, run_dir = run_optimization(n_trials=30)

# Number of epochs for final training
final_model, final_metrics = train_best_model(best_trial.params, run_dir, n_epochs=100)
```

Modify data file path (lines 113, 336):
```python
json_file = "data/hte-all-corrected_splits_train_val_tests_lnk.json"
```

**Outputs:**

Results are saved to timestamped directory: `optimization_runs/run_YYYYMMDD_HHMMSS/`

Directory structure:
```
optimization_runs/run_YYYYMMDD_HHMMSS/
├── plots/
│   ├── optimization_history.png       # Optuna trial history
│   ├── hyperparameter_importance.png  # Parameter importance plot
│   ├── final_training_curves.png      # Training/validation loss curves
│   └── optimized_parity_plot.png      # Predicted vs actual PRISM HTE rates
├── predictions/
│   ├── train_predictions.csv          # Training set predictions
│   ├── val_predictions.csv            # Validation set predictions
│   ├── test_predictions.csv           # Test set predictions
│   └── all_predictions.csv            # Combined predictions
├── model/
│   └── optimized_final.jpt            # Saved TorchScript model
├── best_params.json                   # Best hyperparameters found
├── all_trials.json                    # All Optuna trials
├── final_metrics.json                 # Final model metrics (R², MAE)
└── run_summary.json                   # Complete run summary
```

**Performance Metrics:**
- R² (coefficient of determination)
- MAE (mean absolute error)
- Reported for train/validation/test splits

---

## Complete Workflow Example

### Step 1: Prepare Data Splits
```bash
cd utils
python assign_splits.py \
    ../../data/rates/hte-all-corrected.csv \
    --output-file ../data/hte-all-corrected_splits_train_val_tests_lnk.json \
    --val-ratio 0.1 \
    --add-features
```

### Step 2: Convert to H5 Format
(Using NGDataset - see `utils/ngdataset.py`)

### Step 3: Verify H5 Structure
```bash
cd utils
python read_h5.py
```

### Step 4: Run Hyperparameter Optimization
```bash
python hyperparameter_optimization.py
```

### Step 5: Review Results
Check the timestamped directory in `optimization_runs/` for:
- Model performance plots
- Predictions CSV files
- Saved model file (.jpt)
- Best hyperparameters

---

## Dependencies

Key Python packages required:
- `torch` - PyTorch for model training
- `optuna` - Hyperparameter optimization
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Train/test splitting
- `h5py` - HDF5 file handling

Hardware:
- Automatically detects and uses best available device (CUDA/MPS/CPU)
- GPU recommended for faster training

---

## Notes

**Feature Selection:**
- By default, uses all molecular features: adjacency (`_a`), charges (`_q`), and AIM features (`_aim`)
- Alternative: Use only AIM features by setting `include_all_molecular=False` in `get_feature_columns()`
- Additional molecular descriptors are dynamically loaded from JSON

**Model Input Dimension:**
- Current: 513 (256 AIM + 256 AIM + 1 charge = 513 per molecule)
- AIM-only: 256 (only AIM features)

**Training Strategy:**
- Uses AdamW optimizer with AMSGrad
- Gradient clipping at 0.05 to prevent exploding gradients
- Median pruner for efficient hyperparameter search
- Early stopping via Optuna's pruning mechanism

**Random Seeds:**
- Default random state: 42 (for reproducibility)
- Set in `assign_splits.py` for consistent train/val splits

---

## Final Results

The optimized model and results used in the final publication are saved in:
```
optimization_runs/run_YYYYMMDD_HHMMSS/
```

Refer to `run_summary.json` in the output directory for complete details on:
- Best hyperparameters used
- Final model performance (R², MAE on train/val/test)
- Training configuration
- All generated files

