# PRISM HTE Rate Classification and Bias Correction Models

This directory contains the workflow for building machine learning models to classify and correct PRISM HTE (High-Throughput Experimentation) rate measurements for amide coupling reactions.

## Overview

The workflow performs two types of analysis:

1. **Bias Correction**: Classify reactions as measurable vs. fast unmeasurable, and predict bias magnitude to correct raw PRISM HTE rates
2. **HTE Rate Prediction**: Predict corrected PRISM HTE rates directly from molecular features

## Directory Structure

```
build_class_and_bias_models/
├── run_analysis.py          # Main analysis script
├── src/                     # Source modules
│   ├── data_processing.py   # Data loading and feature selection
│   ├── model_building.py    # Model training with hyperparameter optimization
│   ├── model_evaluation.py  # Model validation and testing
│   ├── visualization.py     # Plotting and visualization
│   └── load_models.py       # Model inference utilities
├── models/                  # Trained models (generated)
├── results/                 # Analysis results and reports (generated)
└── plots/                   # Visualizations (generated)
```

Data files are located in the parent `PRISM/data/` directory:
- `data/rates/` - PRISM HTE rate measurements (raw and corrected)
- `data/features/` - Molecular descriptors for acyl chlorides and amines

## Current Models

The current models in `models/` were obtained using the following configuration:

### Configuration Used
```python
config = {
    'analysis_type': 'bias_correction',
    'target_col': 'bias',
    'single_run': True,              # Parameter sweep mode
    'mode': 'in_all',                 # Combined feature selection
    'feature_selection_mode': 'selected',
    'specific_features':['amine_class_1_mixture', 'acyl_class_aromatic', 'acyl_Charges_secondary_1', 'amine_Charges_secondary_1', 'acyl_pka_aHs_x_has_acidic_H', 'amine_pka_basic', 'acyl_BV_secondary_2', 'amine_BV_secondary_avg'],
    'hyperparameter_optimization': True,
    'run_scrambling_test': True,
    'n_scrambling_trials': 10,
    'save_plots': True,
    'save_models': True
}
```

### Results Generated

Three parameter sweep analyses were performed:
- `results/reports_each_reduced/` - Sequential feature selection from each descriptor set (reduced)
- `results/reports_each_all/` - Sequential feature selection from each descriptor set (all features)
- `results/reports_combined_in_all/` - Combined feature selection from all descriptors

The best models are saved with suffix patterns like `_each_8_1_optuna` indicating:
- Feature selection mode (`each` or `in_all`)
- Number of features (e.g., `8`)
- Hyperparameter optimization method (`optuna`)

## Running the Analysis

### Basic Execution

```bash
cd build_class_and_bias_models
python run_analysis.py
```

### Customizing the Analysis

Edit the `config` dictionary in `run_analysis.py` (lines 465-492):

#### Analysis Type
```python
'analysis_type': 'bias_correction'  # or 'hte_prediction'
'target_col': 'HTE_lnk_corrected'   # Target for prediction
```

#### Run Mode
```python
'single_run': True   # Single configuration
'single_run': False  # Parameter sweep (tests all feature counts)
'mode': 'each'       # Select features from each descriptor set separately
'mode': 'in_all'     # Select features from combined descriptor set
'n_features': 8      # Number of features (ignored in sweep mode)
```

#### Feature Selection
```python
'feature_selection_mode': 'sequential'   # Forward sequential selection
'feature_selection_mode': 'correlation'  # Correlation-based selection
'feature_selection_mode': 'selected'     # Use specific pre-defined features
```

#### Model Training
```python
'hyperparameter_optimization': True   # Use Optuna for hyperparameter tuning
'hyperparameter_optimization': False  # Use default hyperparameters
```

## Using Trained Models

```python
from src.load_models import BiasPredictor

# Load trained models
predictor = BiasPredictor(
    classifier_suffix="_each_8_1_optuna",
    regressor_suffix="_each_8_1_optuna"
)

# Make predictions
feature_data = {'acyl_pka_aHs_x_has_acidic_H': 1.0, ...}
predicted_class = predictor.predict_rate_class(feature_data)  # 0=measurable, 1=fast
predicted_bias = predictor.predict_bias(feature_data)

# Batch predictions from CSV
from src.load_models import load_and_predict_batch
results = load_and_predict_batch(
    '../data/new_rates/combined_features_hte_rates_drug_scope.csv',
    classifier_suffix='_each_8_1_optuna',
    regressor_suffix='_each_8_1_optuna',
    save_results=True
)
```

## Output Files

### Models (`models/`)
- `best_classifier_*.pkl` - Classification model (measurable vs. fast)
- `best_regressor_*.pkl` - Regression model (bias magnitude or rate prediction)
- `scaler_class_*.pkl` - Feature scaler for classifier
- `scaler_reg_*.pkl` - Feature scaler for regressor
- `features_*.pkl` - Selected feature names

### Results (`results/`)
- `parameter_sweep_results.csv` - Summary of all tested configurations
- `*_evaluation_report_*.txt` - Detailed model performance metrics
- `corrected_hte_rates_*.csv` - PRISM HTE rates with bias corrections applied

### Plots (`plots/`)
- `model_performance_*.png` - Model comparison plots
- `parity_plot_*.png` - Predicted vs. actual plots
- `classification_performance_*.png` - Confusion matrices and ROC curves

## Dependencies

```bash
# Core scientific computing
pandas numpy matplotlib seaborn scipy

# Machine learning
scikit-learn xgboost lightgbm catboost optuna
```
