# ML Measurable HTE Rate Analysis - Modular Version

This directory contains a modular refactoring of the original monolithic workflow script. The analysis supports three main types of studies: **rate classification** (measurable vs fast), **bias correction** for HTE rate measurements, and **HTE rate prediction** from molecular features.

## 🚀 Quick Start

```bash
# Navigate to the ml_measurable_hte_rates directory
cd ml_measurable_hte_rates

# Run analysis with default settings (currently configured for HTE prediction)
python run_analysis.py

# Load and use trained models for predictions
python -c "from src.load_models import example_usage; example_usage()"
```

## 📁 Module Structure

### Core Modules

1. **`data_processing.py`** - Data loading and feature processing
   - Load HTE data for bias correction or rate prediction analysis
   - Calculate bias metrics for slow/unreliable reactions
   - Load and merge molecular descriptors (acyl chlorides and amines)
   - Feature correlation analysis and sequential feature selection
   - Data preprocessing and cleaning

2. **`model_building.py`** - Machine learning model training
   - **Bias Correction**: Classification (measurable vs fast) + Regression (bias magnitude)
   - **HTE Prediction**: Regression models for predicting corrected HTE rates
   - Hyperparameter optimization with Optuna
   - Support for multiple ML algorithms (RF, XGB, LightGBM, CatBoost, Linear models)
   - Model saving and loading utilities

3. **`model_evaluation.py`** - Model validation and testing
   - Apply bias corrections to HTE rate data
   - Apply rate classification (measurable vs unmeasurable)
   - Validate corrections and predictions
   - Feature scrambling and Y-scrambling validation tests
   - Comprehensive parity plot generation
   - Cross-validation analysis and reporting

4. **`visualization.py`** - Plotting and visual analysis
   - Model performance comparison plots
   - Bias correction visualizations
   - Classification performance plots (confusion matrices, ROC curves)
   - Parity plots for regression models
   - Feature correlation heatmaps
   - Scrambling test result visualizations

5. **`load_models.py`** - Model inference utility
   - `BiasPredictor` class for loading and using trained models
   - Support for different model suffixes (classifier vs regressor)
   - Batch prediction capabilities
   - Model information and debugging utilities

### Orchestration Scripts

6. **`run_analysis.py`** - Main workflow orchestration
   - Supports two analysis types: `bias_correction` and `hte_prediction`
   - Handles single runs and parameter sweeps
   - Configurable analysis parameters and feature selection modes
   - Automated model training, validation, and reporting
   - Output generation with plots, models, and evaluation reports

### Source Package (`src/`)

All core modules are organized in the `src/` directory:
- **`src/data_processing.py`**, **`src/model_building.py`**, **`src/model_evaluation.py`**, **`src/visualization.py`**
- **`src/load_models.py`** - Model loading utilities with flexible suffix support
- **`src/__init__.py`** - Package initialization with key imports

## ⚙️ Configuration

Edit the configuration in `run_analysis.py` to customize your analysis:

```python
config = {
    # Analysis type and target
    'analysis_type': 'hte_prediction',        # 'bias_correction' or 'hte_prediction'
    'target_col': 'HTE_lnk_corrected',        # Target column for prediction
    
    # Run mode
    'single_run': False,                      # True for single run, False for parameter sweep
    'mode': "in_all",                         # "each" or "in_all" feature selection
    'n_features': 0,                          # Number of features (0 for parameter sweep)
    
    # Feature selection
    'feature_selection_mode': 'sequential',   # 'sequential', 'correlation', or 'selected'
    'include_features': [],                   # Always include these features
    'specific_features': [...],               # Use these specific features (if mode='selected')
    
    # Model training
    'hyperparameter_optimization': True,      # Enable Optuna optimization
    
    # Testing and validation
    'run_scrambling_test': True,              # Run validation tests
    'n_scrambling_trials': 10,                # Number of scrambling trials
    
    # Output
    'save_plots': True,                       # Generate visualization files
    'save_models': True                       # Save trained models
}
```

## 🔬 Analysis Types

### 1. Rate Classification Analysis
**Purpose**: Classify reactions as measurable or fast unmeasurable based on molecular features

**Data**: Uses either raw or corrected HTE data
- Binary classification: 0 = measurable rate, 1 = fast unmeasurable rate
- Used as a preprocessing step for other analyses

**Models**:
- **Classifier**: Predicts if reaction has measurable rate (0) or is fast unmeasurable (1)

**Target**: `Fast_unmeasurable` (binary classification)

### 2. Bias Correction Analysis
**Purpose**: Identify and correct systematic measurement bias in HTE rate data

**Data**: Uses `hte_rates_raw_split_into_2tests.csv` with raw HTE measurements
- Focuses on reactions marked as "Slow_unreliable" 
- Calculates bias as: `Controls * 1.5 - HTE_rate` for biased reactions
- Combines classification and regression for comprehensive correction

**Models**:
- **Classifier**: Predicts if reaction is measurable (0) or fast unmeasurable (1)
- **Regressor**: Predicts bias magnitude for correction

**Target**: `bias` (calculated bias values) and `Fast_unmeasurable` (classification)

### 3. HTE Rate Prediction Analysis  
**Purpose**: Predict corrected HTE rates directly from molecular features

**Data**: Uses `corrected_hte_rates.csv` with bias-corrected HTE measurements
- Filters to measurable reactions only (`Fast_unmeasurable == False`)
- Uses log-transformed rates: `HTE_lnk_corrected = log10(HTE_rate_corrected)`

**Models**:
- **Regressor**: Predicts corrected HTE rates or log-transformed rates

**Target**: `HTE_rate_corrected` or `HTE_lnk_corrected`

## 🔮 Using Trained Models

The `src/load_models.py` module provides flexible model loading with support for different suffixes:

### Option 1: Same Suffix (Simple)
```python
from src.load_models import load_models_simple

# Load all models with the same suffix (for bias correction models)
predictor = load_models_simple("_each_8_optuna")

# Make predictions for bias correction
feature_data = {'acyl_pka_aHs_x_has_acidic_H': 1.0, 'amine_pka_basic': 2.0, ...}
predicted_bias = predictor.predict_bias(feature_data)
predicted_class = predictor.predict_rate_class(feature_data)  # 0=measurable, 1=fast
```

### Option 2: Different Suffixes (Flexible)
```python
from src.load_models import BiasPredictor

# Load models with different suffixes for maximum flexibility
predictor = BiasPredictor(
    classifier_suffix="_each_8_optuna",
    regressor_suffix="_in_all_15_optuna", 
    features_suffix="_each_8_optuna"  # Optional, defaults to classifier_suffix
)

# Make predictions
predicted_bias = predictor.predict_bias(feature_data)
```

### Batch Predictions
```python
from src.load_models import load_and_predict_batch

# Process entire CSV files
results = load_and_predict_batch(
    'new_data/combined_features_hte_rates_drug_scope.csv',
    classifier_suffix='_each_8_optuna',
    regressor_suffix='_each_8_optuna',
    save_results=True
)
```

### Model Information
```python
# Get information about loaded models
info = predictor.get_model_info()
print(f"Classifier: {info['classifier_type']}")
print(f"Regressor: {info['regressor_type']}")
print(f"Features ({info['n_features']}): {info['features']}")
```

## 📂 Output Structure

```
ml_measurable_hte_rates/
├── src/                        # 📦 Source package
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   ├── visualization.py
│   └── load_models.py
├── run_analysis.py             # 🎯 Main analysis script
├── README.md                   # 📖 Documentation
├── data/                       # 📊 Input data files
│   ├── hte_rates_raw_split_into_2tests.csv    # Raw HTE data for bias correction
│   ├── corrected_hte_rates.csv                # Corrected HTE data for prediction
│   └── features/                               # Molecular descriptor files
│       ├── descriptors_acyl_chlorides.csv
│       └── descriptors_amines.csv
├── results/                    # 📊 Analysis results and reports
│   ├── combined_df_105_features.csv           # Combined feature dataset
│   ├── acyl_feature_correlations.csv          # Feature correlation analysis
│   ├── amine_feature_correlations.csv
│   ├── parameter_sweep_results.csv            # Parameter sweep results
│   └── reports/                                # Detailed evaluation reports
│       └── hte_prediction_report_*.txt
├── plots/                      # 📈 Visualization files
│   ├── model_performance_*.png                 # Model comparison plots
│   ├── parity_plot_*.png                      # Regression parity plots
│   ├── bias_corrections_*.png                 # Bias correction visualizations
│   ├── acyl_feature_correlations.png          # Feature correlation heatmaps
│   └── amine_feature_correlations.png
├── models/                     # 🤖 Trained model files
│   ├── best_classifier_each_8_optuna.pkl      # Classification models
│   ├── best_regressor_in_all_15_optuna.pkl    # Regression models
│   ├── scaler_class_each_8_optuna.pkl         # Feature scalers
│   ├── scaler_reg_in_all_15_optuna.pkl
│   └── features_each_8_optuna.pkl             # Selected features
└── new_data/                   # 📊 External prediction data
    └── combined_features_hte_rates_drug_scope.csv
```

## ✨ Key Features

### 🧩 Three Analysis Types
- **Rate Classification**: Classify reactions as measurable vs fast unmeasurable
- **Bias Correction**: Predict and correct measurement bias in HTE rates
- **HTE Rate Prediction**: Predict corrected HTE rates from molecular features
- Seamless switching between analysis types via configuration

### 🧩 Modular Design
- Each module has a single, focused responsibility
- Easy to test, modify, and extend individual components
- Clear interfaces between modules
- Reduced complexity: ~3000 lines → ~1500 lines across 5 core modules

### ⚙️ Flexible Configuration
- All parameters centralized in `run_analysis.py`
- Support for different model suffixes and feature selection modes
- Easy to switch between analysis types and modes
- Parameter sweep capabilities for optimization

### 🔬 Comprehensive Validation
- **Feature scrambling tests** - Verify meaningful learning vs. bias exploitation
- **Y-scrambling tests** - Check model architecture flexibility
- **Cross-validation** - Assess model generalization
- **Parity plots** - Visual validation of regression performance

### 📊 Rich Visualizations
- Automated generation of analysis plots
- Model performance comparison plots
- Parity plots for regression models
- Feature correlation heatmaps
- Classification performance visualizations (confusion matrices, ROC curves)

### 🔄 Model Management
- Save/load models with flexible naming conventions
- Support for different classifier/regressor versions
- Batch prediction capabilities
- Model information and debugging tools
- Hyperparameter optimization with Optuna

## 🔄 Analysis Workflows

### Rate Classification Workflow
1. **Data Loading**: Load HTE data (raw or corrected)
2. **Feature Processing**: Load molecular descriptors and select features
3. **Model Training**: Train binary classifier (measurable vs fast unmeasurable)
4. **Classification**: Apply trained model to classify reaction rates
5. **Evaluation**: Generate classification performance metrics and plots

### Bias Correction Workflow
1. **Data Loading**: Load raw HTE data with bias patterns
2. **Feature Processing**: Load molecular descriptors and perform correlation analysis
3. **Model Training**: Train classifier (measurable vs fast) + regressor (bias magnitude)
4. **Bias Correction**: Apply trained models to correct HTE rates
5. **Validation**: Validate corrections and generate evaluation reports

### HTE Rate Prediction Workflow
1. **Data Loading**: Load corrected HTE rate data
2. **Feature Processing**: Load molecular descriptors and select features
3. **Model Training**: Train regression models to predict HTE rates
4. **Evaluation**: Generate parity plots and performance metrics
5. **Reporting**: Create comprehensive evaluation reports

### Parameter Sweep Mode
- Automatically tests different numbers of features
- Identifies optimal feature count for best performance
- Generates summary of all tested configurations
- Saves detailed results for each configuration

## 📋 Dependencies

```bash
# Core scientific computing
pandas numpy matplotlib seaborn

# Machine learning
scikit-learn

# Advanced ML models  
xgboost lightgbm catboost

# Hyperparameter optimization
optuna
```

## 🆘 Troubleshooting

### Common Issues

**Models not found:**
```python
# Check what files exist in models directory
import os
print(os.listdir('models/'))

# Use correct suffix that matches your saved models
predictor = load_models_simple("_each_8_optuna")  # Match your saved models
```

**Missing features error:**
```python
# Check required features for loaded models
from src.load_models import load_models_simple
predictor = load_models_simple("_each_8_optuna")
info = predictor.get_model_info()
print("Required features:", info['features'])
```

**Analysis type configuration:**
```python
# In run_analysis.py, set:
config['analysis_type'] = 'bias_correction'    # For bias correction (includes classification)
config['analysis_type'] = 'hte_prediction'     # For HTE rate prediction
# Note: Rate classification is embedded within bias_correction analysis
```

**Parameter sweep vs single run:**
```python
# In run_analysis.py, set:
config['single_run'] = True   # For single analysis with specific features
config['single_run'] = False  # For parameter sweep across feature counts
```

**Feature selection modes:**
```python
# In run_analysis.py, choose:
config['feature_selection_mode'] = 'sequential'   # Sequential feature selection
config['feature_selection_mode'] = 'correlation'  # Correlation-based selection
config['feature_selection_mode'] = 'selected'     # Use specific_features list
```

## 🤝 Contributing

The modular structure makes it easy to:
- Add new feature selection methods in `data_processing.py`
- Implement new models in `model_building.py` 
- Add validation techniques in `model_evaluation.py`
- Create new visualizations in `visualization.py`

Each module has clear interfaces and focused responsibilities, making development and testing straightforward. 