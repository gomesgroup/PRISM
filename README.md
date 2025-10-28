# PRISM: Parallelized Reaction-rates via Indicator Spectrometry using Machine-vision

This repository contains the complete workflow for predicting the type and amide coupling PRISM reaction rates using machine learning and graph neural networks.

## Contents

- **`build_class_and_bias_models/`** - the ML models (classifiers and regressors) with hyperparameter optimization via Optuna to classify the type of PRISM rate and predict the bias and correct the PRISM rates. [See README](build_class_and_bias_models/README.md)
- **`build_gnn_model/`** - the AIM Graph neural network model for predicting the PRISM reaction rate value. [See README](build_gnn_model/README.md)
- **`data/`** - Datasets including molecular and atomistic descriptors, reaction rates, and XYZ molecular structures.
- **`generate_features/`** - Scripts for generating molecular/atomistic features from structures using the Morfeus python package and pKa calculators.
- **`image_analysis/`** - Image processing scripts for analyzing the PRISM high-throughput experimental plate data. [See README](image_analysis/Amide_Code/README.md)
- **`predictions_from_class_bias.ipynb`** - Jupyter notebook for making PRISM classification predictions on new reaction combinations. [Open notebook](predictions_from_class_bias.ipynb)
- **`predictions_from_gnn.ipynb`** - Jupyter notebook for making PRISM HTE rate predictions on new reaction combinations. [Open notebook](predictions_from_gnn.ipynb)

## Citation

Research paper coming soon! To cite:

```
[Citation will be added upon publication]
```

