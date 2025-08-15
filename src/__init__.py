"""
ML Measurable HTE Rate Bias Analysis - Source Package
====================================================

This package contains the core modules for bias analysis and correction.
"""

from .data_processing import (
    load_hte_data,
    load_and_process_features,
    analyze_bias_patterns,
    process_feature_correlations,
    select_features_sequentially,
    select_top_features_combined
)

from .model_building import (
    build_models,
    save_models,
    load_models
)

from .model_evaluation import (
    apply_improved_corrections,
    validate_corrections,
    test_model_with_scrambled_features,
    test_model_with_y_scrambling,
    save_evaluation_report
)

from .visualization import (
    visualize_model_performance,
    plot_bias_corrections,
    plot_scrambling_test_results
)

from .load_models import (
    BiasPredictor,
    load_models_simple,
    load_and_predict_batch
)

__version__ = "1.0.0"
__author__ = "ML Bias Analysis Team"

__all__ = [
    # Data processing
    'load_hte_data',
    'load_and_process_features', 
    'analyze_bias_patterns',
    'process_feature_correlations',
    'select_features_sequentially',
    'select_top_features_combined',
    
    # Model building
    'build_models',
    'save_models',
    'load_models',
    
    # Model evaluation
    'apply_improved_corrections',
    'validate_corrections',
    'test_model_with_scrambled_features',
    'test_model_with_y_scrambling',
    'save_evaluation_report',
    
    # Visualization
    'visualize_model_performance',
    'plot_bias_corrections',
    'plot_scrambling_test_results',
    
    # Model loading
    'BiasPredictor',
    'load_models_simple',
    'load_and_predict_batch'
] 