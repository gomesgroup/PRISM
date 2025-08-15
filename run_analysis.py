#!/usr/bin/env python3
"""
Main Analysis Script
===================
Run ML bias correction analysis from the ml_measurable_hte_rates directory.
This script can be called from the parent directory and uses the src package.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules from src
from src.data_processing import (
    load_hte_data, load_and_process_features, analyze_bias_patterns,
    process_feature_correlations, select_features_sequentially,
    select_top_features_combined
)
from src.model_building import build_models, build_hte_prediction_models, save_models, create_combined_dataset
from src.model_evaluation import (
    apply_improved_corrections, validate_corrections, apply_rate_classification,
    test_model_with_scrambled_features, test_model_with_y_scrambling,
    save_evaluation_report, plot_parity, plot_parity_with_residuals,
    evaluate_regression_with_parity_plots, save_hte_prediction_evaluation_report
)
from src.visualization import (
    visualize_model_performance, visualize_classification_performance, 
    plot_bias_corrections, plot_scrambling_test_results
)

def create_directories():
    """Create necessary directories for outputs."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def run_single_bias_correction_analysis(config):
    """Run a single analysis with given configuration."""
    
    if config['feature_selection_mode'] == 'selected':
        config['n_features'] = len(config['specific_features'])
    
    print(f"\n{'='*50}")
    print(f"Running Analysis: {config['mode']} with {config['n_features']} features")
    print(f"{'='*50}")
    
    # Load and process data
    print("\n=== Loading Data ===")
    df = load_hte_data()
    
    # Analyze bias patterns
    # analyze_bias_patterns(df, save_plot=config['save_plots'])
    
    # Load and process features
    acid_feature_data, amine_feature_data = load_and_process_features(df)
    # Process feature correlations
    print("\n=== Processing Feature Correlations ===")
    reduced_acid_features, acid_corr_df = process_feature_correlations(
        acid_feature_data,
        target_col='max_bias',
        features=[f for f in config['specific_features'] if f.startswith('acyl_')],
        correlation_threshold=0.95,
        top_n_to_plot=6,
        corr_csv_filename='results/acyl_feature_correlations.csv',
        plot_filename='plots/acyl_feature_correlations.png' if config['save_plots'] else None,
        plot_title='Acyl Chloride Feature Correlations vs. Max Bias'
    )
    
    reduced_amine_features, amine_corr_df = process_feature_correlations(
        amine_feature_data,
        target_col='max_bias',
        features=[f for f in config['specific_features'] if f.startswith('amine_')],
        correlation_threshold=0.95,
        top_n_to_plot=6,
        corr_csv_filename='results/amine_feature_correlations.csv',
        plot_filename='plots/amine_feature_correlations.png' if config['save_plots'] else None,
        plot_title='Amine Feature Correlations vs. Max Bias'
    )
    
    print(f"Reduced to {len(reduced_acid_features)} acyl and {len(reduced_amine_features)} amine features")

    # Select final features
    print(f"\n=== Selecting Features ({config['feature_selection_mode']}) ===")
    if config['mode'] == "each":
        if config['feature_selection_mode'] == 'sequential':
            acyl_feats = select_features_sequentially(
                acid_feature_data[reduced_acid_features + ['max_bias']], 
                'max_bias', 
                n_top=config['n_features']
            )
            amine_feats = select_features_sequentially(
                amine_feature_data[reduced_amine_features + ['max_bias']], 
                'max_bias', 
                n_top=config['n_features']
            )
            final_features = list(set(acyl_feats + amine_feats + config['include_features']))
        elif config['feature_selection_mode'] == 'correlation':
            final_features = (reduced_acid_features[:config['n_features']] + 
                            reduced_amine_features[:config['n_features']] + 
                            config['include_features'])
        elif config['feature_selection_mode'] == 'selected':
            final_features = list(set(reduced_acid_features + reduced_amine_features))
        else:
            raise ValueError(f"Invalid feature selection mode: {config['feature_selection_mode']}")
        
        final_features = list(set(final_features))
    elif config['mode'] == "in_all":
        final_features = select_top_features_combined(
            acid_corr_df, amine_corr_df, 
            n_features=config['n_features'], 
            include_features=config['include_features']
        )
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")

    print(f"Final selected features ({len(final_features)}): {final_features}")
    
    # Build models
    print(f"\n=== Building Models ===")
    model_results = build_models(
        acid_feature_data, amine_feature_data, df, final_features,
        single_run=config['single_run'],
        hyperparameter_optimization=config['hyperparameter_optimization']
    )
    
    classifier, regressor, scaler_class, scaler_reg, valid_features, results, predict_bias_func, combined_df, predict_class_func = model_results
    
    if classifier is None or regressor is None:
        print("Failed to build models")
        return None
    
    # Set suffix
    if config['hyperparameter_optimization']:
        suffix = f"_{config['mode']}_{config['n_features']}_optuna"
    else:
        suffix = f"_{config['mode']}_{config['n_features']}"
        
    # Apply corrections
    print(f"\n=== Applying Bias Corrections ===")
    df_corrected = apply_improved_corrections(df, predict_bias_func)
    df_corrected = apply_rate_classification(df_corrected, predict_class_func, suffix=suffix, 
                                             save_results=config['single_run'])
    
    # Validate corrections
    validation_results = validate_corrections(df_corrected, save_results=config['save_models'], suffix=suffix, 
                                            selection_mode=config['mode'], n_features=config['n_features'])
    
    # Run testing if requested
    scrambling_results = None
    y_scrambling_results = None
    
    if config['run_scrambling_test']:
        print(f"\n=== Running Validation Tests ===")
        
        # Feature scrambling test
        scrambling_results = test_model_with_scrambled_features(
            acid_feature_data, amine_feature_data, df, final_features,
            classifier, regressor, scaler_class, scaler_reg, valid_features,
            n_scrambling_trials=config['n_scrambling_trials']
        )
        
        # Y-scrambling test
        y_scrambling_results = test_model_with_y_scrambling(
            acid_feature_data, amine_feature_data, df, final_features, valid_features,
            n_scrambling_trials=config['n_scrambling_trials']
        )
    
    # Generate visualizations
    if config['save_plots']:
        print(f"\n=== Generating Visualizations ===")
        
        # Model performance
        visualize_model_performance(results, save_plot=True, suffix=suffix)
        
        # Classification performance plots for train and test
        if results and 'classification' in results and 'y_train' in results['classification'] and 'y_test' in results['classification']:
            print("Generating classification performance plots...")
            
            # Train dataset visualization
            visualize_classification_performance(
                results['classification']['y_train'], 
                results['classification']['y_train_pred'],
                dataset_type="train",
                save_plot=True,
                suffix=suffix
            )
            
            # Test dataset visualization  
            visualize_classification_performance(
                results['classification']['y_test'], 
                results['classification']['y_test_pred'],
                dataset_type="test",
                save_plot=True,
                suffix=suffix
            )
        
        # Bias corrections
        plot_bias_corrections(df_corrected, save_plot=config['single_run'], suffix=suffix)
        
        # Scrambling test results
        if scrambling_results:
            plot_scrambling_test_results(scrambling_results, save_plot=True, suffix=suffix)
    
    # Save models
    if config['save_models']:
        save_models(classifier, regressor, scaler_class, scaler_reg, valid_features, suffix=suffix)
    
    # Save evaluation report
    save_evaluation_report(validation_results, results, df_corrected, final_features, 
                          selection_mode=config.get('selection_mode', 'manual'), 
                          n_features=config.get('n_features', len(final_features)),
                          scrambling_results=scrambling_results, 
                          y_scrambling_results=y_scrambling_results, 
                          suffix=suffix)
    
    # Compile results
    analysis_results = {
        'configuration': config,
        'n_features': len(final_features),
        'selected_features': final_features,
        'model_results': results,
        'validation_results': validation_results,
        'scrambling_results': scrambling_results,
        'y_scrambling_results': y_scrambling_results,
        'corrected_data': df_corrected
    }
    
    return analysis_results

def run_single_hte_prediction_analysis(config):
    """Run a single analysis with given configuration."""
    
    if config['feature_selection_mode'] == 'selected':
        config['n_features'] = len(config['specific_features'])
    
    print(f"\n{'='*50}")
    print(f"Running HTE Prediction Analysis: {config['mode']} with {config['n_features']} features")
    print(f"{'='*50}")
    
    # Load and process data
    print("\n=== Loading Data ===")
    df = load_hte_data(analysis_type='hte_prediction', target_col=config['target_col'])
    
    # Load and process features
    acid_feature_data, amine_feature_data = load_and_process_features(df, target_col=config['target_col'])

    # Select final features
    print(f"\n=== Selecting Features ({config['feature_selection_mode']}) ===")
    if config['mode'] == "each":
        if config['feature_selection_mode'] == 'sequential':
            acyl_feats = select_features_sequentially(
                acid_feature_data, 
                config['target_col'], 
                n_top=config['n_features']
            )
            amine_feats = select_features_sequentially(
                amine_feature_data, 
                config['target_col'], 
                n_top=config['n_features']
            )
            final_features = list(set(acyl_feats + amine_feats + config['include_features']))
        elif config['feature_selection_mode'] == 'selected':
            final_features = list(set(config['specific_features']))
        else:
            raise ValueError(f"Invalid feature selection mode: {config['feature_selection_mode']}")
    
    elif config['mode'] == "in_all":
        selected_features = list(set(acid_feature_data.columns[1:].tolist() + amine_feature_data.columns[1:].tolist()))

        combined_df, feature_cols = create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features, save_df=False)
        combined_df = combined_df[feature_cols + [config['target_col']]]
        
        final_features = select_features_sequentially(
            combined_df, 
            config['target_col'], 
            n_top=config['n_features']
        )
    
    final_features = list(set(final_features))
    print(f"Final selected features ({len(final_features)}): {final_features}")
    
    # Build Models
    model_results = build_hte_prediction_models(
        acid_feature_data, amine_feature_data, df, final_features,
        target_col=config['target_col'],
        single_run=config['single_run'],
        hyperparameter_optimization=config['hyperparameter_optimization']
        )
    
    regressor, scaler_reg, valid_features, combined_df, final_results = model_results
    
    # Set suffix
    if config['hyperparameter_optimization']:
        suffix = f"_{config['mode']}_{config['n_features']}_optuna"
    else:
        suffix = f"_{config['mode']}_{config['n_features']}"
        
    # Run testing if requested #### Todo later
    scrambling_results = None
    y_scrambling_results = None
    
        
    # Visualize model performance
    if config['save_plots']:
        print(f"\n=== Generating Visualizations ===")
        
        # Model performance
        visualize_model_performance(final_results, save_plot=True, suffix=suffix)
        
        # Parity plot of predicted vs. actual HTE rates
        if regressor is not None and combined_df is not None:
            model_name = final_results['regression'].get('model', 'Regression Model')
            
            # Use comprehensive evaluation function
            parity_results = evaluate_regression_with_parity_plots(
                regressor=regressor,
                scaler=scaler_reg,
                combined_df=combined_df,
                valid_features=valid_features,
                target_col=config['target_col'],
                model_name=model_name,
                save_plots=True,
                suffix=suffix
            )
            
    # Save evaluation report
    save_hte_prediction_evaluation_report(final_results, combined_df, final_features, 
                          selection_mode=config.get('selection_mode', 'manual'), 
                          n_features=config.get('n_features', len(final_features)),
                          scrambling_results=scrambling_results, 
                          y_scrambling_results=y_scrambling_results, 
                          suffix=suffix)
    
    # Compile results
    analysis_results = {
        'configuration': config,
        'n_features': len(final_features),
        'selected_features': final_features,
        'model_results': final_results,
        'scrambling_results': scrambling_results,
        'y_scrambling_results': y_scrambling_results,
        'predicted_data': combined_df
    }
    
    return analysis_results

def run_parameter_sweep(config):
    """Run parameter sweep across different feature numbers."""
    print(f"\n{'='*50}")
    print("Running Parameter Sweep")
    print(f"{'='*50}")
    
    # Load and process data once
    print("\n=== Loading Data ===")
    df = load_hte_data(analysis_type=config['analysis_type'], target_col=config['target_col'])
    
    # analyze_bias_patterns(df, save_plot=False)
    acid_feature_data, amine_feature_data = load_and_process_features(df, target_col=config['target_col'])
    
    # # Process feature correlations
    # reduced_acid_features, acid_corr_df = process_feature_correlations(
    #     acid_feature_data, target_col=config['target_col'], correlation_threshold=0.95
    # )
    # reduced_amine_features, amine_corr_df = process_feature_correlations(
    #     amine_feature_data, target_col=config['target_col'], correlation_threshold=0.95
    # )
    
    # Determine feature range
    # max_feats_each = min(len(reduced_acid_features), len(reduced_amine_features))
    # feature_range = range(1, max_feats_each + 1)
    if config['mode'] == "each":
        max_feats = min(len(acid_feature_data.columns), len(amine_feature_data.columns))
    elif config['mode'] == "in_all":
        max_feats = len(acid_feature_data.columns) + len(amine_feature_data.columns)
    
    feature_range = range(1, max_feats + 1)
    print(f"Testing feature range: 1 to {max_feats}")

    all_results = []
    
    for n_features in feature_range:
        print(f"\n--- Testing {n_features} features ---")
        
        # Update config for this run
        sweep_config = config.copy()
        sweep_config['n_features'] = n_features
        sweep_config['save_plots'] = False  # Don't save plots for each run
        sweep_config['run_scrambling_test'] = False  # Skip for sweep
        sweep_config['save_models'] = False
        
        if config['analysis_type'] == 'bias_correction':
            try:
                result = run_single_bias_correction_analysis(sweep_config)
                
                if result:
                    # Add sweep-specific metrics
                    sweep_result = {
                        'n_features': n_features,
                        'r2_corrected': result['validation_results'].get('r_squared_corr', 0),
                        'r2_improvement': result['validation_results'].get('r2_improvement', 0),
                        'point_improvement': result['validation_results'].get('n_point_improvement', 0),
                        'class_cv_f1_mean': result['model_results']['classification'].get('cv_f1_mean', 0),
                        'reg_cv_r2_mean': result['model_results']['regression'].get('cv_r2_mean', 0),
                        'reg_test_r2': result['model_results']['regression'].get('test_r2', 0),
                        'class_model': result['model_results']['classification'].get('model', 'Unknown'),
                        'reg_model': result['model_results']['regression'].get('model', 'Unknown'),
                        'selection_mode': config['mode'],
                        'features': result['selected_features']
                    }
                    all_results.append(sweep_result)
            except Exception as e:
                print(f"Error with Bias Correction analysis with {n_features} features: {e}")
                continue
            
        elif config['analysis_type'] == 'hte_prediction':
            try:
                result = run_single_hte_prediction_analysis(sweep_config)
                
                if result:
                    # Add sweep-specific metrics
                    sweep_result = {
                        'n_features': n_features,
                        'reg_model': result['model_results']['regression'].get('model', 'Unknown'),
                        'reg_cv_r2_mean': result['model_results']['regression'].get('cv_r2_mean', 0),
                        'reg_test_r2': result['model_results']['regression'].get('test_r2', 0),
                        'selection_mode': config['mode'],
                        'features': result['selected_features']
                    }
                    all_results.append(sweep_result)
            except Exception as e:
                print(f"Error with HTE Prediction analysis with {n_features} features: {e}")
                continue
    
    # Save sweep results
    if all_results:
        import pandas as pd
        sweep_df = pd.DataFrame(all_results)
        sweep_df.to_csv('results/parameter_sweep_results.csv', index=False)
        print(f"\nParameter sweep results saved to results/parameter_sweep_results.csv")
        
        # Find best configuration
        if config['analysis_type'] == 'bias_correction':
            best_idx = sweep_df['r2_improvement'].idxmax()
        elif config['analysis_type'] == 'hte_prediction':
            best_idx = sweep_df['reg_test_r2'].idxmax()
        best_config = sweep_df.iloc[best_idx]
        print(f"\nBest configuration:")
        print(f"  Features: {best_config['n_features']}")
        if config['analysis_type'] == 'bias_correction':
            print(f"  R² improvement: {best_config['r2_improvement']:.3f}")
            print(f"  CV F1 score: {best_config['class_cv_f1_mean']:.3f}")
        print(f"  CV R² score: {best_config['reg_cv_r2_mean']:.3f}")
        if config['analysis_type'] == 'hte_prediction':
            print(f"  Test R² score: {best_config['reg_test_r2']:.3f}")
    
    return all_results

def main():
    """Main function to orchestrate the analysis."""
    print("=== ML Measurable HTE Rate Bias Analysis ===")
    
    # Configuration
    config = {
        # Type of analysis
        'analysis_type': 'hte_prediction', # OPTIONS: bias_correction, hte_prediction
        
        # Target column
        'target_col': 'HTE_lnk_corrected', # OPTIONS: bias, HTE_rate_corrected, HTE_lnk_corrected
        
        # Run mode
        'single_run': False,
        'mode': "in_all",  # "each" or "in_all"
        'n_features': 0,
        
        # Feature selection
        'feature_selection_mode': 'sequential',  # OPTIONS: sequential, correlation, selected
        'include_features': [], # ['amine_pka_basic', 'acyl_pka_aHs_x_has_acidic_H',  'amine_BV_secondary_avg', 'acyl_BV_secondary_2'],
        'specific_features': ['amine_class_1_mixture', 'acyl_class_aromatic', 'acyl_Charges_secondary_1', 'amine_Charges_secondary_1', 'acyl_pka_aHs_x_has_acidic_H', 'amine_pka_basic', 'acyl_BV_secondary_2', 'amine_BV_secondary_avg'], 
        
        ### Try combination: 
        # 6 feats previous: ['amine_pka_basic', 'acyl_pka_aHs_x_has_acidic_H', 'amine_BV_secondary_avg', 'acyl_BV_secondary_2', 'acyl_num_Hs_x_has_num_Hs', 'amine_class_2_aliphatic']
        # 6 feats: ['amine_class_2_aliphatic', 'acyl_Charges_secondary_1','acyl_pka_aHs_x_has_acidic_H', 'amine_pka_basic', 'acyl_BV_secondary_2', 'amine_BV_secondary_avg']
        # 7 feats: 6feats + ['amine_Charges_secondary_1']
        # 8 feats: ['amine_class_1_mixture', 'acyl_class_aromatic', 'acyl_Charges_secondary_1', 'amine_Charges_secondary_1', 'acyl_pka_aHs_x_has_acidic_H', 'amine_pka_basic', 'acyl_BV_secondary_2', 'amine_BV_secondary_avg']
        # 9 feats: 8feats + ['amine_class_2_aliphatic']
        # 8_2 and 9_2 feats: swap 'acyl_Charges_secondary_1' to 'acyl_Charges_secondary_2'
        
        # Model training
        'hyperparameter_optimization': True,
        
        # Testing and validation
        'run_scrambling_test': True,
        'n_scrambling_trials': 10,
        
        # Output
        'save_plots': True,
        'save_models': True
    }
    
    # Create directories
    create_directories()
    
    # Run analysis
    if config['single_run']:
        if config['analysis_type'] == 'bias_correction':
            results = run_single_bias_correction_analysis(config)
        elif config['analysis_type'] == 'hte_prediction':
            results = run_single_hte_prediction_analysis(config)
        else:
            raise ValueError(f"Invalid analysis type: {config['analysis_type']}")
        
        if results:
            print(f"\n{'='*50}")
            print("ANALYSIS COMPLETE")
            print(f"{'='*50}")
            print(f"Check the following directories for outputs:")
            print(f"  - plots/: Visualization files")
            print(f"  - results/: Data files and reports")
            print(f"  - models/: Trained model files")
        else:
            print("Analysis failed")
    else:
        # Run parameter sweep
        results = run_parameter_sweep(config)
        print(f"\nParameter sweep complete with {len(results)} configurations tested")

if __name__ == "__main__":
    main() 