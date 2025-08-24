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
import argparse
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules from src
from src.data_processing import (
    load_hte_data, load_and_process_features, analyze_bias_patterns,
    process_feature_correlations, select_features_sequentially,
    select_top_features_combined, load_reaction_energy_features,
    load_and_process_features_with_smiles
)
from src.data_processing import select_features_by_correlation_fast
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
from src.splits import generate_random_splits
from src.smiles_features import chemprop_embed, fastprop_descriptors

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

    # Optional reaction-energy features
    rxn_features = None
    if config.get('use_reaction_energies', False):
        rxn_features = load_reaction_energy_features(config.get('reaction_energies_path', 'data/reaction_energies/reaction_TSB_w_aimnet2.csv'))
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
    final_features = []
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
        hyperparameter_optimization=config['hyperparameter_optimization'],
        use_gpu=config.get('use_gpu', False),
        early_stopping=config.get('early_stopping', False),
        group_by=config.get('group_by', None),
        enable_stacking=config.get('enable_stacking', False),
        rxn_features=rxn_features
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
    # Append optional user-provided tag to avoid overwriting artifacts across parallel runs
    extra_tag = config.get('suffix_tag')
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
    # Append optional user-provided tag to avoid overwriting artifacts across parallel runs
    extra_tag = config.get('suffix_tag')
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
    # Append optional user-provided tag to avoid overwriting artifacts across parallel runs
    extra_tag = config.get('suffix_tag')
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
    # Optional extra tag to keep outputs separate across ablations
    extra_tag = config.get('suffix_tag', '')
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
        
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
    
    # Optionally compare without reaction-energy features
    if config.get('compare_rxn_features', False) and config.get('use_reaction_energies', False):
        print("\n=== Comparing without reaction-energy features ===")
        model_results_no_rxn = build_models(
            acid_feature_data, amine_feature_data, df, final_features,
            single_run=config['single_run'],
            hyperparameter_optimization=config['hyperparameter_optimization'],
            use_gpu=config.get('use_gpu', False),
            early_stopping=config.get('early_stopping', False),
            group_by=config.get('group_by', None),
            enable_stacking=config.get('enable_stacking', False),
            rxn_features=None
        )
        _, _, _, _, _, results_no_rxn, _, _, _ = model_results_no_rxn
        try:
            import json, os
            os.makedirs('results', exist_ok=True)
            compare_file = f"results/rxn_feature_comparison_bias{suffix}.json"
            with open(compare_file, 'w') as f:
                json.dump({'with_rxn': results, 'without_rxn': results_no_rxn}, f, indent=2)
            print(f"Saved comparison to {compare_file}")
        except Exception as e:
            print(f"Failed to save comparison file: {e}")

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
    if config.get('use_random_split', True):
        df = generate_random_splits(
            df,
            group_by=config.get('group_by', None),
            ratios=(0.6, 0.2, 0.2),
            seed=config.get('random_split_seed', 42)
        )
    
    # Load and process features
    if config.get('use_smiles_features', True):
        acid_feature_data, amine_feature_data, acid_smiles_map, amine_smiles_map = load_and_process_features_with_smiles(df, target_col=config['target_col'])
        # attach smiles columns for further embedding
        acid_feature_data = acid_feature_data.reset_index()
        acid_feature_data['smiles'] = acid_feature_data['acyl_chlorides'].map(acid_smiles_map)
        acid_feature_data = acid_feature_data.set_index('acyl_chlorides')
        amine_feature_data = amine_feature_data.reset_index()
        amine_feature_data['smiles'] = amine_feature_data['amines'].map(amine_smiles_map)
        amine_feature_data = amine_feature_data.set_index('amines')
        # Generate chemprop embeddings and fastprop descriptors if possible
        try:
            base_acid = acid_feature_data.reset_index()[['acyl_chlorides','smiles']].dropna()
            cp_a = chemprop_embed(base_acid, id_col='acyl_chlorides', smiles_col='smiles', embed_size=256).add_prefix('acyl_')
            acid_feature_data = acid_feature_data.join(cp_a, how='left')
        except Exception:
            pass
        try:
            base_amine = amine_feature_data.reset_index()[['amines','smiles']].dropna()
            cp_b = chemprop_embed(base_amine, id_col='amines', smiles_col='smiles', embed_size=256).add_prefix('amine_')
            amine_feature_data = amine_feature_data.join(cp_b, how='left')
        except Exception:
            pass
        try:
            fp_a = fastprop_descriptors(base_acid, id_col='acyl_chlorides', smiles_col='smiles', compute_padel=False).add_prefix('acyl_')
            acid_feature_data = acid_feature_data.join(fp_a, how='left')
        except Exception:
            pass
        try:
            fp_b = fastprop_descriptors(base_amine, id_col='amines', smiles_col='smiles', compute_padel=False).add_prefix('amine_')
            amine_feature_data = amine_feature_data.join(fp_b, how='left')
        except Exception:
            pass
    else:
        acid_feature_data, amine_feature_data = load_and_process_features(df, target_col=config['target_col'])

    # Optional reaction-energy features
    rxn_features = None
    if config.get('use_reaction_energies', False):
        rxn_features = load_reaction_energy_features(config.get('reaction_energies_path', 'data/reaction_energies/reaction_TSB_w_aimnet2.csv'))

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
        # Use fast correlation-based selector on TRAIN only to avoid leakage
        selected_features = list(set(acid_feature_data.columns[1:].tolist() + amine_feature_data.columns[1:].tolist()))
        tmp_combined_df, feature_cols = create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features, save_df=False)
        cols_for_corr = feature_cols + [config['target_col']]
        if 'test splits' in tmp_combined_df.columns:
            train_mask = (tmp_combined_df['test splits'] == 'TRAIN')
            tmp = tmp_combined_df.loc[train_mask, cols_for_corr].copy()
        else:
            tmp = tmp_combined_df[cols_for_corr].copy()
        final_features = select_features_by_correlation_fast(tmp, config['target_col'], n_top=config['n_features'])
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")
    
    final_features = list(set(final_features))
    print(f"Final selected features ({len(final_features)}): {final_features}")
    
    # Build Models
    model_results = build_hte_prediction_models(
        acid_feature_data, amine_feature_data, df, final_features,
        target_col=config['target_col'],
        single_run=config['single_run'],
        hyperparameter_optimization=config['hyperparameter_optimization'],
        use_gpu=config.get('use_gpu', False),
        early_stopping=config.get('early_stopping', False),
        group_by=config.get('group_by', None),
        enable_stacking=config.get('enable_stacking', False),
        rxn_features=rxn_features,
        train_only_feature_selection=config.get('train_only_feature_selection', True),
        train_sfs_n=config.get('train_sfs_n', 30),
        n_jobs=None,
        allowed_regressors=config.get('allowed_regressors'),
        use_bagging_ensemble=config.get('use_bagging_ensemble', True),
        bagging_estimators=config.get('bagging_estimators', 15),
        bagging_max_samples=config.get('bagging_max_samples', 0.8)
        )
    
    regressor, scaler_reg, valid_features, combined_df, final_results = model_results
    
    # Set suffix
    if config['hyperparameter_optimization']:
        suffix = f"_{config['mode']}_{config['n_features']}_optuna"
    else:
        suffix = f"_{config['mode']}_{config['n_features']}"
    # Append optional user-provided tag to avoid overwriting artifacts across parallel runs
    extra_tag = config.get('suffix_tag')
    if extra_tag:
        suffix = f"{suffix}_{extra_tag}"
    
    # Run testing if requested #### Todo later
    scrambling_results = None
    y_scrambling_results = None
    
        
    # Visualize model performance
    if config['save_plots']:
        print(f"\n=== Generating Visualizations ===")
        
        # Model performance
        visualize_model_performance(final_results, save_plot=True, suffix=suffix)
        
        # Parity plot of predicted vs. actual HTE rates
        if regressor is not None and combined_df is not None and final_results and 'regression' in final_results:
            model_name = final_results.get('regression', {}).get('model', 'Regression Model')
            
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
            # Save per-sample predictions (TRAIN/VAL/TEST) with identifiers for later ensembling
            try:
                import pandas as _pd
                preds = parity_results.get('predictions', {}) if isinstance(parity_results, dict) else {}
                # Convert to plain lists to ensure RangeIndex alignment with ID frames
                y_train_true = _pd.Series(list(preds.get('y_train_true', [])), name='y_true')
                y_train_pred = _pd.Series(list(preds.get('y_train_pred', [])), name='y_pred')
                y_test_true = _pd.Series(list(preds.get('y_test_true', [])), name='y_true')
                y_test_pred = _pd.Series(list(preds.get('y_test_pred', [])), name='y_pred')
                y_test_std = preds.get('y_test_std', None)
                if y_test_std is not None:
                    y_test_std = _pd.Series(list(y_test_std), name='y_std')

                train_mask = combined_df['test splits'] == 'TRAIN'
                val_mask = combined_df['test splits'] == 'VAL'
                test_mask = (combined_df['test splits'] == 'TEST1') | (combined_df['test splits'] == 'TEST2')

                train_ids = combined_df.loc[train_mask, ['acyl_chlorides', 'amines']].reset_index(drop=True)
                val_ids = combined_df.loc[val_mask, ['acyl_chlorides', 'amines']].reset_index(drop=True)
                test_ids = combined_df.loc[test_mask, ['acyl_chlorides', 'amines']].reset_index(drop=True)

                train_df = _pd.concat([train_ids, y_train_true, y_train_pred], axis=1)
                train_df['split'] = 'TRAIN'
                # Build VAL predictions directly using regressor/scaler to avoid leakage
                val_df = None
                try:
                    if val_mask.any():
                        _X_val = combined_df.loc[val_mask, valid_features]
                        _trained_cols = getattr(scaler_reg, 'feature_names_in_', None)
                        if _trained_cols is not None:
                            try:
                                _X_val = _X_val.reindex(columns=list(_trained_cols))
                            except Exception:
                                _common = [c for c in _trained_cols if c in _X_val.columns]
                                _X_val = _X_val.reindex(columns=_common)
                        _X_val_scaled = scaler_reg.transform(_X_val)
                        _y_val_pred = regressor.predict(_X_val_scaled)
                        _y_val_true = combined_df.loc[val_mask, config['target_col']].reset_index(drop=True)
                        _y_val_true = _pd.Series(list(_y_val_true), name='y_true')
                        _y_val_pred = _pd.Series(list(_y_val_pred), name='y_pred')
                        val_df = _pd.concat([val_ids, _y_val_true, _y_val_pred], axis=1)
                        val_df['split'] = 'VAL'
                except Exception:
                    val_df = None
                test_df = _pd.concat([test_ids, y_test_true, y_test_pred], axis=1)
                if y_test_std is not None:
                    test_df = _pd.concat([test_df, y_test_std], axis=1)
                test_df['split'] = 'TEST'

                _frames = [train_df]
                if val_df is not None:
                    _frames.append(val_df)
                _frames.append(test_df)
                preds_df = _pd.concat(_frames, axis=0, ignore_index=True)
                import os as _os
                _os.makedirs('results', exist_ok=True)
                preds_path = f"results/predictions{suffix}.csv"
                preds_df.to_csv(preds_path, index=False)
                print(f"Saved predictions to {preds_path}")
            except Exception as _e:
                print(f"Warning: failed to save predictions CSV: {_e}")
            
    # Save combined dataset snapshot and models
    try:
        import os as _os
        _os.makedirs('results', exist_ok=True)
        combined_path = f"results/combined_df{suffix}.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Saved combined dataset to {combined_path}")
    except Exception as _e:
        print(f"Warning: failed to save combined dataset: {_e}")

    if config.get('save_models', False):
        try:
            save_models(None, regressor, None, scaler_reg, valid_features, suffix=suffix)
        except Exception as _e:
            print(f"Warning: failed to save models: {_e}")

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
    
    # Optionally compare without reaction-energy features
    if config.get('compare_rxn_features', False) and config.get('use_reaction_energies', False):
        print("\n=== Comparing without reaction-energy features (HTE prediction) ===")
        model_results_no_rxn = build_hte_prediction_models(
            acid_feature_data, amine_feature_data, df, final_features,
            target_col=config['target_col'],
            single_run=config['single_run'],
            hyperparameter_optimization=config['hyperparameter_optimization'],
            use_gpu=config.get('use_gpu', False),
            early_stopping=config.get('early_stopping', False),
            group_by=config.get('group_by', None),
            enable_stacking=config.get('enable_stacking', False),
            rxn_features=None,
            train_only_feature_selection=config.get('train_only_feature_selection', True),
            train_sfs_n=config.get('train_sfs_n', 30),
            n_jobs=None,
            allowed_regressors=config.get('allowed_regressors')
            # keep regressors default set
            )
        _, _, _, combined_df_no_rxn, final_results_no_rxn = model_results_no_rxn
        try:
            import json, os
            os.makedirs('results', exist_ok=True)
            compare_file = f"results/rxn_feature_comparison_hte{suffix}.json"
            with open(compare_file, 'w') as f:
                json.dump({'with_rxn': final_results, 'without_rxn': final_results_no_rxn}, f, indent=2)
            print(f"Saved comparison to {compare_file}")
        except Exception as e:
            print(f"Failed to save comparison file: {e}")

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
    rxn_features = None
    if config.get('use_reaction_energies', False):
        rxn_features = load_reaction_energy_features(config.get('reaction_energies_path', 'data/reaction_energies/reaction_TSB_w_aimnet2.csv'))
    
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
    max_feats = 1
    if config['mode'] == "each":
        max_feats = min(len(acid_feature_data.columns), len(amine_feature_data.columns))
    elif config['mode'] == "in_all":
        max_feats = len(acid_feature_data.columns) + len(amine_feature_data.columns)
    else:
        raise ValueError(f"Invalid mode: {config['mode']}")
    
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
        try:
            if config['analysis_type'] == 'bias_correction' and 'r2_improvement' in sweep_df.columns:
                best_idx = sweep_df['r2_improvement'].idxmax()
            elif config['analysis_type'] == 'hte_prediction' and 'reg_test_r2' in sweep_df.columns:
                best_idx = sweep_df['reg_test_r2'].idxmax()
            else:
                best_idx = None
            if best_idx is not None:
                best_config = sweep_df.iloc[best_idx]
                print(f"\nBest configuration:")
                print(f"  Features: {best_config['n_features']}")
                if config['analysis_type'] == 'bias_correction':
                    print(f"  R² improvement: {best_config.get('r2_improvement', 0):.3f}")
                    print(f"  CV F1 score: {best_config.get('class_cv_f1_mean', 0):.3f}")
                print(f"  CV R² score: {best_config.get('reg_cv_r2_mean', 0):.3f}")
                if config['analysis_type'] == 'hte_prediction':
                    print(f"  Test R² score: {best_config.get('reg_test_r2', 0):.3f}")
        except Exception as e:
            print(f"Failed to summarize best configuration: {e}")
    
    return all_results

def main():
    """Main function to orchestrate the analysis."""
    print("=== ML Measurable HTE Rate Bias Analysis ===")
    
    # Configuration
    config = {
        # Type of analysis
        'analysis_type': 'hte_prediction',  # OPTIONS: bias_correction, hte_prediction
        # Target column
        'target_col': 'HTE_lnk_corrected',  # OPTIONS: bias, HTE_rate_corrected, HTE_lnk_corrected
        # Run mode
        'single_run': True,
        'mode': "each",  # "each" or "in_all"
        'n_features': 8,
        # Feature selection
        'feature_selection_mode': 'selected',  # OPTIONS: sequential, correlation, selected
        'include_features': [],
        'specific_features': [
            'amine_class_1_mixture',
            'acyl_class_aromatic',
            'acyl_Charges_secondary_1',
            'amine_Charges_secondary_1',
            'acyl_pka_aHs_x_has_acidic_H',
            'amine_pka_basic',
            'acyl_BV_secondary_2',
            'amine_BV_secondary_avg'
        ],
        # Model training
        'hyperparameter_optimization': True,
        'use_gpu': True,
        'early_stopping': True,
        'enable_stacking': True,
        'group_by': 'pair',  # OPTIONS: None, 'acyl', 'amine', 'pair' — can try 'amine' or 'acyl' too
        # Reaction-energy / TS features
        'use_reaction_energies': True,
        'reaction_energies_path': 'data/reaction_energies/reaction_TSB_w_aimnet2.csv',
        'compare_rxn_features': True,
        # Random split control
        'use_random_split': True,
        'random_split_seed': 42,
        # SMILES features
        'use_smiles_features': True,
        # Train-only feature selection within folds (to avoid leakage)
        'train_only_feature_selection': True,
        'train_sfs_n': 30,
        # Testing and validation
        'run_scrambling_test': False,
        'n_scrambling_trials': 10,
        # Output
        'save_plots': True,
        'save_models': True
    }
    # CLI overrides for ablations
    parser = argparse.ArgumentParser(description='Run HTE analysis with overrides')
    parser.add_argument('--mode', choices=['each', 'in_all'])
    parser.add_argument('--n-features', type=int)
    parser.add_argument('--feature-selection-mode', choices=['sequential', 'correlation', 'selected'])
    parser.add_argument('--group-by', choices=['pair', 'amine', 'acyl', 'none'])
    parser.add_argument('--use-rxn', type=int, choices=[0, 1])
    parser.add_argument('--target-col')
    parser.add_argument('--suffix-tag')
    parser.add_argument('--split-seed', type=int)
    parser.add_argument('--no-random-split', action='store_true')
    parser.add_argument('--no-smiles', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-bagging', action='store_true')
    parser.add_argument('--no-hpo', action='store_true', help='Disable hyperparameter optimization to evaluate full regressor set')
    parser.add_argument('--no-stacking', action='store_true', help='Disable stacking meta-learner step')
    parser.add_argument('--allowed-regressors', type=str, help='Comma-separated list of regressor names to evaluate (e.g., "Extra Trees,SVR,Kernel Ridge,Gaussian Process")')
    args, unknown = parser.parse_known_args()

    if args.mode:
        config['mode'] = args.mode
    if args.n_features is not None:
        config['n_features'] = args.n_features
    if args.feature_selection_mode:
        config['feature_selection_mode'] = args.feature_selection_mode
    if args.group_by:
        config['group_by'] = None if args.group_by == 'none' else args.group_by
    if args.use_rxn is not None:
        config['use_reaction_energies'] = bool(args.use_rxn)
    if args.target_col:
        config['target_col'] = args.target_col
    if args.suffix_tag:
        config['suffix_tag'] = args.suffix_tag
    if args.split_seed is not None:
        config['random_split_seed'] = args.split_seed
    if args.no_random_split:
        config['use_random_split'] = False
    if args.no_smiles:
        config['use_smiles_features'] = False
    if args.cpu:
        config['use_gpu'] = False
    if args.no_bagging:
        config['use_bagging_ensemble'] = False
    if args.no_hpo:
        config['hyperparameter_optimization'] = False
    if args.no_stacking:
        config['enable_stacking'] = False
    if args.allowed_regressors:
        config['allowed_regressors'] = [s.strip() for s in args.allowed_regressors.split(',') if s.strip()]
    
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