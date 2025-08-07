#!/usr/bin/env python3
"""
Model Evaluation Module
======================
Handles model validation, bias corrections, and testing procedures.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

def apply_rate_classification(df, predict_rate_func, suffix="", save_results=True):
    """Apply rate classification to HTE rates using the trained model."""
    
    df_classified = df.copy()
    
    # Initialize classified rate with original rate
    df_classified['classified_rate'] = df_classified['Fast_unmeasurable'].copy()
    
    # Apply classification
    for idx in df_classified.index:
        acid_id = df_classified.loc[idx, 'acyl_chlorides']
        amine_id = df_classified.loc[idx, 'amines']
        
        # Predict rate class
        predicted_class = predict_rate_func(acid_id, amine_id)
        
        # Update classified rate
        df_classified.loc[idx, 'classified_rate'] = predicted_class
        
    # save classified rates
    if save_results:
        df_classified.to_csv(f'results/classified_hte_rates{suffix}.csv', index=False)
    
    return df_classified
          
def apply_improved_corrections(df, predict_bias_func):
    """Apply bias corrections to HTE rates using the trained model."""
    
    df_corrected = df.copy()
    
    # Initialize corrected rate with original rate
    df_corrected['corrected_HTE_rate_all'] = df_corrected['HTE_rate'].copy()
    df_corrected['corrected_HTE_rate'] = df_corrected['HTE_rate'].copy()
    df_corrected['predicted_bias'] = 0.0
    df_corrected['correction_applied'] = False
    
    # Apply corrections
    corrections_applied = 0
    for idx in df_corrected.index:
        acid_id = df_corrected.loc[idx, 'acyl_chlorides']
        amine_id = df_corrected.loc[idx, 'amines']
        original_rate = df_corrected.loc[idx, 'HTE_rate']
        is_biased = df_corrected.loc[idx, 'is_biased']
        if 'nmr_rate_2' in df_corrected.columns:
            nmr_rate = df_corrected.loc[idx, 'nmr_rate_2']
        
        # Predict bias
        predicted_bias = predict_bias_func(acid_id, amine_id)
        df_corrected.loc[idx, 'predicted_bias'] = predicted_bias
        
        # Apply correction if bias is predicted
        if predicted_bias > 0: 
            #### Subtract the bias control to the HTE rate
            corrected_rate = original_rate - predicted_bias
            #### Add the delta bias from the NMR rate (delta bias = HTE rate - NMR rate)
            # corrected_rate = nmr_rate + predicted_bias
        
            # Apply physical constraints
            corrected_rate = max(corrected_rate, 0.0)  # Non-negative rates
            df_corrected.loc[idx, 'corrected_HTE_rate_all'] = corrected_rate
            
            if is_biased:
                df_corrected.loc[idx, 'corrected_HTE_rate'] = corrected_rate
                df_corrected.loc[idx, 'correction_applied'] = True
                corrections_applied += 1
    
    print(f"Applied corrections to {corrections_applied} out of {len(df_corrected)} reactions")

    return df_corrected

def load_and_merge_nmr_hte_data(df):
    """Load NMR data for validation."""
    try:
        nmr_df = pd.read_csv('../data/nmr_rates_only.csv')
        df = df.merge(nmr_df, on=['acyl_chlorides', 'amines'], how='left')
        return df
    except FileNotFoundError:
        print("NMR validation file not found")
        return df

def validate_corrections(df_corrected, save_results=True, suffix="", selection_mode="each", n_features=2):
    """Validate the bias corrections with comprehensive statistical analysis."""
    print("=== Validating Corrections ===")
    
    # Load NMR data
    df = load_and_merge_nmr_hte_data(df_corrected)
    
    # Check if NMR data is available
    if 'NMR_rate' not in df.columns or df['NMR_rate'].isna().all():
        print("No NMR validation data available")
        return {}
    
    print(f"Found {df['NMR_rate'].notna().sum()} reactions with NMR validation data")
    
    # Create residual bias column for analysis
    df['residual_bias'] = df['predicted_bias'] - df['bias']
    
    # Create a figure for the main validation plots
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import os
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Original vs Corrected rates (all points)
    ax = axes[0, 0]
    ax.scatter(df["Controls"], df['HTE_rate'], 
              alpha=0.6, label='Original', color='blue', s=30)
    ax.scatter(df["Controls"], df['corrected_HTE_rate'], 
              alpha=0.6, label='Corrected', color='green', s=30)
    ax.plot([0.1, 1000], [0.1, 1000], 'k--', alpha=0.5, label='1:1 line')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Control Rate')
    ax.set_ylabel('HTE Rate')
    ax.set_title('Effect of Bias Correction (All Points)')
    ax.legend()
    
    # 2. Residual bias distribution (biased points only)
    ax = axes[0, 1]
    biased_mask = df['is_biased']
    if biased_mask.sum() > 0:
        residuals = df[biased_mask]['residual_bias']
        ax.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', label='Zero bias')
        ax.axvline(residuals.mean(), color='blue', linestyle='--', 
                   label=f'Mean: {residuals.mean():.2f}')
        ax.set_xlabel('Residual Bias')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residual Bias (Biased Points)')
        ax.legend()
    
    # 3. Predicted vs Actual bias (biased points)
    ax = axes[1, 0]
    biased_df = df[biased_mask]
    if len(biased_df) > 0:
        ax.scatter(biased_df['bias'], biased_df['predicted_bias'], alpha=0.6)
        
        # Add diagonal line
        max_bias = max(biased_df['bias'].max(), biased_df['predicted_bias'].max())
        ax.plot([0, max_bias], [0, max_bias], 'r--', alpha=0.5, label='Perfect prediction')
        
        # Add regression line
        z = np.polyfit(biased_df['bias'], biased_df['predicted_bias'], 1)
        p = np.poly1d(z)
        ax.plot(biased_df['bias'], p(biased_df['bias']), 'b--', alpha=0.8, 
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel('Actual Bias')
        ax.set_ylabel('Predicted Bias')
        ax.set_title('Bias Prediction Accuracy')
        ax.legend()
    
    # 4. Improvement metrics
    ax = axes[1, 1]
    ax.text(0.1, 0.9, 'Correction Metrics:', transform=ax.transAxes, 
            fontsize=14, fontweight='bold')
    
    # Calculate metrics for all points
    if len(biased_df) > 0:
        actual_bias = f"{biased_df['bias'].mean():.2f}"
        pred_bias = f"{biased_df['predicted_bias'].mean():.2f}"
        res_bias = f"{residuals.mean():.2f}"
        bias_reduction = f"{(1 - abs(residuals.mean()) / biased_df['bias'].mean()) * 100:.1f}%"
    else:
        actual_bias = "N/A"
        pred_bias = "N/A" 
        res_bias = "N/A"
        bias_reduction = "N/A"
    
    all_metrics_text = f"""
ALL POINTS METRICS:
Total entries: {len(df)}
Biased entries: {biased_mask.sum()}
Corrections applied: {df['correction_applied'].sum()}

Mean predicted bias (all): {df['predicted_bias'].mean():.2f}
Mean HTE rate before: {df['HTE_rate'].mean():.2f}
Mean HTE rate after: {df['corrected_HTE_rate'].mean():.2f}

BIASED POINTS METRICS:
Mean actual bias: {actual_bias}
Mean predicted bias: {pred_bias}
Mean residual bias: {res_bias}
Bias reduction: {bias_reduction}
"""
    
    ax.text(0.1, 0.1, all_metrics_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    if save_results:
        plt.savefig(f'plots/enhanced_validation_main{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Generate Final Publication-Quality Plot ---
    print("\nGenerating final publication-quality HTE vs NMR comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 7))

    # Clean data
    total_rates = len(df['HTE_rate'])
    total_biased_rates = len(df[df['is_biased'] == True])
    plot_df = df.copy()
    plot_df.dropna(subset=['NMR_rate', 'corrected_HTE_rate', 'HTE_rate', "Controls"], inplace=True)
    plot_df = plot_df[(plot_df['NMR_rate'] > 0) & (plot_df['HTE_rate'] > 0)]

    if plot_df.empty:
        print("No data available for plotting after cleaning.")
        return {}
    
    # Calculate log values
    plot_df['NMR_log_rate'] = np.log10(plot_df['NMR_rate'])
    plot_df['HTE_log_rate_corrected'] = np.log10(plot_df['corrected_HTE_rate'])
    plot_df['HTE_log_rate_original'] = np.log10(plot_df['HTE_rate'])

    # Using 'is_biased' to be consistent with how the original R^2 was calculated
    biased_mask = plot_df['is_biased']
    biased_data = plot_df[biased_mask]
    unbiased_data = plot_df[~biased_mask]

    # Define colors
    blue_color = '#3b5f8a'
    gray_color = '#aeaeae'
    green_color = '#2ca02c'
    red_color = 'red'

    # Plot original unbiased points
    ax.scatter(unbiased_data['NMR_log_rate'], unbiased_data['HTE_log_rate_original'], s=70,
               c=blue_color, alpha=0.9, edgecolor='black', marker='s', 
               label=f'Valid Points (n={len(unbiased_data)})')

    # Plot original biased points
    ax.scatter(biased_data['NMR_log_rate'], biased_data['HTE_log_rate_original'], s=70,
               c=gray_color, alpha=0.7, edgecolor='k', marker='s', 
               label=f'Biased Points (Original, n={len(biased_data)})')

    # Plot corrected biased points
    ax.scatter(biased_data['NMR_log_rate'], biased_data['HTE_log_rate_corrected'], s=70,
               c=green_color, alpha=0.9, edgecolor='black', marker='s', 
               label=f'Biased Points (Corrected, n={len(biased_data)})')

    # Add arrows
    for _, row in biased_data.iterrows():
        ax.annotate('',
                   xy=(row['NMR_log_rate'], row['HTE_log_rate_corrected']),
                   xytext=(row['NMR_log_rate'], row['HTE_log_rate_original']),
                   arrowprops=dict(arrowstyle='->', color=red_color, alpha=0.6, lw=1.5, mutation_scale=15))

    # --- Regression Lines and R² ---
    # Original regression (on unbiased data)
    r_squared_orig = 0
    if not unbiased_data.empty:
        unbiased_points = len(unbiased_data)
        slope_orig, intercept_orig, r_value_orig, _, _ = stats.linregress(
            unbiased_data['NMR_log_rate'], unbiased_data['HTE_log_rate_original'])
        r_squared_orig = r_value_orig**2
        x_orig = np.linspace(plot_df['NMR_log_rate'].min(), plot_df['NMR_log_rate'].max(), 100)
        ax.plot(x_orig, slope_orig * x_orig + intercept_orig, color=blue_color, linestyle='--', lw=2,
                label=f'Valid points Fit (R² = {r_squared_orig:.2f}, n={unbiased_points})\ny={slope_orig:.2f}x + {intercept_orig:.2f}')

    # Corrected regression (on all data)
    corr_plot_df = plot_df[plot_df['HTE_log_rate_corrected'].notna() & 
                          np.isfinite(plot_df['HTE_log_rate_corrected'])]
    num_points_on_plot = len(corr_plot_df)
    slope_corr, intercept_corr, r_value_corr, _, _ = stats.linregress(
        corr_plot_df['NMR_log_rate'], corr_plot_df['HTE_log_rate_corrected'])
    r_squared_corr = r_value_corr**2
    x_corr = np.linspace(corr_plot_df['NMR_log_rate'].min(), corr_plot_df['NMR_log_rate'].max(), 100)
    ax.plot(x_corr, slope_corr * x_corr + intercept_corr, color=green_color, linestyle='-', lw=2,
            label=f'Corrected Fit (R² = {r_squared_corr:.2f}, n={num_points_on_plot})\ny={slope_corr:.2f}x + {intercept_corr:.2f}')
    
    # Formatting
    ax.set_title(f'Comparison of HTE and NMR Rates (Original vs. Corrected)\n Trained on {total_biased_rates}/{total_rates} biased points', fontsize=16)
    ax.set_xlabel('NMR Log(rate) (M$^{-1}$s$^{-1}$)', fontsize=14)
    ax.set_ylabel('HTE Log(rate) (M$^{-1}$s$^{-1}$)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(-1, 4.5)
    ax.set_ylim(-1, 4.5)

    plt.tight_layout()
    os.makedirs('plots/reg_plots', exist_ok=True)
    plt.savefig(f'plots/reg_plots/measurable_HTE_NMR_comparison{suffix}.png', dpi=300)
    plt.close()
    
    # Calculate validation metrics
    validation_results = {
        'selection_mode': selection_mode,
        'n_features': n_features,
        'n_points_on_plot': num_points_on_plot,
        'n_points_total': total_rates,
        'r_squared_orig': r_squared_orig,
        'r_squared_corr': r_squared_corr,
        'r2_improvement': (r_squared_corr - r_squared_orig) / r_squared_orig * 100 if r_squared_orig > 0 else 0,
        'n_point_improvement': (num_points_on_plot - len(unbiased_data)) if not unbiased_data.empty else num_points_on_plot,
        'n_corrections_applied': df['correction_applied'].sum(),
        'avg_correction_magnitude': df[df['correction_applied']]['predicted_bias'].mean() if df['correction_applied'].sum() > 0 else 0,
        'max_correction_magnitude': df['predicted_bias'].max(),
        'correction_rate': df['correction_applied'].sum() / len(df)
    }
    
    print(f"Validation Results:")
    print(f"  Original R² vs NMR: {r_squared_orig:.3f}")
    print(f"  Corrected R² vs NMR: {r_squared_corr:.3f}")
    print(f"  R² Improvement: {validation_results['r2_improvement']:.1f}%")
    print(f"  Points on plot: {num_points_on_plot}")
    
    # Correction statistics
    print(f"Correction Statistics:")
    print(f"  Corrections applied: {validation_results['n_corrections_applied']}/{len(df)} ({validation_results['correction_rate']:.1%})")
    print(f"  Average correction: {validation_results['avg_correction_magnitude']:.3f}")
    print(f"  Maximum correction: {validation_results['max_correction_magnitude']:.3f}")
    
    # Save corrected data
    if save_results:
        output_file = f'results/corrected_hte_rates{suffix}.csv'
        df.to_csv(output_file, index=False)
        print(f"Corrected data saved to {output_file}")
    
    return validation_results

def test_model_with_scrambled_features(acid_feature_data, amine_feature_data, df, 
                                     selected_features, classifier, regressor, 
                                     scaler_class, scaler_reg, valid_features, 
                                     n_scrambling_trials=10, random_seed=42):
    """Test model robustness by scrambling features."""
    print("=== Running Feature Scrambling Test ===")
    
    from .model_building import create_combined_dataset
    import random
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get baseline performance
    combined_df, _ = create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features)
    
    # Classification baseline
    y_class = combined_df['Fast_unmeasurable'].astype(int)
    X_class = combined_df[valid_features]
    X_class_scaled = scaler_class.transform(X_class)
    
    baseline_class_predictions = classifier.predict(X_class_scaled)
    baseline_class_f1 = f1_score(y_class, baseline_class_predictions)
    
    # Regression baseline (only on biased cases)
    biased_df = combined_df[combined_df['is_biased'] == True]
    if len(biased_df) > 0:
        y_reg = biased_df['bias']
        X_reg = biased_df[valid_features]
        X_reg_scaled = scaler_reg.transform(X_reg)
        
        baseline_reg_predictions = regressor.predict(X_reg_scaled)
        baseline_reg_r2 = r2_score(y_reg, baseline_reg_predictions)
    else:
        baseline_reg_r2 = 0.0
    
    # Scrambling trials
    scrambled_class_f1_scores = []
    scrambled_reg_r2_scores = []
    
    for trial in range(n_scrambling_trials):
        # Create scrambled features
        X_class_scrambled = X_class.copy()
        for col in valid_features:
            X_class_scrambled[col] = np.random.permutation(X_class_scrambled[col])
        
        X_class_scrambled_scaled = scaler_class.transform(X_class_scrambled)
        
        # Test classification
        scrambled_class_pred = classifier.predict(X_class_scrambled_scaled)
        scrambled_f1 = f1_score(y_class, scrambled_class_pred)
        scrambled_class_f1_scores.append(scrambled_f1)
        
        # Test regression
        if len(biased_df) > 0:
            X_reg_scrambled = X_reg.copy()
            for col in valid_features:
                X_reg_scrambled[col] = np.random.permutation(X_reg_scrambled[col])
            
            X_reg_scrambled_scaled = scaler_reg.transform(X_reg_scrambled)
            scrambled_reg_pred = regressor.predict(X_reg_scrambled_scaled)
            scrambled_r2 = r2_score(y_reg, scrambled_reg_pred)
            scrambled_reg_r2_scores.append(scrambled_r2)
    
    results = {
        'baseline_class_f1': baseline_class_f1,
        'scrambled_class_f1_mean': np.mean(scrambled_class_f1_scores),
        'scrambled_class_f1_std': np.std(scrambled_class_f1_scores),
        'baseline_reg_r2': baseline_reg_r2,
        'scrambled_reg_r2_mean': np.mean(scrambled_reg_r2_scores),
        'scrambled_reg_r2_std': np.std(scrambled_reg_r2_scores)
    }
    
    print(f"Feature Scrambling Results:")
    print(f"  Classification F1 - Baseline: {baseline_class_f1:.3f}, Scrambled: {results['scrambled_class_f1_mean']:.3f} ± {results['scrambled_class_f1_std']:.3f}")
    print(f"  Regression R² - Baseline: {baseline_reg_r2:.3f}, Scrambled: {results['scrambled_reg_r2_mean']:.3f} ± {results['scrambled_reg_r2_std']:.3f}")
    
    return results

def test_model_with_y_scrambling(acid_feature_data, amine_feature_data, df, 
                                selected_features, valid_features,
                                n_scrambling_trials=10, random_seed=42):
    """Test model with scrambled target variables (Y-scrambling)."""
    print("=== Running Y-Scrambling Test ===")
    
    from .model_building import create_combined_dataset, build_classification_model, build_regression_model
    import random
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get combined dataset
    combined_df, _ = create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features)
    
    # Run scrambling trials
    class_f1_scores = []
    reg_r2_scores = []
    
    for trial in range(n_scrambling_trials):
        # Create scrambled targets
        scrambled_df = combined_df.copy()
        
        # Scramble classification target
        scrambled_df['Fast_unmeasurable'] = np.random.permutation(scrambled_df['Fast_unmeasurable'])
        
        # Scramble regression target (bias)
        scrambled_df['bias'] = np.random.permutation(scrambled_df['bias'])
        scrambled_df['is_biased'] = scrambled_df['bias'] > 0
        
        # Train models on scrambled data
        try:
            classifier, scaler_class, class_results = build_classification_model(
                scrambled_df, valid_features, hyperparameter_optimization=False
            )
            class_f1_scores.append(class_results.get('cv_f1_mean', 0))
        except:
            class_f1_scores.append(0)
        
        try:
            regressor, scaler_reg, reg_results = build_regression_model(
                scrambled_df, valid_features, hyperparameter_optimization=False
            )
            reg_r2_scores.append(reg_results.get('cv_r2_mean', 0))
        except:
            reg_r2_scores.append(0)
    
    # Calculate chance performance
    y_class = combined_df['Fast_unmeasurable'].astype(int)
    class_distribution = y_class.value_counts(normalize=True)
    chance_f1 = 2 * class_distribution[0] * class_distribution[1] / (class_distribution[0] + class_distribution[1])
    chance_r2 = 0.0  # R² for random predictions should be around 0
    
    results = {
        'class_f1_mean': np.mean(class_f1_scores),
        'class_f1_std': np.std(class_f1_scores),
        'reg_r2_mean': np.mean(reg_r2_scores),
        'reg_r2_std': np.std(reg_r2_scores),
        'chance_f1': chance_f1,
        'chance_r2': chance_r2
    }
    
    print(f"Y-Scrambling Results:")
    print(f"  Classification F1 - Scrambled: {results['class_f1_mean']:.3f} ± {results['class_f1_std']:.3f}, Chance: {chance_f1:.3f}")
    print(f"  Regression R² - Scrambled: {results['reg_r2_mean']:.3f} ± {results['reg_r2_std']:.3f}, Chance: {chance_r2:.3f}")
    
    return results

def compute_bias_metrics(df, group_col):
    """Compute bias metrics for different groups."""
    bias_stats = df.groupby(group_col).agg({
        'bias': ['count', 'mean', 'std', 'max'],
        'is_biased': 'sum'
    }).round(3)
    
    bias_stats.columns = ['total_reactions', 'mean_bias', 'std_bias', 'max_bias', 'biased_reactions']
    bias_stats['bias_frequency'] = bias_stats['biased_reactions'] / bias_stats['total_reactions']
    
    return bias_stats

def cross_validate_model_performance(acid_feature_data, amine_feature_data, df, 
                                   selected_features, cv_folds=5):
    """Perform cross-validation to assess model generalization."""
    print(f"=== Cross-Validation with {cv_folds} folds ===")
    
    from .model_building import create_combined_dataset, build_classification_model, build_regression_model
    from sklearn.model_selection import KFold
    
    # Get combined dataset
    combined_df, valid_features = create_combined_dataset(
        acid_feature_data, amine_feature_data, df, selected_features
    )
    
    if not valid_features:
        return {}
    
    # Prepare data
    X = combined_df[valid_features]
    y_class = combined_df['Fast_unmeasurable'].astype(int)
    
    # Cross-validation for classification
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    class_scores = []
    reg_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/{cv_folds}")
        
        # Split data
        train_df = combined_df.iloc[train_idx]
        val_df = combined_df.iloc[val_idx]
        
        # Train models
        classifier, scaler_class, class_results = build_classification_model(
            train_df, valid_features, hyperparameter_optimization=False
        )
        
        regressor, scaler_reg, reg_results = build_regression_model(
            train_df, valid_features, hyperparameter_optimization=False
        )
        
        # Validate classification
        X_val_class = val_df[valid_features]
        y_val_class = val_df['Fast_unmeasurable'].astype(int)
        X_val_class_scaled = scaler_class.transform(X_val_class)
        
        class_pred = classifier.predict(X_val_class_scaled)
        class_f1 = f1_score(y_val_class, class_pred)
        class_scores.append(class_f1)
        
        # Validate regression (on biased cases)
        val_biased = val_df[val_df['is_biased'] == True]
        if len(val_biased) > 0:
            X_val_reg = val_biased[valid_features]
            y_val_reg = val_biased['bias']
            X_val_reg_scaled = scaler_reg.transform(X_val_reg)
            
            reg_pred = regressor.predict(X_val_reg_scaled)
            reg_r2 = r2_score(y_val_reg, reg_pred)
            reg_scores.append(reg_r2)
    
    results = {
        'cv_class_f1_mean': np.mean(class_scores),
        'cv_class_f1_std': np.std(class_scores),
        'cv_reg_r2_mean': np.mean(reg_scores) if reg_scores else 0,
        'cv_reg_r2_std': np.std(reg_scores) if reg_scores else 0,
        'n_folds': cv_folds
    }
    
    print(f"Cross-Validation Results:")
    print(f"  Classification F1: {results['cv_class_f1_mean']:.3f} ± {results['cv_class_f1_std']:.3f}")
    print(f"  Regression R²: {results['cv_reg_r2_mean']:.3f} ± {results['cv_reg_r2_std']:.3f}")
    
    return results

def save_evaluation_report(validation_results, model_results, df_corrected, selected_features, 
                         selection_mode="manual", n_features=2, scrambling_results=None, 
                         y_scrambling_results=None, suffix=""):
    """Save comprehensive evaluation report similar to v2 script."""
    import os
    from sklearn.metrics import r2_score
    
    # Ensure results directory exists
    os.makedirs('results/reports', exist_ok=True)
    
    report_file = f'results/reports/hte_rates_report_{selection_mode}_{n_features}{suffix}.txt'
    
    with open(report_file, 'w') as f:
        f.write("Measurable HTE Rate Bias Correction Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("METHODOLOGY: Two-Stage Model\n")
        f.write("1. Part 1 (Binary Classifier): Predicts if reaction has measurable rate:\n")
        f.write("   - Class 0: Measurable rates (rate > 0, not biased)\n")
        f.write("   - Class 1: Fast unmeasurable rates (rate ≤ 0)\n")
        f.write("2. Part 2 (Regressor): Predicts bias magnitude for measurable but biased reactions.\n\n")
        
        # Dataset summary
        f.write(f"Total entries: {len(df_corrected)}\n")
        f.write(f"Fast unmeasurable rates (≤ 0): {df_corrected['Fast_unmeasurable'].sum() if 'Fast_unmeasurable' in df_corrected.columns else 'N/A'}\n")
        f.write(f"Biased entries (ground truth): {df_corrected['is_biased'].sum() if 'is_biased' in df_corrected.columns else 'N/A'}\n")
        f.write(f"Corrections applied: {df_corrected['correction_applied'].sum() if 'correction_applied' in df_corrected.columns else 'N/A'}\n\n")
        f.write(f"Number of features for {selection_mode} selection: {n_features}\n")
        f.write(f"Final features selected: {selected_features}\n\n")
        
        # --- Classification Performance Analysis ---
        f.write("--- CLASSIFICATION PERFORMANCE ANALYSIS ---\n")
        if 'classification' in model_results:
            class_results = model_results['classification']
            f.write(f"Best Model: {class_results.get('model', 'Unknown')}\n")
            f.write(f"Train Accuracy: {class_results.get('train_accuracy', 0):.3f}\n")
            f.write(f"Test Accuracy: {class_results.get('test_accuracy', 0):.3f}\n")
            f.write(f"CV F1 Score: {class_results.get('cv_f1_mean', 0):.3f} ± {class_results.get('cv_f1_std', 0):.3f}\n")
        else:
            f.write("Classification metrics not available.\n")
        f.write("\n")
        
        # --- Regression Performance ---
        f.write("--- REGRESSION PERFORMANCE (on biased data) ---\n")
        if 'regression' in model_results:
            reg_results = model_results['regression']
            f.write(f"Best Model: {reg_results.get('model', 'Unknown')}\n")
            f.write(f"Train R²: {reg_results.get('train_r2', 0):.3f}\n")
            f.write(f"Test R²: {reg_results.get('test_r2', 0):.3f}\n")
            f.write(f"CV R²: {reg_results.get('cv_r2_mean', 0):.3f} ± {reg_results.get('cv_r2_std', 0):.3f}\n")
        else:
            f.write("Regression metrics not available.\n")
        f.write("\n")
        
        # --- Regression Performance To Ground Truth ---
        f.write("--- REGRESSION PERFORMANCE TO NMR Rates ---\n")
        f.write(f"  Original R² vs NMR: {validation_results.get('r_squared_orig', 0):.3f}\n")
        f.write(f"  Corrected R² vs NMR: {validation_results.get('r_squared_corr', 0):.3f}\n")
        r2_improvement = validation_results.get('r2_improvement', 0)
        f.write(f"  R² Improvement: {r2_improvement:.1f}%\n")
        f.write(f"  Point Improvement: {validation_results.get('n_point_improvement', 0)}\n\n")
        f.write("\n")
        
        # --- Overall Correction Metrics ---
        f.write("--- OVERALL CORRECTION METRICS ---\n")
        if 'is_biased' in df_corrected.columns:
            biased_mask = df_corrected['is_biased']
            if biased_mask.sum() > 0:
                f.write("BIASED POINTS METRICS:\n")
                f.write(f"  Mean actual bias: {df_corrected[biased_mask]['bias'].mean():.2f}\n")
                if 'predicted_bias' in df_corrected.columns:
                    f.write(f"  Mean predicted bias: {df_corrected[biased_mask]['predicted_bias'].mean():.2f}\n")
                if 'residual_bias' in df_corrected.columns:
                    residual_bias_mean = df_corrected[biased_mask]['residual_bias'].mean()
                    actual_bias_mean = df_corrected[biased_mask]['bias'].mean()
                    f.write(f"  Mean residual bias: {residual_bias_mean:.2f}\n")
                    f.write(f"  Residual bias std: {df_corrected[biased_mask]['residual_bias'].std():.2f}\n")
                    if actual_bias_mean != 0:
                        bias_reduction = (1 - abs(residual_bias_mean) / actual_bias_mean) * 100
                        f.write(f"  Bias reduction: {bias_reduction:.1f}%\n")
                f.write("\n")
        
        f.write("ALL POINTS METRICS:\n")
        if 'predicted_bias' in df_corrected.columns:
            f.write(f"  Mean predicted bias: {df_corrected['predicted_bias'].mean():.2f}\n")
        f.write(f"  Mean HTE rate before correction: {df_corrected['HTE_rate'].mean():.2f}\n")
        if 'corrected_HTE_rate' in df_corrected.columns:
            f.write(f"  Mean HTE rate after correction: {df_corrected['corrected_HTE_rate'].mean():.2f}\n")
        f.write("\n")
        # Scrambling test results
        if scrambling_results:
            f.write("--- SCRAMBLED FEATURES VALIDATION TEST ---\n")
            f.write("(Tests if model is learning meaningful patterns vs. exploiting biases)\n\n")
            
            f.write("CLASSIFIER SCRAMBLING TEST:\n")
            class_drop = scrambling_results.get('baseline_class_f1', 0) - scrambling_results.get('scrambled_class_f1_mean', 0)
            f.write(f"  Baseline Test F1 Score: {scrambling_results.get('baseline_class_f1', 0):.3f}\n")
            f.write(f"  Scrambled Mean F1 Score: {scrambling_results.get('scrambled_class_f1_mean', 0):.3f} ± {scrambling_results.get('scrambled_class_f1_std', 0):.3f}\n")
            f.write(f"  Performance Drop: {class_drop:.3f}\n")
            
            if class_drop > 0.1:
                f.write("  ✅ RESULT: Model is learning meaningful patterns (substantial drop)\n")
            elif class_drop > 0.05:
                f.write("  ⚠️  RESULT: Model shows moderate learning (moderate drop)\n")
            else:
                f.write("  ❌ RESULT: Model may be exploiting biases (little drop)\n")
            
            f.write("\nREGRESSOR SCRAMBLING TEST:\n")
            reg_drop = scrambling_results.get('baseline_reg_r2', 0) - scrambling_results.get('scrambled_reg_r2_mean', 0)
            f.write(f"  Baseline Test R²: {scrambling_results.get('baseline_reg_r2', 0):.3f}\n")
            f.write(f"  Scrambled Mean R²: {scrambling_results.get('scrambled_reg_r2_mean', 0):.3f} ± {scrambling_results.get('scrambled_reg_r2_std', 0):.3f}\n")
            f.write(f"  Performance Drop: {reg_drop:.3f}\n")
            
            if reg_drop > 0.1:
                f.write("  ✅ RESULT: Model is learning meaningful patterns (substantial drop)\n")
            elif reg_drop > 0.05:
                f.write("  ⚠️  RESULT: Model shows moderate learning (moderate drop)\n")
            else:
                f.write("  ❌ RESULT: Model may be exploiting biases (little drop)\n")
            f.write("\n")
        
        # Y-scrambling test results
        if y_scrambling_results:
            f.write("--- Y-SCRAMBLING/Y-RANDOMIZATION VALIDATION TEST ---\n")
            f.write("(Tests if model architecture/features are too flexible for random targets)\n\n")
            
            f.write("CLASSIFIER Y-SCRAMBLING TEST:\n")
            class_above_chance = y_scrambling_results.get('class_f1_mean', 0) - y_scrambling_results.get('chance_f1', 0)
            f.write(f"  Expected chance F1 score: {y_scrambling_results.get('chance_f1', 0):.3f}\n")
            f.write(f"  Scrambled Mean F1 Score: {y_scrambling_results.get('class_f1_mean', 0):.3f} ± {y_scrambling_results.get('class_f1_std', 0):.3f}\n")
            f.write(f"  Performance above chance: {class_above_chance:.3f}\n")
            
            if class_above_chance < 0.05:
                f.write("  ✅ RESULT: Performance close to chance - appropriate architecture/features\n")
            elif class_above_chance < 0.15:
                f.write("  ⚠️  RESULT: Performance moderately above chance - may be too flexible\n")
            else:
                f.write("  ❌ RESULT: Performance substantially above chance - too flexible\n")
            
            f.write("\nREGRESSOR Y-SCRAMBLING TEST:\n")
            reg_above_chance = y_scrambling_results.get('reg_r2_mean', 0) - y_scrambling_results.get('chance_r2', 0)
            f.write(f"  Expected chance R²: {y_scrambling_results.get('chance_r2', 0):.3f}\n")
            f.write(f"  Scrambled Mean R²: {y_scrambling_results.get('reg_r2_mean', 0):.3f} ± {y_scrambling_results.get('reg_r2_std', 0):.3f}\n")
            f.write(f"  Performance above chance: {reg_above_chance:.3f}\n")
            
            if reg_above_chance < 0.1:
                f.write("  ✅ RESULT: Performance close to chance - appropriate architecture/features\n")
            elif reg_above_chance < 0.3:
                f.write("  ⚠️  RESULT: Performance moderately above chance - may be too flexible\n")
            else:
                f.write("  ❌ RESULT: Performance substantially above chance - too flexible\n")
            f.write("\n")
    
    print(f"Comprehensive evaluation report saved to {report_file}")

# Import required functions for compatibility
from sklearn.metrics import f1_score 