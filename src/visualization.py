#!/usr/bin/env python3
"""
Visualization Module
===================
Handles plotting and visual analysis of model results and data patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def visualize_model_performance(results, save_plot=False, suffix=""):
    """Create a comprehensive bar plot to compare model performances for both classification and regression."""
    if not results:
        return
    
    # Handle the nested structure - for backward compatibility, check if we have the old flat structure
    if 'classifier' in results and 'regressor' in results:
        # This is the v2 structure with multiple models per type
        class_results = results['classifier']
        reg_results = results['regressor']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))
        
        # --- Binary Classifier Performance Plot ---
        class_names = list(class_results.keys())
        class_scores = [res['cv_mean_f1'] for res in class_results.values()]
        
        ax = axes[0]
        bars = ax.bar(class_names, class_scores, color=sns.color_palette("husl", len(class_names)))
        ax.set_ylabel('Cross-Validation F1')
        ax.set_title('Classifier Model Performance Comparison')
        ax.set_ylim(bottom=max(0, min(class_scores) - 0.1), top=1.0) # Adjust y-axis
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

        # --- Regressor Performance Plot ---
        reg_names = list(reg_results.keys())
        cv_scores = [res['cv_mean_r2'] for res in reg_results.values()]
        val_scores = [res.get('val_r2', np.nan) for res in reg_results.values()]
        test_scores = [res.get('test_r2', np.nan) for res in reg_results.values()]
        loo_scores = [res.get('loo_mean_r2', np.nan) for res in reg_results.values()]

        ax = axes[1]
        x = np.arange(len(reg_names))  # the label locations
        width = 0.2  # the width of the bars (reduced to fit 4 bars)

        # Using four different color palettes to distinguish the bars
        rects1 = ax.bar(x - 1.5*width, cv_scores, width, label='K-Fold CV R²', color=sns.color_palette("viridis", len(reg_names)))
        rects2 = ax.bar(x - 0.5*width, val_scores, width, label='Val R²', color=sns.color_palette("plasma", len(reg_names)))
        rects3 = ax.bar(x + 0.5*width, test_scores, width, label='Test R²', color=sns.color_palette("inferno", len(reg_names)))
        rects4 = ax.bar(x + 1.5*width, loo_scores, width, label='LOO CV R²', color=sns.color_palette("cividis", len(reg_names)))

        ax.set_ylabel('R² Score')
        ax.set_title('Regressor Model Performance Comparison (K-Fold vs. Val vs. Test vs. LOO)')
        ax.set_xticks(x)
        ax.set_xticklabels(reg_names, rotation=15, ha="right")
        ax.legend()
        ax.axhline(0, color='black', linestyle='--', lw=1)

        for rect_group in [rects1, rects2, rects3, rects4]:
            for bar in rect_group:
                yval = bar.get_height()
                if np.isfinite(yval):
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', 
                            va='bottom' if yval >= 0 else 'top', ha='center', fontsize=7)

        # Adjust y-axis to better show the data
        all_scores = [s for s in cv_scores + val_scores + test_scores + loo_scores if np.isfinite(s)]
        if all_scores:
            min_score = min(all_scores)
            max_score = max(all_scores)
            y_margin = (max_score - min_score) * 0.1
            ax.set_ylim(min_score - y_margin, max_score + y_margin)
            
    else:
        # This is the new modular structure with single models
        class_results = results.get('classification', {})
        reg_results = results.get('regression', {})
        
        # Check what results we have and adjust subplot layout accordingly
        has_classification = bool(class_results and class_results.get('model', 'None') != 'None')
        has_regression = bool(reg_results and reg_results.get('model', 'None') != 'None')
        
        if has_classification and has_regression:
            # Both results available - use 2 subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            class_ax = axes[0]
            reg_ax = axes[1]
        elif has_regression and not has_classification:
            # Only regression results available - use 1 subplot
            fig, reg_ax = plt.subplots(1, 1, figsize=(12, 6))
            class_ax = None
        elif has_classification and not has_regression:
            # Only classification results available - use 1 subplot
            fig, class_ax = plt.subplots(1, 1, figsize=(12, 6))
            reg_ax = None
        else:
            # No results available
            print("No valid results to plot")
            return
        
        # --- Classification Performance ---
        if has_classification and class_ax is not None:
            class_metrics = {
                'Train Accuracy': class_results.get('train_accuracy', 0),
                'Test Accuracy': class_results.get('test_accuracy', 0),
                'CV F1 Score': class_results.get('cv_f1_mean', 0)
            }
            best_class_name = class_results.get('model', 'None')
            
            class_names = list(class_metrics.keys())
            class_values = list(class_metrics.values())
            bars = class_ax.bar(class_names, class_values, color=sns.color_palette("husl", len(class_names)))
            class_ax.set_ylabel('Score')
            class_ax.set_title(f'Classification Model Performance: {best_class_name}')
            class_ax.set_ylim(0, 1.1)
            
            for bar, value in zip(bars, class_values):
                height = bar.get_height()
                class_ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # --- Regression Performance ---
        if has_regression and reg_ax is not None:
            reg_metrics = {
                'Train R²': reg_results.get('train_r2', 0),
                'Test R²': reg_results.get('test_r2', 0),
                'CV R²': reg_results.get('cv_r2_mean', 0)
            }
            best_reg_name = reg_results.get('model', 'None')
            
            reg_names = list(reg_metrics.keys())
            reg_values = list(reg_metrics.values())
            bars = reg_ax.bar(reg_names, reg_values, color=sns.color_palette("viridis", len(reg_names)))
            reg_ax.set_ylabel('R² Score')
            reg_ax.set_title(f'Regression Model Performance: {best_reg_name}')
            
            # Adjust y-axis based on data range
            min_val = min(reg_values) if reg_values else 0
            max_val = max(reg_values) if reg_values else 1
            y_margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            reg_ax.set_ylim(min_val - y_margin, max_val + y_margin)
            reg_ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
            
            for bar, value in zip(bars, reg_values):
                height = bar.get_height()
                reg_ax.text(bar.get_x() + bar.get_width()/2., height + (y_margin * 0.1),
                        f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/model_performance_comparison{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_classification_performance(y_actual, y_predicted, 
                                       dataset_type="test", save_plot=False, suffix=""):
    """Create a comprehensive confusion matrix visualization for binary classification performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import os
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_actual, y_predicted)
    
    # Create figure with additional subplot for metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Confusion Matrix Heatmap ---
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Measurable (0)', 'Fast (1)'],
                yticklabels=['Measurable (0)', 'Fast (1)'])
    ax.set_title(f'Confusion Matrix ({dataset_type.title()})')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    
    # --- Class Distribution Comparison ---
    ax = axes[1]
    actual_counts = pd.Series(y_actual).value_counts()
    predicted_counts = pd.Series(y_predicted).value_counts()
    
    # Create a consistent structure for both classes (0 and 1 only)
    binary_classes = [0, 1]
    actual_values = [actual_counts.get(i, 0) for i in binary_classes]
    predicted_values = [predicted_counts.get(i, 0) for i in binary_classes]
    
    x = np.arange(len(binary_classes))
    width = 0.35
    
    ax.bar(x - width/2, actual_values, width, label='Actual', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, predicted_values, width, label='Predicted', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Rate Class')
    ax.set_ylabel('Count')
    ax.set_title(f'Class Distribution: Actual vs Predicted ({dataset_type.title()})')
    ax.set_xticks(x)
    ax.set_xticklabels(['Measurable (0)', 'Fast (1)'])
    ax.legend()
    
    # Add count labels on bars
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        ax.text(i - width/2, actual + max(actual_values) * 0.01, str(actual), 
                ha='center', va='bottom')
        ax.text(i + width/2, predicted + max(predicted_values) * 0.01, str(predicted), 
                ha='center', va='bottom')
    
    # --- Binary Classification Metrics ---
    ax = axes[2]
    ax.axis('off')  # Remove axes for text display
    
    # Convert to binary if needed (map any class > 1 to class 1)
    y_actual_binary = np.where(np.array(y_actual) > 1, 1, np.array(y_actual))
    y_predicted_binary = np.where(np.array(y_predicted) > 1, 1, np.array(y_predicted))
    
    accuracy = accuracy_score(y_actual_binary, y_predicted_binary)
    precision = precision_score(y_actual_binary, y_predicted_binary, average='binary', zero_division=0)
    recall = recall_score(y_actual_binary, y_predicted_binary, average='binary', zero_division=0)
    f1 = f1_score(y_actual_binary, y_predicted_binary, average='binary', zero_division=0)
    
    # Calculate specificity and sensitivity using binary confusion matrix
    cm_binary = confusion_matrix(y_actual_binary, y_predicted_binary)
    tn, fp, fn, tp = cm_binary.ravel() if cm_binary.size == 4 else (0, 0, 0, len(y_actual_binary))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    
    # Try to calculate AUC-ROC if possible
    try:
        auc_roc = roc_auc_score(y_actual_binary, y_predicted_binary)
        auc_text = f"AUC-ROC: {auc_roc:.3f}"
    except:
        auc_roc = 0.0
        auc_text = "AUC-ROC: N/A"
    
    # Create metrics text
    metrics_text = f"""Binary Classification Metrics:
    
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall (Sensitivity): {recall:.3f}
Specificity: {specificity:.3f}
F1-Score: {f1:.3f}
{auc_text}

Confusion Matrix Values:
True Negatives: {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives: {tp}"""
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax.set_title(f'Performance Metrics ({dataset_type.title()})')
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs('plots/class_plots', exist_ok=True)
        filename = f'plots/class_plots/binary_classification_performance_{dataset_type}{suffix}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Classification plot saved: {filename}")
    else:
        plt.show()
        
    class_results = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }
    
    return class_results

def plot_bias_corrections(df_corrected, save_plot=False, suffix=""):
    """Plot bias correction analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original vs Corrected rates
    corrected_data = df_corrected[df_corrected['correction_applied'] == True]
    
    if len(corrected_data) > 0:
        axes[0, 0].scatter(corrected_data['HTE_rate'], corrected_data['corrected_HTE_rate'], 
                          alpha=0.6, color='red', label='Corrected')
        
        # Add diagonal line
        min_val = min(corrected_data['HTE_rate'].min(), corrected_data['corrected_HTE_rate'].min())
        max_val = max(corrected_data['HTE_rate'].max(), corrected_data['corrected_HTE_rate'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[0, 0].set_xlabel('Original HTE Rate')
        axes[0, 0].set_ylabel('Corrected HTE Rate')
        axes[0, 0].set_title('Bias Corrections Applied')
        axes[0, 0].legend()
    
    # 2. Predicted bias distribution
    bias_data = df_corrected[df_corrected['predicted_bias'] > 0]
    if len(bias_data) > 0:
        axes[0, 1].hist(bias_data['predicted_bias'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Predicted Bias')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Predicted Bias')
    
    # 3. Bias by acyl chloride
    bias_by_acid = df_corrected.groupby('acyl_chlorides')['predicted_bias'].mean().sort_values(ascending=False)
    top_acids = bias_by_acid.head(10)
    
    if len(top_acids) > 0:
        axes[1, 0].barh(range(len(top_acids)), top_acids.values)
        axes[1, 0].set_yticks(range(len(top_acids)))
        axes[1, 0].set_yticklabels(top_acids.index)
        axes[1, 0].set_xlabel('Average Predicted Bias')
        axes[1, 0].set_title('Top 10 Acyl Chlorides by Bias')
    
    # 4. Correction effectiveness (if NMR data available)
    try:
        nmr_df = pd.read_csv('data/rates/nmr_rates_only.csv')
        df_with_nmr = df_corrected.merge(nmr_df, on=['acyl_chlorides', 'amines'], how='left')
        nmr_data = df_with_nmr[df_with_nmr['NMR_rate'].notna()]
        
        if len(nmr_data) > 0:
            # Plot NMR vs original and corrected
            axes[1, 1].scatter(nmr_data['NMR_rate'], nmr_data['HTE_rate'], 
                             alpha=0.6, color='blue', label='Original')
            axes[1, 1].scatter(nmr_data['NMR_rate'], nmr_data['corrected_HTE_rate'], 
                             alpha=0.6, color='red', label='Corrected')
            
            # Add diagonal and R² values
            min_val = nmr_data['NMR_rate'].min()
            max_val = nmr_data['NMR_rate'].max()
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            r2_orig = r2_score(nmr_data['NMR_rate'], nmr_data['HTE_rate'])
            r2_corr = r2_score(nmr_data['NMR_rate'], nmr_data['corrected_HTE_rate'])
            
            axes[1, 1].set_xlabel('NMR Rate')
            axes[1, 1].set_ylabel('HTE Rate')
            axes[1, 1].set_title(f'Validation vs NMR\nOriginal R²: {r2_orig:.3f}, Corrected R²: {r2_corr:.3f}')
            axes[1, 1].legend()
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No NMR validation data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('NMR Validation')
    
    except FileNotFoundError:
        axes[1, 1].text(0.5, 0.5, 'No NMR validation file found', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('NMR Validation')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/bias_corrections{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, model_type='classifier', save_plot=False, suffix=""):
    """Plot feature importance from trained model."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
    else:
        print("Model does not have feature importance information")
        return
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))  # Show top 15 features
    
    plt.figure(figsize=(10, 8))
    plt.title(f'{model_type.capitalize()} Feature Importance')
    plt.barh(range(top_n), importances[indices[:top_n]])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/feature_importance_{model_type}{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residuals_analysis(y_true, y_pred, save_plot=False, suffix=""):
    """Plot residuals analysis for regression model."""
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Predicted vs Actual
    axes[0].scatter(y_pred, y_true, alpha=0.6)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_title(f'Predicted vs Actual\nR² = {r2_score(y_true, y_pred):.3f}')
    
    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs Predicted')
    
    # 3. Residuals distribution
    axes[2].hist(residuals, bins=15, alpha=0.7)
    axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residuals Distribution')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/residuals_analysis{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_validation_summary(validation_results, save_plot=False, suffix=""):
    """Plot validation summary across different configurations."""
    
    if not validation_results or len(validation_results) == 0:
        return
    
    # Convert to DataFrame
    df_results = pd.DataFrame(validation_results)
    
    if 'n_features' not in df_results.columns:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. R² vs Number of Features
    if 'r2_corrected' in df_results.columns:
        axes[0, 0].plot(df_results['n_features'], df_results['r2_corrected'], 'o-', label='Corrected')
    if 'r2_original' in df_results.columns:
        axes[0, 0].plot(df_results['n_features'], df_results['r2_original'], 'o-', label='Original')
    
    axes[0, 0].set_xlabel('Number of Features')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].set_title('R² vs Number of Features')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Classification Performance
    if 'f1_score' in df_results.columns:
        axes[0, 1].plot(df_results['n_features'], df_results['f1_score'], 'o-', color='green')
        axes[0, 1].set_xlabel('Number of Features')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Classification F1 vs Number of Features')
        axes[0, 1].grid(True)
    
    # 3. Number of Corrections Applied
    if 'n_corrections' in df_results.columns:
        axes[1, 0].plot(df_results['n_features'], df_results['n_corrections'], 'o-', color='orange')
        axes[1, 0].set_xlabel('Number of Features')
        axes[1, 0].set_ylabel('Number of Corrections')
        axes[1, 0].set_title('Corrections Applied vs Number of Features')
        axes[1, 0].grid(True)
    
    # 4. Model Complexity vs Performance
    if 'r2_corrected' in df_results.columns:
        scatter = axes[1, 1].scatter(df_results['n_features'], df_results['r2_corrected'], 
                                   c=df_results.get('n_corrections', 0), cmap='viridis', s=100)
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('R² (Corrected)')
        axes[1, 1].set_title('Model Performance vs Complexity')
        axes[1, 1].grid(True)
        plt.colorbar(scatter, ax=axes[1, 1], label='Corrections Applied')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/validation_summary{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_scrambling_test_results(scrambling_results, save_plot=False, suffix=""):
    """Plot results from feature scrambling tests."""
    
    if not scrambling_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Classification results
    class_baseline = scrambling_results.get('baseline_class_f1', 0)
    class_scrambled_mean = scrambling_results.get('scrambled_class_f1_mean', 0)
    class_scrambled_std = scrambling_results.get('scrambled_class_f1_std', 0)
    
    ax1.bar(['Baseline', 'Scrambled'], [class_baseline, class_scrambled_mean], 
           yerr=[0, class_scrambled_std], capsize=5, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Classification: Baseline vs Scrambled Features')
    ax1.set_ylim(0, 1)
    
    # Add performance drop annotation
    drop = class_baseline - class_scrambled_mean
    ax1.text(0.5, 0.8, f'Drop: {drop:.3f}', transform=ax1.transAxes, 
            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Regression results
    reg_baseline = scrambling_results.get('baseline_reg_r2', 0)
    reg_scrambled_mean = scrambling_results.get('scrambled_reg_r2_mean', 0)
    reg_scrambled_std = scrambling_results.get('scrambled_reg_r2_std', 0)
    
    ax2.bar(['Baseline', 'Scrambled'], [reg_baseline, reg_scrambled_mean], 
           yerr=[0, reg_scrambled_std], capsize=5, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('R² Score')
    ax2.set_title('Regression: Baseline vs Scrambled Features')
    ax2.set_ylim(-0.2, 1)
    
    # Add performance drop annotation
    drop = reg_baseline - reg_scrambled_mean
    ax2.text(0.5, 0.8, f'Drop: {drop:.3f}', transform=ax2.transAxes, 
            ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'plots/scrambling_test_results{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


