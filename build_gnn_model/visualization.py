#!/usr/bin/env python3
"""
Simple parity plot for reaction rate predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os


def set_plot_style():
    """Set clean plotting style similar to GPG style."""
    plt.rcParams.update({
        "figure.figsize": (3.5, 3.2),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def calculate_metrics(y_true, y_pred):
    """Calculate R² and MAE metrics."""
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"R²: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    
    return r2, mae

def plot_parity(y_true_train, y_pred_train, y_true_val, y_pred_val, y_true_test, y_pred_test, 
                output_path="parity_plot.png", title="Reaction Rate Predictions"):
    """
    Create a parity plot comparing predicted vs true values for train, validation, and test sets.
    
    Args:
        y_true_train: True values for training set
        y_pred_train: Predicted values for training set
        y_true_val: True values for validation set
        y_pred_val: Predicted values for validation set
        y_true_test: True values for test set
        y_pred_test: Predicted values for test set
        output_path: Path to save the plot
        title: Plot title
    """
    set_plot_style()
    
    # Combine all data for axis limits
    all_true = np.concatenate([y_true_train, y_true_val, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_val, y_pred_test])
    
    # Set axis limits with padding
    y_min = min(all_true.min(), all_pred.min())
    y_max = max(all_true.max(), all_pred.max())
    pad = 0.1 * (y_max - y_min)
    y_lo = y_min - pad
    y_hi = y_max + pad
    
    fig, ax = plt.subplots()
    
    # 1:1 diagonal line
    ax.plot([y_lo, y_hi], [y_lo, y_hi], color="black", linewidth=1.0, zorder=1.5)
    
    # Trend lines (faded)
    try:
        # Training trend line
        if len(y_true_train) > 1:
            coeffs_train = np.polyfit(y_true_train, y_pred_train, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_train[0] * xs + coeffs_train[1], 
                   color="#B0B0B0", linewidth=0.8, alpha=0.25, zorder=1.0)
        
        # Validation trend line
        if len(y_true_val) > 1:
            coeffs_val = np.polyfit(y_true_val, y_pred_val, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_val[0] * xs + coeffs_val[1], 
                   color="#455C5C", linewidth=0.8, alpha=0.25, zorder=1.0)
        
        # Test trend line
        if len(y_true_test) > 1:
            coeffs_test = np.polyfit(y_true_test, y_pred_test, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_test[0] * xs + coeffs_test[1], 
                   color="#6380A6", linewidth=0.8, alpha=0.4, zorder=1.0)
    except:
        pass
    
    # Plot training points (gray squares)
    ax.scatter(y_true_train, y_pred_train, 
              s=4.5, marker="s", facecolors="#D0D0D0", edgecolors="#D0D0D0",
              linewidths=0.5, zorder=3.0, alpha=0.7)
    
    # Plot validation points (orange squares #ff7f0e)
    ax.scatter(y_true_val, y_pred_val,
              s=6, marker="s", facecolors="#8AA2A2", edgecolors="#8AA2A2", 
              linewidths=0.5, zorder=3.0, alpha=0.8)
    
    # Plot test points (red squares #d62728)
    ax.scatter(y_true_test, y_pred_test,
              s=6, marker="s", facecolors="#6380A6", edgecolors="#6380A6", 
              linewidths=0.5, zorder=3.0, alpha=0.8)
    
    # Set limits and ticks
    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(y_lo, y_hi)
    
    # Labels
    ax.set_xlabel("True ln k")
    ax.set_ylabel("Predicted ln k")
    
    # Grid
    ax.set_axisbelow(True)
    ax.grid(True, color="#ececec", linewidth=0.6, zorder=0.1)
    
    # Calculate and display metrics
    r2_train, mae_train = calculate_metrics(y_true_train, y_pred_train)
    r2_val, mae_val = calculate_metrics(y_true_val, y_pred_val)
    r2_test, mae_test = calculate_metrics(y_true_test, y_pred_test)
    
    # Legend with metrics
    handles = [
        Line2D([0], [0], marker='s', linestyle='None', 
               markerfacecolor="#D0D0D0", markeredgecolor="#D0D0D0", markersize=4,
               label=f"train  R²={r2_train:.2f}   MAE={mae_train:.2f}"),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor="#8AA2A2", markeredgecolor="#8AA2A2", markersize=4, 
               label=f"val    R²={r2_val:.2f}   MAE={mae_val:.2f}"),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor="#6380A6", markeredgecolor="#6380A6", markersize=4, 
               label=f"test   R²={r2_test:.2f}   MAE={mae_test:.2f}")
    ]
    
    ax.legend(handles=handles, loc='lower right', frameon=False,
             handletextpad=0.3, labelspacing=0.3, borderaxespad=0.25,
             borderpad=0.2, markerscale=0.8, handlelength=0.9)
    
    # Save plot
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\no Saved parity plot: {output_path}")
    
    return r2_train, mae_train, r2_val, mae_val, r2_test, mae_test

def plot_parity_mae(y_true_train, y_pred_train, y_true_val, y_pred_val, y_true_test, y_pred_test, 
                output_path="parity_plot_mae.png", title="Reaction Rate Predictions"):
    set_plot_style()
    
    # Combine all data for axis limits
    all_true = np.concatenate([y_true_train, y_true_val, y_true_test])
    all_pred = np.concatenate([y_pred_train, y_pred_val, y_pred_test])
    
    # Set axis limits with padding
    y_min = min(all_true.min(), all_pred.min())
    y_max = max(all_true.max(), all_pred.max())
    pad = 0.1 * (y_max - y_min)
    y_lo = y_min - pad
    y_hi = y_max + pad
    
    fig, ax = plt.subplots()
    
    # 1:1 diagonal line
    ax.plot([y_lo, y_hi], [y_lo, y_hi], color="black", linewidth=1.0, zorder=1.5)
    
    # Trend lines (faded)
    try:
        # Training trend line
        if len(y_true_train) > 1:
            coeffs_train = np.polyfit(y_true_train, y_pred_train, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_train[0] * xs + coeffs_train[1], 
                   color="#B0B0B0", linewidth=0.8, alpha=0.25, zorder=1.0)
        
        # Validation trend line
        if len(y_true_val) > 1:
            coeffs_val = np.polyfit(y_true_val, y_pred_val, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_val[0] * xs + coeffs_val[1], 
                   color="#829292", linewidth=0.8, alpha=0.25, zorder=1.0)
        
        # Test trend line
        if len(y_true_test) > 1:
            coeffs_test = np.polyfit(y_true_test, y_pred_test, 1)
            xs = np.linspace(y_lo, y_hi, 200)
            ax.plot(xs, coeffs_test[0] * xs + coeffs_test[1], 
                   color="#285A92", linewidth=0.8, alpha=0.4, zorder=1.0)
    except:
        pass
    
    # Plot training points (gray squares)
    ax.scatter(y_true_train, y_pred_train, 
              s=18, marker="s", facecolors="#D0D0D0", edgecolors="#D0D0D0",
              linewidths=0.5, zorder=3.0, alpha=0.7)
    
    # Plot validation points (orange squares #ff7f0e)
    ax.scatter(y_true_val, y_pred_val,
              s=20, marker="s", facecolors="#8AA2A2", edgecolors="#8AA2A2", 
              linewidths=0.5, zorder=3.0, alpha=0.8)
    
    # Plot test points (red squares #d62728)
    ax.scatter(y_true_test, y_pred_test,
              s=20, marker="s", facecolors="#6380A6", edgecolors="#6380A6", 
              linewidths=0.5, zorder=3.0, alpha=0.8)
    
    # Set limits and ticks
    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(y_lo, y_hi)
    
    # Labels
    ax.set_xlabel("True ln k")
    ax.set_ylabel("Predicted ln k")
    
    # Grid
    ax.set_axisbelow(True)
    ax.grid(True, color="#ececec", linewidth=0.6, zorder=0.1)
    
    # Calculate and display metrics
    r2_train, mae_train = calculate_metrics(y_true_train, y_pred_train)
    r2_val, mae_val = calculate_metrics(y_true_val, y_pred_val)
    r2_test, mae_test = calculate_metrics(y_true_test, y_pred_test)
    
    # Legend with metrics
    handles = [
        Line2D([0], [0], marker='s', linestyle='None', 
               markerfacecolor="#D0D0D0", markeredgecolor="#D0D0D0", markersize=4,
               label=f"train  MAE={mae_train:.2f}"),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor="#8AA2A2", markeredgecolor="#8AA2A2", markersize=4, 
               label=f"val    MAE={mae_val:.2f}"),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor="#6380A6", markeredgecolor="#6380A6", markersize=4, 
               label=f"test   MAE={mae_test:.2f}")
    ]
    
    ax.legend(handles=handles, loc='lower right', frameon=False,
             handletextpad=0.3, labelspacing=0.3, borderaxespad=0.25,
             borderpad=0.2, markerscale=0.8, handlelength=0.9)
    
    # Save plot
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\no Saved parity plot: {output_path}")
    
    return r2_train, mae_train, r2_val, mae_val, r2_test, mae_test

def plot_high_low_fidelity(df, column_name_hte, column_name_nmr, corrected=False,
                           output_path="plot_high_low_fidelity.png", title="Comparison of HTE and NMR Rates"):
    
    set_plot_style()
    fig, ax = plt.subplots()
    df.dropna(subset=[column_name_hte, column_name_nmr], inplace=True)
    
    if corrected:
        mask_faster_blank = df['classified_rate'] == 1
    else:
        mask_faster_blank = df['Slow_unreliable'] == True
    
    # Set proper axis limits based on data range
    x_min, x_max = -1, 3  # Adjusted to cover min: -0.8901 to max: 2.6537
    y_min, y_max = df[column_name_hte].min() - 0.5, df[column_name_hte].max() + 0.5
    
    # Plot points with classified_rate = 1 in gray
    ax.scatter(df[mask_faster_blank][column_name_nmr], df[mask_faster_blank][column_name_hte], 
              s=30, marker="s", facecolors="#D0D0D0", edgecolors="#D0D0D0",
              linewidths=0.5, zorder=3.0, alpha=0.7) #, label='HTE rate < Control rate')
    # ax.scatter(df[mask_faster_blank][column_name_nmr], df[mask_faster_blank][column_name_hte], 
    #            s=50, alpha=0.4, marker='s',
    #            color='#D0D0D0', edgecolor='black', label='HTE rate < Control rate')
    
    # Plot points with classified_rate = 0 with color
    # scatter = ax.scatter(df[~mask_faster_blank][column_name_nmr], df[~mask_faster_blank][column_name_hte], 
    #                      s=50, c="#285A92", alpha=0.8, edgecolor='black', marker='s')
    ax.scatter(df[~mask_faster_blank][column_name_nmr], df[~mask_faster_blank][column_name_hte], 
              s=30, marker="s", facecolors="#6380A6", edgecolors="black", 
              linewidths=0.5, zorder=3.0, alpha=0.8) #, label=f'Num Rates = {len(df[~mask_faster_blank])}', )
    

    df_no_faster_blank = df[~mask_faster_blank]
    slope, intercept = np.polyfit(df_no_faster_blank[column_name_nmr], df_no_faster_blank[column_name_hte], 1)
    x_min_data = df_no_faster_blank[column_name_nmr].min()
    x_max_data = df_no_faster_blank[column_name_nmr].max()
    
    # Calculate R-squared
    correlation_matrix = np.corrcoef(df_no_faster_blank[column_name_nmr], df_no_faster_blank[column_name_hte])
    r_squared = correlation_matrix[0, 1]**2
    
    # Generate x values for the line that span only the data points
    x_line = np.linspace(x_min_data, x_max_data, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black', alpha=0.5, linewidth=2,) #linestyle='--', 
            #label=f'R² = {r_squared:.2f}\ny = {slope:.2f}x + {intercept:.2f}')
    
    # Legend with metrics
    handles = [
        Line2D([0], [0], marker='s', linestyle='None', 
               markerfacecolor="#D0D0D0", markeredgecolor="#D0D0D0", markersize=4,
               label='PRISM rate < Control rate'),
        # Line2D([0], [0], marker='s', linestyle='None',
        #        markerfacecolor="#6380A6", markeredgecolor="#6380A6", markersize=4, 
        #        label=f'Num Rates = {len(df[~mask_faster_blank])}'),
        Line2D([0], [0], linestyle='-', color='black', alpha=0.5, linewidth=2,
               label=f'R² = {r_squared:.2f}\ny = {slope:.2f}x + {intercept:.2f}')
    ]
    if corrected:
        handles.append(Line2D([0], [0], marker='s', linestyle='None',
                    markerfacecolor="#6380A6", markeredgecolor="#6380A6", markersize=4, 
                    label=f'Num Rates = {len(df[~mask_faster_blank])}'))
    ax.legend(handles=handles, loc='upper left', frameon=False,
             handletextpad=0.3, labelspacing=0.3, borderaxespad=0.25,
             borderpad=0.5, markerscale=0.8, handlelength=0.9, fontsize=8)
    
    # Set axis limits
    ax.set_xlim(x_min, x_max+1)
    ax.set_ylim(y_min, y_max)
    
    # Set labels and title
    num_of_points = len(df[column_name_hte].dropna().tolist())
    ax.set_title(title)
    ax.set_xlabel('NMR Log(rate) (M$^{-1}$s$^{-1}$)')
    ax.set_ylabel('PRISM Log(rate) (M$^{-1}$s$^{-1}$)')
    
    # Save plot
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\no Saved high-low fidelity plot: {output_path}")
    
    return r_squared, slope, intercept


def plot_training_curves(train_losses, val_losses, output_path="training_curves.png"):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        output_path: Path to save the plot
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation losses
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"\no Saved training curves: {output_path}")


def plot_hyperparameter_importance(study, output_path="hyperparameter_importance.png"):
    """
    Plot hyperparameter importance from Optuna study.
    
    Args:
        study: Optuna study object
        output_path: Path to save the plot
    """
    try:
        import optuna
        
        set_plot_style()
        
        # Get parameter importance
        importance = optuna.importance.get_param_importances(study)
        
        if not importance:
            print("No parameter importance data available")
            return
            
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        params = list(importance.keys())
        values = list(importance.values())
        
        # Create horizontal bar plot
        bars = ax.barh(params, values, alpha=0.7, color='steelblue')
        
        # Formatting
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', ha='left', fontsize=9)
        
        # Save plot
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"\no Saved hyperparameter importance: {output_path}")
        
    except ImportError:
        print("Optuna not available for importance plotting")
    except Exception as e:
        print(f"Error plotting hyperparameter importance: {e}")


def plot_optimization_history(study, output_path="optimization_history.png"):
    """
    Plot optimization history from Optuna study.
    
    Args:
        study: Optuna study object
        output_path: Path to save the plot
    """
    try:
        set_plot_style()
        
        # Get trial data
        trials = study.trials
        if not trials:
            print("No trials data available")
            return
            
        trial_numbers = [t.number for t in trials if t.value is not None]
        values = [t.value for t in trials if t.value is not None]
        
        if not values:
            print("No completed trials with values")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Objective value per trial
        ax1.plot(trial_numbers, values, 'b-', alpha=0.6, linewidth=1)
        ax1.scatter(trial_numbers, values, alpha=0.7, s=20, color='blue')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Objective Value (Validation Loss)')
        ax1.set_title('Optimization History')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best value so far
        best_values = []
        best_so_far = float('inf')
        for val in values:
            if val < best_so_far:
                best_so_far = val
            best_values.append(best_so_far)
            
        ax2.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
        ax2.set_xlabel('Trial')
        ax2.set_ylabel('Best Objective Value')
        ax2.set_title('Best Value Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Save plot
        fig.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"\no Saved optimization history: {output_path}")
        
    except Exception as e:
        print(f"Error plotting optimization history: {e}")
