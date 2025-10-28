#!/usr/bin/env python3
"""
Hyperparameter optimization for AmidePredictor using Optuna.
Optimizes learning rate, batch size, model architecture, and other key parameters.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from datetime import datetime
from utils.ngdataset import AMIDEDataset, NamedSampler
from graph_model_v3 import AmidePredictor
from visualization import plot_parity, plot_training_curves, plot_hyperparameter_importance, plot_optimization_history

# Set precision for faster training
torch.set_float32_matmul_precision('high')

def mse_loss(true, pred):
    """Standard Mean Squared Error loss function."""
    return torch.mean((true - pred) ** 2)

def get_feature_columns(addn_features, include_all_molecular=True):
    """
    Get the feature columns for model training.
    
    Args:
        addn_features: List of additional features to include
        include_all_molecular: If True, includes all molecular features (amine, acid, int with a, q, aim).
                              If False, includes only AIM features (amine_aim, acid_aim, int_aim).
    
    Returns:
        List of feature column names
    """
    if include_all_molecular:
        # Include all molecular features: a (adjacency), q (charges), aim (AIM features)
        base_features = [
            'amine_a', 'amine_q', 'amine_aim', 
            'acid_a', 'acid_q', 'acid_aim', 
            'int_a', 'int_q', 'int_aim'
        ]
    else:
        # Include only AIM features
        base_features = ['amine_aim', 'acid_aim', 'int_aim']
    
    return base_features + addn_features

def create_run_directory():
    """Create a timestamped directory for this optimization run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"optimization_runs/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{run_dir}/plots", exist_ok=True)
    os.makedirs(f"{run_dir}/model", exist_ok=True)
    os.makedirs(f"{run_dir}/predictions", exist_ok=True)
    
    return run_dir

def get_device():
    """Get the best available device."""
    # Always use float32 for compatibility
    torch.set_default_dtype(torch.float32)
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def load_datasets():
    """Load train, validation, and test datasets."""
    train_ds = AMIDEDataset()
    train_ds.load_h5('splits/train.h5')
    
    val_ds = AMIDEDataset()
    val_ds.load_h5('splits/val.h5')
    
    test_ds = AMIDEDataset()
    test_ds.load_h5('splits/test.h5')
    
    return train_ds, val_ds, test_ds

def train_model(trial, train_ds, val_ds, device, n_epochs=50, verbose=False):
    """
    Train a model with hyperparameters suggested by Optuna trial.
    
    Args:
        trial: Optuna trial object
        train_ds: Training dataset
        val_ds: Validation dataset
        device: Device to train on
        n_epochs: Number of epochs to train
        verbose: Whether to print training progress
    
    Returns:
        Best validation loss achieved
    """
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    n_graph_layers = trial.suggest_int('n_graph_layers', 2, 5)
    n_output_layers = trial.suggest_int('n_output_layers', 1, 3)
    samples_per_epoch = trial.suggest_int('samples_per_epoch', 50, 300, step=50)
    
    from utils.feature_utils import get_additional_features
    json_file = "data/hte-all-corrected_splits_train_val_tests_lnk.json"  #### SELECT FILE HERE
    addn_features = get_additional_features(json_file)
    if len(addn_features) > 10:
        print(f"Using {len(addn_features)} additional features")
    else:
        print(f"Using {len(addn_features)} additional features: {addn_features}")

    x = get_feature_columns(addn_features, include_all_molecular=True)
    # Alternative: x = get_feature_columns(addn_features, include_all_molecular=False)  # Only AIM features
    y = ['rate']
    
    # Calculate number of additional features for model architecture
    n_additional_features = len(addn_features)
    
    # Model architecture
    n_dim = train_ds[train_ds.keys()[0]]['amine_aim'].shape[-1] * 2 + 1  # 513 (256 + 256 + 1)
    # n_dim = train_ds[train_ds.keys()[0]]['amine_aim'].shape[-1] #using only AIM features (256 dim each)
    
    model = AmidePredictor(
        graph_in_dim=n_dim,
        n_graph_layers=n_graph_layers,
        n_output_layers=n_output_layers,
        use_control=True,
        n_additional_features=n_additional_features
    ).to(device)
    
    model = torch.jit.script(model)
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
    
    train_sampler = NamedSampler(train_ds)
    val_sampler = NamedSampler(val_ds)

    train_loader = train_ds.get_loader(train_sampler, x=x, y=y, num_workers=0, pin_memory=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for _ in range(samples_per_epoch):
            data = next(iter(train_loader))
            
            # Move data to device
            for key in x:
                data[key] = data[key].squeeze(0).to(torch.float32).to(device)
            
            # Forward pass
            output = model(data)
            loss = mse_loss(output['rate'].to(torch.float32).to(device), output['pred_rate'].to(torch.float32).to(device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for k, d in val_ds.items():
                data = create_data_dict(d, device, addn_features)
                
                output = model(data)
                loss = mse_loss(output['rate'].to(torch.float32), output['pred_rate'].to(torch.float32))
                epoch_val_losses.append(loss.item())
        
        # Record losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Report intermediate value for pruning
        trial.report(avg_val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    return best_val_loss, train_losses, val_losses

def create_data_dict(d, device, addn_features):
    """Helper function to create data dictionary with all features"""
    n = d['acid_aim'].shape[0]
    nn = d['amine_aim'].shape[0]
    nnn = d['int_aim'].shape[0]
    
    # Base molecular features
    data = {
        'acid_a': torch.tensor(d['acid_a'], dtype=torch.float32).reshape(n, -1).to(device),
        'acid_q': torch.tensor(d['acid_q'], dtype=torch.float32).to(device),
        'acid_aim': torch.tensor(d['acid_aim'], dtype=torch.float32).reshape(n, -1).to(device),
        'amine_a': torch.tensor(d['amine_a'], dtype=torch.float32).reshape(nn, -1).to(device),
        'amine_q': torch.tensor(d['amine_q'], dtype=torch.float32).to(device),
        'amine_aim': torch.tensor(d['amine_aim'], dtype=torch.float32).reshape(nn, -1).to(device),
        'int_a': torch.tensor(d['int_a'], dtype=torch.float32).reshape(nnn, -1).to(device),
        'int_q': torch.tensor(d['int_q'], dtype=torch.float32).to(device),
        'int_aim': torch.tensor(d['int_aim'], dtype=torch.float32).reshape(nnn, -1).to(device),
        'rate': torch.tensor(d['rate'], dtype=torch.float32).to(device)
    }
    
    # Add all additional features dynamically
    for feature in addn_features:
        if feature in d:
            data[feature] = torch.tensor(d[feature], dtype=torch.float32).to(device)
    
    return data

def objective(trial):
    """Objective function for Optuna optimization."""
    device = get_device()
    train_ds, val_ds, test_ds = load_datasets()
    
    try:
        best_val_loss, _, _ = train_model(trial, train_ds, val_ds, device, n_epochs=100)
        return best_val_loss
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

def run_optimization(n_trials=50, study_name="amide_predictor_optimization"):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
    
    Returns:
        study: Optuna study object
        best_trial: Best trial object
        run_dir: Directory path for this run
    """
    # Create timestamped directory for this run
    run_dir = create_run_directory()
    
    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    print(f"Using device: {get_device()}")
    print(f"Results will be saved to: {run_dir}")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print("\no Optimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    
    print("\no Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value:.6f}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results to timestamped directory
    # Save best parameters
    with open(f'{run_dir}/best_params.json', 'w') as f:
        json.dump(best_trial.params, f, indent=2)
    
    # Save all trial results
    trial_data = []
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        trial_data.append(trial_info)
    
    with open(f'{run_dir}/all_trials.json', 'w') as f:
        json.dump(trial_data, f, indent=2)
    
    # Create plots
    plot_optimization_history(study, f"{run_dir}/plots/optimization_history.png")
    plot_hyperparameter_importance(study, f"{run_dir}/plots/hyperparameter_importance.png")
    
    return study, best_trial, run_dir

def train_best_model(best_params, run_dir, n_epochs=100):
    """
    Train the final model with the best hyperparameters.
    
    Args:
        best_params: Dictionary of best hyperparameters
        run_dir: Directory to save results
        n_epochs: Number of epochs for final training
    """
    print(f"\nTraining final model with best parameters for {n_epochs} epochs...")
    
    device = get_device()
    train_ds, val_ds, test_ds = load_datasets()
    
    # Dynamically extract additional features from JSON data
    from utils.feature_utils import get_additional_features
    json_file = "data/hte-all-corrected_splits_train_val_tests_lnk.json"  #### SELECT FILE HERE
    addn_features = get_additional_features(json_file)

    n_additional_features = len(addn_features)
    
    # Create model with best parameters
    n_dim = train_ds[train_ds.keys()[0]]['amine_aim'].shape[-1] * 2 + 1  # 513 (256 + 256 + 1)
    # n_dim = train_ds[train_ds.keys()[0]]['amine_aim'].shape[-1]  # Should be 256 for AIM features
    model = AmidePredictor(
        graph_in_dim=n_dim,
        n_graph_layers=best_params['n_graph_layers'],
        n_output_layers=best_params['n_output_layers'],
        use_control=True,
        n_additional_features=n_additional_features
    ).to(device)
    
    model = torch.jit.script(model)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], amsgrad=True)
    
    # Set up data features
    x = get_feature_columns(addn_features, include_all_molecular=True)
    # Alternative: x = get_feature_columns(addn_features, include_all_molecular=False)  # Only AIM features
    y = ['rate']
    
    train_sampler = NamedSampler(train_ds)
    train_loader = train_ds.get_loader(train_sampler, x=x, y=y, num_workers=0, pin_memory=True)
    
    # Training with progress tracking
    train_losses = []
    val_losses = []
    
    print("\no Training progress:")
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for _ in range(best_params['samples_per_epoch']):
            data = next(iter(train_loader))
            
            for key in x:
                data[key] = data[key].squeeze(0).to(torch.float32).to(device)
            
            output = model(data)
            loss = mse_loss(output['rate'].to(torch.float32).to(device), output['pred_rate'].to(torch.float32).to(device))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for k, d in val_ds.items():
                data = create_data_dict(d, device, addn_features)
                
                output = model(data)
                loss = mse_loss(output['rate'].to(torch.float32), output['pred_rate'].to(torch.float32))
                epoch_val_losses.append(loss.item())
        
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Save final model
    model.save(f'{run_dir}/model/optimized_final.jpt')
    print(f"\no Saved optimized model: {run_dir}/model/optimized_final.jpt")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, f"{run_dir}/plots/final_training_curves.png")
    
    # Generate final predictions and parity plot
    print("\no Generating final predictions...")
    
    # Get predictions for all datasets
    def get_predictions(dataset, model, device):
        model.eval()
        true_vals, pred_vals, sample_names = [], [], []
        
        with torch.no_grad():
            for k, d in dataset.items():
                n = d['acid_aim'].shape[0]
                nn = d['amine_aim'].shape[0]
                nnn = d['int_aim'].shape[0]
                
                data = create_data_dict(d, device, addn_features)
                
                output = model(data)
                true_vals.append(output['rate'].cpu().numpy())
                pred_vals.append(output['pred_rate'].cpu().numpy())
                sample_names.append(k)
        
        return np.concatenate(true_vals).flatten(), np.concatenate(pred_vals).flatten(), sample_names
    
    train_true, train_pred, train_names = get_predictions(train_ds, model, device)
    val_true, val_pred, val_names = get_predictions(val_ds, model, device)
    test_true, test_pred, test_names = get_predictions(test_ds, model, device)
    
    # Save predictions to CSV
    def save_predictions_to_csv(true_vals, pred_vals, sample_names, split_name, run_dir):
        """Save predictions to CSV file with sample names and true HTE lnk values."""
        predictions_data = []
        
        for i, sample_name in enumerate(sample_names):
            predictions_data.append({
                'sample_name': sample_name,
                'true_hte_lnk': true_vals[i],
                'predicted_hte_lnk': pred_vals[i],
                'split': split_name
            })
        
        df = pd.DataFrame(predictions_data)
        csv_path = f"{run_dir}/predictions/{split_name}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {split_name} predictions to {csv_path}")
        return df
    
    print("\no Saving predictions to CSV files...")
    train_df = save_predictions_to_csv(train_true, train_pred, train_names, 'train', run_dir)
    val_df = save_predictions_to_csv(val_true, val_pred, val_names, 'val', run_dir)
    test_df = save_predictions_to_csv(test_true, test_pred, test_names, 'test', run_dir)
    
    # Combine all predictions into a single CSV file
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_csv_path = f"{run_dir}/predictions/all_predictions.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"  Saved combined predictions to {combined_csv_path}")
    
    # Create final parity plot
    r2_train, mae_train, r2_val, mae_val, r2_test, mae_test = plot_parity(
        train_true, train_pred, val_true, val_pred, test_true, test_pred,
        output_path=f"{run_dir}/plots/optimized_parity_plot.png",
        title="Optimized Amide Reaction Rate Predictions"
    )
    
    print(f"\no Final Model Performance:")
    print(f"  Training    - R²: {r2_train:.3f}, MAE: {mae_train:.3f}")
    print(f"  Validation  - R²: {r2_val:.3f}, MAE: {mae_val:.3f}")
    print(f"  Test        - R²: {r2_test:.3f}, MAE: {mae_test:.3f}")
    
    # Save final metrics
    final_metrics = {
        'train': {'r2': float(r2_train), 'mae': float(mae_train)},
        'val': {'r2': float(r2_val), 'mae': float(mae_val)},
        'test': {'r2': float(r2_test), 'mae': float(mae_test)},
        'best_params': best_params
    }
    
    with open(f'{run_dir}/final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Create a summary file for easy reference
    summary = {
        'run_directory': run_dir,
        'timestamp': datetime.now().isoformat(),
        'n_epochs': n_epochs,
        'best_params': best_params,
        'final_performance': {
            'train_r2': float(r2_train),
            'train_mae': float(mae_train),
            'val_r2': float(r2_val),
            'val_mae': float(mae_val),
            'test_r2': float(r2_test),
            'test_mae': float(mae_test)
        },
        'files_created': [
            'best_params.json',
            'all_trials.json',
            'final_metrics.json',
            'plots/optimization_history.png',
            'plots/hyperparameter_importance.png',
            'plots/final_training_curves.png',
            'plots/optimized_parity_plot.png',
            'model/optimized_final.jpt'
        ]
    }
    
    with open(f'{run_dir}/run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model, final_metrics

if __name__ == '__main__':
    # Run hyperparameter optimization
    study, best_trial, run_dir = run_optimization(n_trials=30)
    
    # Train final model with best parameters
    final_model, final_metrics = train_best_model(best_trial.params, run_dir, n_epochs=100)
    
    print(f"\nOptimization complete! Check '{run_dir}' directory for plots and results.")
