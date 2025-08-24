#!/usr/bin/env python3
"""
High-Performance Multi-Fidelity Learning Proof of Concept
Using PyTorch for GPU acceleration and Polars for efficient data handling
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Dict, Any, Optional
import polars as pl
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import gpytorch

# Configure PyTorch for optimal GPU performance
torch.set_float32_matmul_precision('high')  # Use TensorFloat-32 for A100s
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class MultiFidelityDataLoader:
    """Efficient data loader using Polars."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.hte_data = None
        self.nmr_data = None
        self.features = None
        
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load HTE and NMR data using Polars for efficiency."""
        print("\n=== Loading Multi-Fidelity Data with Polars ===")
        
        # Load HTE rates (low-fidelity)
        hte_df = pl.read_csv(self.data_dir / "rates" / "corrected_hte_rates.csv")
        
        # Filter for measurable rates only
        hte_df = hte_df.filter(
            (pl.col("Fast_unmeasurable") == False) & 
            (pl.col("HTE_rate_corrected") > 0)
        )
        
        # Load NMR rates (high-fidelity)
        nmr_df = pl.read_csv(self.data_dir / "rates" / "nmr_rates_only.csv")
        
        # Merge datasets
        merged_df = hte_df.join(
            nmr_df.select(["acyl_chlorides", "amines", "NMR_rate"]),
            on=["acyl_chlorides", "amines"],
            how="left"
        )
        
        print(f"Total HTE measurements: {len(hte_df)}")
        print(f"Total NMR measurements: {len(nmr_df)}")
        print(f"Overlap (HTE with NMR): {merged_df.filter(pl.col('NMR_rate').is_not_null()).height}")
        
        self.hte_data = hte_df
        self.nmr_data = nmr_df
        
        return merged_df, nmr_df
    
    def prepare_features(self, merged_df: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare feature matrices for PyTorch processing."""
        print("\n=== Preparing Features ===")
        
        # Load molecular descriptors
        acid_features = pl.read_csv(self.data_dir / "features" / "descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv")
        amine_features = pl.read_csv(self.data_dir / "features" / "descriptors_amines_morfeus_addn_w_xtb.csv")
        
        # Load reaction energies
        rxn_energies = pl.read_csv(self.data_dir / "reaction_energies" / "reaction_TSB_w_aimnet2.csv")
        
        # Process acid features - select numeric columns
        acid_numeric_cols = [col for col in acid_features.columns 
                           if col not in ['acyl_chlorides', 'class', 'smiles', 'has_acidic_H_for_pka_aHs', 
                                        'has_acidic_H_for_pka_lowest', 'acyl_chlorides_index']]
        
        # Process amine features - select numeric columns  
        amine_numeric_cols = [col for col in amine_features.columns
                            if col not in ['amines', 'class', 'smiles', 'has_acidic_H_for_pka_basic', 
                                         'has_acidic_H_for_pka_lowest', 'amines_index']]
        
        # Merge features with data
        merged_with_features = (
            merged_df
            .join(acid_features.select(['acyl_chlorides'] + acid_numeric_cols[:10]), 
                  left_on="acyl_chlorides", right_on="acyl_chlorides", how="left")
            .join(amine_features.select(['amines'] + amine_numeric_cols[:10]), 
                  left_on="amines", right_on="amines", how="left", suffix="_amine")
            .join(
                rxn_energies.select(["acid_chlorides", "amines", "barriers_dGTS_from_RXTS_B", 
                                    "barriers_dGTS_from_INT1_B", "rxn_dG_B"]),
                left_on=["acyl_chlorides", "amines"],
                right_on=["acid_chlorides", "amines"],
                how="left"
            )
        )
        
        # Select features for model
        feature_cols = ["barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B", 
                       "HTE_rate_corrected"] + acid_numeric_cols[:5] + amine_numeric_cols[:5]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in merged_with_features.columns]
        
        # Convert to numpy then PyTorch tensors
        features_df = merged_with_features.select(available_cols).fill_null(0)
        X = torch.tensor(features_df.to_numpy(), dtype=torch.float32).to(device)
        
        # Get HTE rates
        y_hte = torch.tensor(
            merged_with_features.select("HTE_rate_corrected").to_numpy().flatten(),
            dtype=torch.float32
        ).to(device)
        
        # Get NMR rates (with mask for available data)
        nmr_col = merged_with_features.select("NMR_rate").to_numpy().flatten()
        has_nmr = ~np.isnan(nmr_col)
        y_nmr = torch.tensor(np.nan_to_num(nmr_col), dtype=torch.float32).to(device)
        nmr_mask = torch.tensor(has_nmr, dtype=torch.bool).to(device)
        
        print(f"Feature shape: {X.shape}")
        print(f"HTE targets: {y_hte.shape}")
        print(f"NMR available: {nmr_mask.sum().item()}/{len(nmr_mask)}")
        
        return X, y_hte, y_nmr, nmr_mask


class MultiFidelityGP(gpytorch.models.ExactGP):
    """Multi-fidelity Gaussian Process using GPyTorch."""
    
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        
        # Multi-task kernel for multi-fidelity
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=num_tasks,
            rank=1
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class DeepMultiFidelityNetwork(nn.Module):
    """
    Deep Neural Network for Multi-Fidelity learning.
    Captures complex non-linear relationships between fidelities.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.1):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim + 1  # +1 for HTE rate
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for mean and uncertainty
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.fc_log_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor, y_lf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate features with low-fidelity output
        x = torch.cat([x, y_lf.unsqueeze(1)], dim=1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Predict mean and log-variance
        mu = self.fc_mu(encoded).squeeze()
        log_var = self.fc_log_var(encoded).squeeze()
        
        return mu, log_var
    
    def loss_function(self, mu: torch.Tensor, log_var: torch.Tensor, 
                     y_true: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
        """Negative log-likelihood with uncertainty regularization."""
        var = torch.exp(log_var)
        
        # NLL loss
        nll = 0.5 * (torch.log(var + 1e-8) + (y_true - mu)**2 / (var + 1e-8))
        
        # Uncertainty regularization
        reg = beta * torch.mean(log_var)
        
        return torch.mean(nll) + reg


def train_multifidelity_gp(X_lf, y_lf, X_hf, y_hf):
    """Train multi-fidelity Gaussian Process with GPU acceleration."""
    print("\n=== Training Multi-Fidelity GP on GPU ===")
    start_time = time.time()
    
    # Prepare multi-task data (low and high fidelity)
    n_lf = X_lf.shape[0]
    n_hf = X_hf.shape[0]
    
    # Stack inputs (each point gets evaluated at both fidelities)
    X_train = torch.cat([X_hf, X_hf], dim=0)
    
    # Create multi-task targets [low_fidelity, high_fidelity]
    # For HF points, we have both LF (HTE) and HF (NMR) observations
    # Simplified: using nearest HTE values for HF points
    y_train = torch.stack([
        torch.cat([y_lf[:n_hf], y_lf[:n_hf]]),  # Low fidelity observations
        torch.cat([torch.zeros_like(y_hf), y_hf])  # High fidelity (0 for missing)
    ], dim=-1)
    
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device)
    model = MultiFidelityGP(X_train, y_train, likelihood).to(device)
    
    # Training mode
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.1)
    
    # Loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    for i in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1}/100, Loss: {loss.item():.4f}")
    
    print(f"Trained in {time.time() - start_time:.2f}s")
    
    return model, likelihood


def train_deep_multifidelity(X_lf, y_lf, X_hf, y_hf, X_val, y_val, 
                            input_dim, epochs=100, batch_size=32):
    """Train deep multi-fidelity network with GPU acceleration."""
    print("\n=== Training Deep Multi-Fidelity Network on GPU ===")
    start_time = time.time()
    
    # Initialize model
    model = DeepMultiFidelityNetwork(input_dim).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    # Get HTE predictions at HF locations (simplified matching)
    y_lf_at_hf = y_lf[:len(X_hf)]
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_hf, y_lf_at_hf, y_hf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    losses = []
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y_lf, batch_y_hf in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            mu, log_var = model(batch_x, batch_y_lf)
            
            # Compute loss
            loss = model.loss_function(mu, log_var, batch_y_hf)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print(f"Trained in {time.time() - start_time:.2f}s")
    
    return model, losses


def evaluate_models(y_true: torch.Tensor, y_pred: torch.Tensor, 
                   y_std: Optional[torch.Tensor] = None, model_name: str = "") -> Dict[str, float]:
    """Evaluate model performance with comprehensive metrics."""
    # Move to CPU for metrics calculation
    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    
    # Basic metrics
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    # Relative error
    rel_error = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
    
    metrics = {
        f"{model_name}_rmse": rmse,
        f"{model_name}_mae": mae,
        f"{model_name}_r2": r2,
        f"{model_name}_rel_error": rel_error
    }
    
    # Uncertainty metrics if provided
    if y_std is not None:
        y_std = y_std.cpu().numpy() if torch.is_tensor(y_std) else y_std
        
        # Coverage at 95% confidence
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        
        # Mean interval width
        interval_width = float(np.mean(upper - lower))
        
        metrics[f"{model_name}_coverage_95"] = coverage
        metrics[f"{model_name}_interval_width"] = interval_width
    
    return metrics


def plot_results(y_true: np.ndarray, predictions: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
                title: str = "Multi-Fidelity Predictions") -> None:
    """Create comprehensive visualization of results."""
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (model_name, (y_pred, y_std)) in enumerate(predictions.items()):
        # Parity plot
        ax = axes[0, idx]
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Add uncertainty bands if available
        if y_std is not None:
            # Sort for cleaner error bars
            sort_idx = np.argsort(y_true)
            ax.fill_between(y_true[sort_idx], 
                          (y_pred - 1.96*y_std)[sort_idx], 
                          (y_pred + 1.96*y_std)[sort_idx],
                          alpha=0.2, color='gray', label='95% CI')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Calculate R²
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        ax.set_title(f'{model_name}\nR² = {r2:.3f}')
        ax.set_xlabel('True NMR Rate')
        ax.set_ylabel('Predicted NMR Rate')
        ax.grid(True, alpha=0.3)
        if y_std is not None:
            ax.legend()
        
        # Residual plot
        ax = axes[1, idx]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Add confidence bands if available
        if y_std is not None:
            ax.fill_between(np.sort(y_pred), 
                          -1.96*y_std[np.argsort(y_pred)], 
                          1.96*y_std[np.argsort(y_pred)],
                          alpha=0.2, color='gray')
        
        ax.set_xlabel('Predicted NMR Rate')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # Add residual statistics
        ax.text(0.05, 0.95, f'Mean: {residuals.mean():.3f}\nStd: {residuals.std():.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/multifidelity_poc_torch_results.png', dpi=150, bbox_inches='tight')
    plt.show()


class SimpleGPModel(gpytorch.models.ExactGP):
    """Simple GP for baseline comparison."""
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("HIGH-PERFORMANCE MULTI-FIDELITY LEARNING PROOF OF CONCEPT")
    print("Using PyTorch (GPU) + Polars for Maximum Performance")
    print("=" * 70)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Load data with Polars
    data_loader = MultiFidelityDataLoader()
    merged_df, nmr_df = data_loader.load_data()
    
    # Step 2: Prepare features for PyTorch
    X, y_hte, y_nmr, nmr_mask = data_loader.prepare_features(merged_df)
    
    # Split data
    X_hf = X[nmr_mask]
    y_hf = y_nmr[nmr_mask]
    X_lf = X  # All data for low-fidelity
    y_lf = y_hte  # HTE rates
    
    print(f"\nHigh-fidelity samples: {len(X_hf)}")
    print(f"Low-fidelity samples: {len(X_lf)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Step 3: Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    all_predictions = {}
    
    # Use first fold for visualization
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_hf.cpu())):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*50}")
        
        # Convert indices to tensors
        train_idx = torch.tensor(train_idx).to(device)
        test_idx = torch.tensor(test_idx).to(device)
        
        # Split high-fidelity data
        X_hf_train = X_hf[train_idx]
        X_hf_test = X_hf[test_idx]
        y_hf_train = y_hf[train_idx]
        y_hf_test = y_hf[test_idx]
        
        fold_predictions = {}
        
        # Model 1: Baseline - Single-fidelity GP on NMR only
        print("\n--- Baseline: Single-Fidelity GP (NMR only) ---")
        
        # Train simple GP
        likelihood_baseline = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model_baseline = SimpleGPModel(X_hf_train, y_hf_train, likelihood_baseline).to(device)
        
        # Training
        model_baseline.train()
        likelihood_baseline.train()
        optimizer = torch.optim.Adam([
            {'params': model_baseline.parameters()},
            {'params': likelihood_baseline.parameters()},
        ], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_baseline, model_baseline)
        
        for i in range(50):
            optimizer.zero_grad()
            output = model_baseline(X_hf_train)
            loss = -mll(output, y_hf_train)
            loss.backward()
            optimizer.step()
        
        # Prediction
        model_baseline.eval()
        likelihood_baseline.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_baseline = likelihood_baseline(model_baseline(X_hf_test))
            y_pred_baseline = pred_baseline.mean
            y_std_baseline = pred_baseline.stddev
        
        baseline_metrics = evaluate_models(y_hf_test, y_pred_baseline, y_std_baseline, "baseline")
        print(f"Baseline R²: {baseline_metrics['baseline_r2']:.4f}")
        fold_predictions['Baseline GP'] = (y_pred_baseline.cpu().numpy(), y_std_baseline.cpu().numpy())
        
        # Model 2: Multi-fidelity Gaussian Process
        print("\n--- Multi-Fidelity Gaussian Process ---")
        try:
            mf_model, mf_likelihood = train_multifidelity_gp(X_lf, y_lf, X_hf_train, y_hf_train)
            
            # Prediction
            mf_model.eval()
            mf_likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # Predict high-fidelity task
                test_x_multi = X_hf_test
                pred_mf = mf_likelihood(mf_model(test_x_multi))
                # Extract high-fidelity predictions (task 2)
                y_pred_mfgp = pred_mf.mean[:, 1]
                y_std_mfgp = pred_mf.stddev[:, 1]
            
            mfgp_metrics = evaluate_models(y_hf_test, y_pred_mfgp, y_std_mfgp, "mfgp")
            print(f"MF-GP R²: {mfgp_metrics['mfgp_r2']:.4f}")
            improvement = (mfgp_metrics['mfgp_r2'] - baseline_metrics['baseline_r2']) / baseline_metrics['baseline_r2'] * 100
            print(f"Improvement over baseline: {improvement:.1f}%")
            fold_predictions['MF-GP'] = (y_pred_mfgp.cpu().numpy(), y_std_mfgp.cpu().numpy())
        except Exception as e:
            print(f"MF-GP failed: {e}")
            mfgp_metrics = {}
        
        # Model 3: Deep Multi-fidelity Network
        print("\n--- Deep Multi-Fidelity Network ---")
        if len(X_hf_train) >= 32:  # Need enough data for batching
            try:
                dnn_model, losses = train_deep_multifidelity(
                    X_lf, y_lf, X_hf_train, y_hf_train, 
                    X_hf_test, y_hf_test,
                    input_dim=X.shape[1],
                    epochs=50, 
                    batch_size=min(32, len(X_hf_train))
                )
                
                # Predict
                dnn_model.eval()
                with torch.no_grad():
                    y_lf_test = y_lf[nmr_mask][test_idx]
                    mu_dnn, log_var_dnn = dnn_model(X_hf_test, y_lf_test)
                    std_dnn = torch.exp(0.5 * log_var_dnn)
                
                dnn_metrics = evaluate_models(y_hf_test, mu_dnn, std_dnn, "dnn")
                print(f"DNN R²: {dnn_metrics['dnn_r2']:.4f}")
                fold_predictions['DNN-MF'] = (mu_dnn.cpu().numpy(), std_dnn.cpu().numpy())
            except Exception as e:
                print(f"DNN failed: {e}")
                dnn_metrics = {}
        
        # Store results
        fold_results = {**baseline_metrics}
        if mfgp_metrics:
            fold_results.update(mfgp_metrics)
        if 'dnn_metrics' in locals() and dnn_metrics:
            fold_results.update(dnn_metrics)
        all_results.append(fold_results)
        
        # Store best predictions for visualization (first fold)
        if fold_idx == 0:
            all_predictions = fold_predictions
            y_test_for_plot = y_hf_test.cpu().numpy()
        
        # Only run one fold for quick POC
        break
    
    # Step 4: Aggregate results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    results_df = pl.DataFrame(all_results)
    
    for col in results_df.columns:
        if col.endswith('_r2'):
            values = results_df[col].drop_nulls()
            if len(values) > 0:
                mean_val = values.mean()
                print(f"{col}: {mean_val:.4f}")
    
    # Step 5: Visualization
    print("\n=== Creating Visualizations ===")
    if all_predictions:
        plot_results(y_test_for_plot, all_predictions, 
                    "Multi-Fidelity Learning: HTE → NMR Prediction (PyTorch GPU)")
    
    # Step 6: Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    if 'baseline_r2' in all_results[0]:
        baseline_r2 = all_results[0]['baseline_r2']
        print(f"Baseline (NMR-only) R²: {baseline_r2:.4f}")
        
        if 'mfgp_r2' in all_results[0]:
            mfgp_r2 = all_results[0]['mfgp_r2']
            print(f"Multi-Fidelity GP R²: {mfgp_r2:.4f}")
            print(f"Relative Improvement: {((mfgp_r2 - baseline_r2) / baseline_r2 * 100):.1f}%")
    
    # Cost-benefit analysis
    print("\n=== Cost-Benefit Analysis ===")
    cost_hte = 1.0  # Relative cost
    cost_nmr = 100.0  # 100x more expensive
    
    n_hte = len(X_lf)
    n_nmr = len(X_hf)
    
    total_cost = n_hte * cost_hte + n_nmr * cost_nmr
    nmr_only_cost = n_nmr * cost_nmr
    
    print(f"Total measurements used: {n_hte} HTE + {n_nmr} NMR")
    print(f"Total cost (relative): {total_cost:.0f}")
    print(f"NMR-only cost: {nmr_only_cost:.0f}")
    print(f"Cost increase: {(total_cost - nmr_only_cost) / nmr_only_cost * 100:.1f}%")
    
    if 'mfgp_r2' in all_results[0] and 'baseline_r2' in all_results[0]:
        perf_gain = ((mfgp_r2 - baseline_r2) / baseline_r2 * 100)
        print(f"Performance gain: {perf_gain:.1f}%")
        value_ratio = perf_gain / ((total_cost - nmr_only_cost) / nmr_only_cost * 100)
        print(f"Value ratio (performance gain / cost increase): {value_ratio:.2f}")
    
    # GPU utilization summary
    if torch.cuda.is_available():
        print(f"\n=== GPU Utilization ===")
        print(f"Peak memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print("\n✅ Proof of Concept Complete!")
    
    return results_df


if __name__ == "__main__":
    # Run POC
    results = main()
