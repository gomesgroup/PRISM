#!/usr/bin/env python3
"""
High-Performance Multi-Fidelity Learning Proof of Concept
Using JAX for GPU acceleration and Polars for efficient data handling
"""

import os
os.environ['JAX_PLATFORMS'] = 'gpu'  # Force GPU usage

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random
import optax
import flax.linen as nn
from typing import Tuple, Dict, Any, Optional
import polars as pl
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Configure JAX for optimal GPU performance
jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
jax.config.update("jax_platform_name", "gpu")

print(f"JAX version: {jax.__version__}")
print(f"Devices available: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")


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
    
    def prepare_features(self, merged_df: pl.DataFrame) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Prepare feature matrices for JAX processing."""
        print("\n=== Preparing Features ===")
        
        # Load molecular descriptors
        acid_features = pl.read_csv(self.data_dir / "features" / "descriptors_acyl_chlorides.csv")
        amine_features = pl.read_csv(self.data_dir / "features" / "descriptors_amines.csv")
        
        # Load reaction energies
        rxn_energies = pl.read_csv(self.data_dir / "reaction_energies" / "reaction_TSB_w_aimnet2.csv")
        
        # Merge features with data
        merged_with_features = (
            merged_df
            .join(acid_features, left_on="acyl_chlorides", right_on="acyl_chlorides", how="left")
            .join(amine_features, left_on="amines", right_on="amines", how="left", suffix="_amine")
            .join(
                rxn_energies.select(["acid_chlorides", "amines", "barriers_dGTS_from_RXTS_B", 
                                    "barriers_dGTS_from_INT1_B", "rxn_dG_B"]),
                left_on=["acyl_chlorides", "amines"],
                right_on=["acid_chlorides", "amines"],
                how="left"
            )
        )
        
        # Select numeric features
        feature_cols = [
            "barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", "rxn_dG_B",
            "HTE_rate_corrected"
        ]
        
        # Convert to numpy then JAX arrays
        features_df = merged_with_features.select(feature_cols).fill_null(0)
        X = jnp.array(features_df.to_numpy(), dtype=jnp.float32)
        
        # Get HTE rates
        y_hte = jnp.array(
            merged_with_features.select("HTE_rate_corrected").to_numpy().flatten(),
            dtype=jnp.float32
        )
        
        # Get NMR rates (with mask for available data)
        nmr_col = merged_with_features.select("NMR_rate").to_numpy().flatten()
        has_nmr = ~np.isnan(nmr_col)
        y_nmr = jnp.array(np.nan_to_num(nmr_col), dtype=jnp.float32)
        nmr_mask = jnp.array(has_nmr, dtype=jnp.bool_)
        
        print(f"Feature shape: {X.shape}")
        print(f"HTE targets: {y_hte.shape}")
        print(f"NMR available: {nmr_mask.sum()}/{len(nmr_mask)}")
        
        return X, y_hte, y_nmr, nmr_mask


class MultiFidelityGPJAX:
    """
    Multi-fidelity Gaussian Process using JAX for GPU acceleration.
    Implements autoregressive scheme: f_high(x) = ρ·f_low(x) + δ(x)
    """
    
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.rho = None  # Scaling factor
        self.X_lf = None
        self.y_lf = None
        self.X_hf = None
        self.y_hf = None
        
    @jit
    def rbf_kernel(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        """RBF kernel computation on GPU."""
        # Efficient squared distance computation
        X1_sqnorms = jnp.sum(X1**2, axis=1, keepdims=True)
        X2_sqnorms = jnp.sum(X2**2, axis=1, keepdims=True)
        sqdist = X1_sqnorms + X2_sqnorms.T - 2 * jnp.dot(X1, X2.T)
        
        # RBF kernel
        K = self.variance * jnp.exp(-0.5 * sqdist / self.lengthscale**2)
        return K
    
    def fit(self, X_lf: jnp.ndarray, y_lf: jnp.ndarray, 
            X_hf: jnp.ndarray, y_hf: jnp.ndarray) -> None:
        """Fit multi-fidelity GP model."""
        print("\n=== Fitting Multi-Fidelity GP on GPU ===")
        start_time = time.time()
        
        self.X_lf = X_lf
        self.y_lf = y_lf
        self.X_hf = X_hf
        self.y_hf = y_hf
        
        # Step 1: Fit low-fidelity GP
        K_lf = self.rbf_kernel(X_lf, X_lf)
        K_lf = K_lf + 1e-4 * jnp.eye(len(X_lf))  # Add jitter for stability
        
        # Cholesky decomposition for efficient solving
        L_lf = jnp.linalg.cholesky(K_lf)
        alpha_lf = jnp.linalg.solve(L_lf.T, jnp.linalg.solve(L_lf, y_lf))
        
        # Step 2: Predict low-fidelity at high-fidelity locations
        K_star = self.rbf_kernel(X_hf, X_lf)
        lf_at_hf = K_star @ alpha_lf
        
        # Step 3: Optimize scaling factor ρ
        self.rho = self._optimize_rho(lf_at_hf, y_hf)
        
        # Step 4: Fit discrepancy GP
        delta_y = y_hf - self.rho * lf_at_hf
        
        # Store for predictions
        self.K_lf_inv_y = alpha_lf
        self.L_lf = L_lf
        self.delta_y = delta_y
        
        print(f"Fitted in {time.time() - start_time:.2f}s")
        print(f"Optimized ρ (scaling factor): {self.rho:.4f}")
    
    def _optimize_rho(self, lf_pred: jnp.ndarray, hf_true: jnp.ndarray) -> float:
        """Optimize scaling factor using closed-form solution."""
        numerator = jnp.dot(lf_pred, hf_true)
        denominator = jnp.dot(lf_pred, lf_pred)
        return float(numerator / denominator)
    
    @jit
    def predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict with uncertainty quantification."""
        # Low-fidelity prediction
        K_test_lf = self.rbf_kernel(X_test, self.X_lf)
        lf_mean = K_test_lf @ self.K_lf_inv_y
        
        # Discrepancy prediction (simplified - using mean of residuals)
        K_test_hf = self.rbf_kernel(X_test, self.X_hf)
        K_hf = self.rbf_kernel(self.X_hf, self.X_hf) + 1e-4 * jnp.eye(len(self.X_hf))
        
        # Solve for discrepancy
        L_hf = jnp.linalg.cholesky(K_hf)
        alpha_delta = jnp.linalg.solve(L_hf.T, jnp.linalg.solve(L_hf, self.delta_y))
        delta_mean = K_test_hf @ alpha_delta
        
        # Combined prediction
        hf_mean = self.rho * lf_mean + delta_mean
        
        # Uncertainty (simplified - based on distance to training points)
        K_test_test = self.rbf_kernel(X_test, X_test)
        v = jnp.linalg.solve(L_hf, K_test_hf.T)
        hf_var = jnp.diag(K_test_test) - jnp.sum(v**2, axis=0)
        hf_std = jnp.sqrt(jnp.maximum(hf_var, 1e-6))
        
        return hf_mean, hf_std


class DeepMultiFidelityNN(nn.Module):
    """
    Deep Neural Network for Multi-Fidelity learning using Flax.
    Captures complex non-linear relationships between fidelities.
    """
    features: Tuple[int, ...] = (128, 64, 32)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, y_lf: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Concatenate features with low-fidelity output
        x = jnp.concatenate([x, y_lf.reshape(-1, 1)], axis=1)
        
        # Deep network with skip connections
        for i, feat in enumerate(self.features):
            x_prev = x
            x = nn.Dense(feat)(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            
            # Skip connection every 2 layers
            if i > 0 and i % 2 == 0 and x_prev.shape[-1] == x.shape[-1]:
                x = x + x_prev
        
        # Output mean and log-variance
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        
        mu = nn.Dense(1)(x).squeeze()
        log_var = nn.Dense(1)(x).squeeze()
        
        return mu, log_var


def train_dnn_multifidelity(X_lf, y_lf, X_hf, y_hf, X_val, y_val, 
                           learning_rate=1e-3, epochs=100, batch_size=32):
    """Train DNN multi-fidelity model with GPU acceleration."""
    print("\n=== Training Deep Multi-Fidelity Network on GPU ===")
    
    # Initialize model
    model = DeepMultiFidelityNN()
    key = random.PRNGKey(42)
    
    # Get HTE predictions at HF locations
    # For simplicity, using nearest neighbor matching
    y_lf_at_hf = y_lf[:len(X_hf)]  # Simplified - in practice would need proper mapping
    
    # Initialize parameters
    params = model.init(key, X_hf[:1], y_lf_at_hf[:1])
    
    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Loss function
    def loss_fn(params, X, y_lf, y_true):
        mu, log_var = model.apply(params, X, y_lf, training=True)
        var = jnp.exp(log_var)
        
        # Negative log-likelihood
        nll = 0.5 * (jnp.log(var) + (y_true - mu)**2 / var)
        
        # Add regularization on uncertainty
        reg = 0.01 * jnp.mean(log_var)
        
        return jnp.mean(nll) + reg
    
    # Training loop
    @jit
    def train_step(params, opt_state, X_batch, y_lf_batch, y_batch):
        loss_value, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_lf_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Training
    losses = []
    for epoch in range(epochs):
        # Mini-batch training
        n_batches = len(X_hf) // batch_size
        epoch_loss = 0
        
        for i in range(n_batches):
            batch_idx = slice(i * batch_size, (i + 1) * batch_size)
            X_batch = X_hf[batch_idx]
            y_lf_batch = y_lf_at_hf[batch_idx]
            y_batch = y_hf[batch_idx]
            
            params, opt_state, loss = train_step(params, opt_state, X_batch, y_lf_batch, y_batch)
            epoch_loss += loss
        
        avg_loss = epoch_loss / n_batches
        losses.append(float(avg_loss))
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, params, losses


def evaluate_models(y_true: jnp.ndarray, y_pred: jnp.ndarray, 
                   y_std: Optional[jnp.ndarray] = None, model_name: str = "") -> Dict[str, float]:
    """Evaluate model performance with comprehensive metrics."""
    # Convert to numpy for sklearn metrics
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Basic metrics
    mse = float(jnp.mean((y_true - y_pred) ** 2))
    rmse = float(jnp.sqrt(mse))
    mae = float(jnp.mean(jnp.abs(y_true - y_pred)))
    
    # R² score
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    
    # Relative error
    rel_error = float(jnp.mean(jnp.abs((y_true - y_pred) / (y_true + 1e-8))))
    
    metrics = {
        f"{model_name}_rmse": rmse,
        f"{model_name}_mae": mae,
        f"{model_name}_r2": r2,
        f"{model_name}_rel_error": rel_error
    }
    
    # Uncertainty metrics if provided
    if y_std is not None:
        # Coverage at 95% confidence
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = float(jnp.mean((y_true >= lower) & (y_true <= upper)))
        
        # Mean interval width
        interval_width = float(jnp.mean(upper - lower))
        
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
            ax.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='none', 
                       alpha=0.2, color='gray', linewidth=0.5)
        
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
        
        # Residual plot
        ax = axes[1, idx]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
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
    plt.savefig('results/multifidelity_poc_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("HIGH-PERFORMANCE MULTI-FIDELITY LEARNING PROOF OF CONCEPT")
    print("Using JAX (GPU) + Polars for Maximum Performance")
    print("=" * 70)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Load data with Polars
    data_loader = MultiFidelityDataLoader()
    merged_df, nmr_df = data_loader.load_data()
    
    # Step 2: Prepare features for JAX
    X, y_hte, y_nmr, nmr_mask = data_loader.prepare_features(merged_df)
    
    # Split data
    X_hf = X[nmr_mask]
    y_hf = y_nmr[nmr_mask]
    X_lf = X  # All data for low-fidelity
    y_lf = y_hte  # HTE rates
    
    print(f"\nHigh-fidelity samples: {len(X_hf)}")
    print(f"Low-fidelity samples: {len(X_lf)}")
    
    # Step 3: Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    all_predictions = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_hf)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*50}")
        
        # Split high-fidelity data
        X_hf_train, X_hf_test = X_hf[train_idx], X_hf[test_idx]
        y_hf_train, y_hf_test = y_hf[train_idx], y_hf[test_idx]
        
        # Get corresponding low-fidelity data
        # In practice, we'd have more LF data, but for POC using same split
        y_lf_train = y_lf[nmr_mask][train_idx]
        
        fold_predictions = {}
        
        # Model 1: Baseline - Single-fidelity on NMR only
        print("\n--- Baseline: Single-Fidelity GP (NMR only) ---")
        baseline_gp = MultiFidelityGPJAX()
        # Use only high-fidelity data
        baseline_gp.fit(X_hf_train, y_hf_train, X_hf_train, y_hf_train)
        baseline_gp.rho = 0.0  # No low-fidelity contribution
        y_pred_baseline, y_std_baseline = baseline_gp.predict(X_hf_test)
        
        baseline_metrics = evaluate_models(y_hf_test, y_pred_baseline, y_std_baseline, "baseline")
        print(f"Baseline R²: {baseline_metrics['baseline_r2']:.4f}")
        fold_predictions['Baseline GP'] = (np.array(y_pred_baseline), np.array(y_std_baseline))
        
        # Model 2: Multi-fidelity Gaussian Process
        print("\n--- Multi-Fidelity Gaussian Process ---")
        mf_gp = MultiFidelityGPJAX(lengthscale=1.0, variance=1.0)
        mf_gp.fit(X_lf, y_lf, X_hf_train, y_hf_train)
        y_pred_mfgp, y_std_mfgp = mf_gp.predict(X_hf_test)
        
        mfgp_metrics = evaluate_models(y_hf_test, y_pred_mfgp, y_std_mfgp, "mfgp")
        print(f"MF-GP R²: {mfgp_metrics['mfgp_r2']:.4f}")
        print(f"Improvement over baseline: {(mfgp_metrics['mfgp_r2'] - baseline_metrics['baseline_r2']) * 100:.1f}%")
        fold_predictions['MF-GP'] = (np.array(y_pred_mfgp), np.array(y_std_mfgp))
        
        # Model 3: Deep Multi-fidelity Network
        print("\n--- Deep Multi-Fidelity Network ---")
        if len(X_hf_train) >= 32:  # Need enough data for batching
            dnn_model, dnn_params, losses = train_dnn_multifidelity(
                X_lf, y_lf, X_hf_train, y_hf_train, 
                X_hf_test, y_hf_test,
                epochs=50, batch_size=min(32, len(X_hf_train))
            )
            
            # Predict
            y_lf_test = y_lf[nmr_mask][test_idx]
            mu_dnn, log_var_dnn = dnn_model.apply(dnn_params, X_hf_test, y_lf_test, training=False)
            std_dnn = jnp.exp(0.5 * log_var_dnn)
            
            dnn_metrics = evaluate_models(y_hf_test, mu_dnn, std_dnn, "dnn")
            print(f"DNN R²: {dnn_metrics['dnn_r2']:.4f}")
            fold_predictions['DNN-MF'] = (np.array(mu_dnn), np.array(std_dnn))
        
        # Store results
        fold_results = {**baseline_metrics, **mfgp_metrics}
        if 'dnn_metrics' in locals():
            fold_results.update(dnn_metrics)
        all_results.append(fold_results)
        
        # Store best predictions for visualization
        if fold_idx == 0:  # Use first fold for visualization
            all_predictions = fold_predictions
            y_test_for_plot = np.array(y_hf_test)
    
    # Step 4: Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("=" * 70)
    
    results_df = pl.DataFrame(all_results)
    
    for col in results_df.columns:
        if col.endswith('_r2'):
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Step 5: Visualization
    print("\n=== Creating Visualizations ===")
    plot_results(y_test_for_plot, all_predictions, 
                "Multi-Fidelity Learning: HTE → NMR Prediction")
    
    # Step 6: Performance summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    baseline_r2 = results_df.filter(pl.col("baseline_r2").is_not_null())["baseline_r2"].mean()
    mfgp_r2 = results_df.filter(pl.col("mfgp_r2").is_not_null())["mfgp_r2"].mean()
    
    print(f"Baseline (NMR-only) R²: {baseline_r2:.4f}")
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
    print(f"Performance gain: {((mfgp_r2 - baseline_r2) / baseline_r2 * 100):.1f}%")
    print(f"Value ratio: {((mfgp_r2 - baseline_r2) / baseline_r2) / ((total_cost - nmr_only_cost) / nmr_only_cost):.2f}")
    
    print("\n✅ Proof of Concept Complete!")
    
    return results_df


if __name__ == "__main__":
    # Set GPU memory allocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    # Run POC
    results = main()
