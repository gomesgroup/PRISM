#!/usr/bin/env python3
"""
High-Performance Multi-Fidelity Learning Proof of Concept
Using the ACTUAL libraries from the analysis document:
- mfpml: Multi-fidelity Probabilistic Machine Learning
- TLlib: Transfer Learning Library
- Polars: Efficient data handling
- PyTorch: GPU acceleration
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional

# Import mfpml components
from mfpml import MultiEIDNN, SingleFidelityBO, MultiFidelityBO
from mfpml import MultiKriging, SingleKriging, CoKriging
from mfpml import VariableKriging, ConstrainedBO

# Import TLlib components for transfer learning
import tllib.alignment.dann as dann
import tllib.alignment.cdan as cdan
import tllib.alignment.jan as jan
from tllib.modules.grl import GradientReverseLayer
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.utils.metric import accuracy
from tllib.utils.data import ForeverDataIterator

# Configure GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class MultiFidelityDataLoader:
    """Efficient data loader using Polars."""
    
    def __init__(self):
        self.data_dir = Path("data")
        
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare multi-fidelity data for mfpml."""
        print("\n=== Loading Multi-Fidelity Data with Polars ===")
        
        # Load HTE rates (low-fidelity)
        hte_df = pl.read_csv(self.data_dir / "rates" / "corrected_hte_rates.csv")
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
        
        # Load molecular descriptors
        acid_features = pl.read_csv(
            self.data_dir / "features" / "descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv"
        )
        amine_features = pl.read_csv(
            self.data_dir / "features" / "descriptors_amines_morfeus_addn_w_xtb.csv"
        )
        
        # Load reaction energies
        rxn_energies = pl.read_csv(
            self.data_dir / "reaction_energies" / "reaction_TSB_w_aimnet2.csv"
        )
        
        # Select numeric columns
        acid_numeric = [c for c in acid_features.columns 
                       if c not in ['acyl_chlorides', 'class', 'smiles'] 
                       and not c.startswith('has_')][:10]
        amine_numeric = [c for c in amine_features.columns 
                        if c not in ['amines', 'class', 'smiles'] 
                        and not c.startswith('has_')][:10]
        
        # Merge features
        merged_with_features = (
            merged_df
            .join(acid_features.select(['acyl_chlorides'] + acid_numeric), 
                  on="acyl_chlorides", how="left")
            .join(amine_features.select(['amines'] + amine_numeric), 
                  on="amines", how="left", suffix="_amine")
            .join(rxn_energies.select([
                "acid_chlorides", "amines", 
                "barriers_dGTS_from_RXTS_B", 
                "barriers_dGTS_from_INT1_B", 
                "rxn_dG_B"
            ]), left_on=["acyl_chlorides", "amines"],
                right_on=["acid_chlorides", "amines"], how="left")
        )
        
        # Extract features and targets
        feature_cols = ["barriers_dGTS_from_RXTS_B", "barriers_dGTS_from_INT1_B", 
                       "rxn_dG_B"] + acid_numeric[:3] + amine_numeric[:3]
        available_cols = [c for c in feature_cols if c in merged_with_features.columns]
        
        X = merged_with_features.select(available_cols).fill_null(0).to_numpy()
        y_hte = merged_with_features.select("HTE_rate_corrected").to_numpy().flatten()
        
        # Get NMR data
        nmr_mask = merged_with_features["NMR_rate"].is_not_null()
        X_hf = merged_with_features.filter(nmr_mask).select(available_cols).fill_null(0).to_numpy()
        y_nmr = merged_with_features.filter(nmr_mask).select("NMR_rate").to_numpy().flatten()
        
        print(f"Low-fidelity (HTE) samples: {len(X)}")
        print(f"High-fidelity (NMR) samples: {len(X_hf)}")
        print(f"Feature dimension: {X.shape[1]}")
        
        return X, y_hte, X_hf, y_nmr


class MFPMLMultiFidelityModel:
    """Multi-fidelity model using mfpml library."""
    
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def train_multi_kriging(self, X_lf, y_lf, X_hf, y_hf):
        """Train Multi-fidelity Kriging model from mfpml."""
        print("\n=== Training Multi-Fidelity Kriging (mfpml) ===")
        start_time = time.time()
        
        # Standardize features
        X_lf_scaled = self.scaler_X.fit_transform(X_lf)
        X_hf_scaled = self.scaler_X.transform(X_hf)
        
        # Standardize targets
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()
        
        # Initialize Multi-fidelity Kriging
        self.model = MultiKriging(
            X_lf=X_lf_scaled,
            y_lf=y_lf_scaled,
            X_hf=X_hf_scaled,
            y_hf=y_hf_scaled,
            regr_lf='linear',  # Linear regression for low-fidelity
            regr_hf='linear',  # Linear regression for high-fidelity
            corr='gauss',      # Gaussian correlation
            theta0=1e-2,       # Initial hyperparameter
            thetaL=1e-6,       # Lower bound
            thetaU=1e2,        # Upper bound
            nugget=1e-6        # Numerical stability
        )
        
        # Fit the model
        self.model.fit()
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        print(f"Optimized theta (LF): {self.model.model_lf.theta_}")
        print(f"Optimized theta (HF): {self.model.model_hf.theta_}")
        print(f"Scaling factor ρ: {self.model.rho:.4f}")
        
        return self
    
    def train_co_kriging(self, X_lf, y_lf, X_hf, y_hf):
        """Train Co-Kriging model from mfpml."""
        print("\n=== Training Co-Kriging (mfpml) ===")
        start_time = time.time()
        
        # Standardize
        X_lf_scaled = self.scaler_X.fit_transform(X_lf)
        X_hf_scaled = self.scaler_X.transform(X_hf)
        y_lf_scaled = self.scaler_y.fit_transform(y_lf.reshape(-1, 1)).flatten()
        y_hf_scaled = self.scaler_y.transform(y_hf.reshape(-1, 1)).flatten()
        
        # Initialize Co-Kriging
        self.model = CoKriging(
            X_lf=X_lf_scaled,
            y_lf=y_lf_scaled,
            X_hf=X_hf_scaled,
            y_hf=y_hf_scaled,
            regr='linear',
            corr='gauss',
            theta0=1e-2,
            thetaL=1e-6,
            thetaU=1e2
        )
        
        self.model.fit()
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        return self
    
    def predict(self, X_test):
        """Predict with uncertainty."""
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Get predictions
        y_pred_scaled, mse = self.model.predict(X_test_scaled, eval_MSE=True)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_std = np.sqrt(mse) * self.scaler_y.scale_[0]
        
        return y_pred, y_std


class TLlibTransferLearning:
    """Transfer learning using TLlib for domain adaptation from HTE to NMR."""
    
    def __init__(self, input_dim, hidden_dim=128):
        self.device = device
        
        # Feature extractor (shared between domains)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        ).to(device)
        
        # Rate predictor (regression head)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        ).to(device)
        
        # Domain discriminator for DANN
        self.domain_discriminator = DomainDiscriminator(
            in_feature=hidden_dim,
            hidden_size=hidden_dim // 2
        ).to(device)
        
        # Gradient reversal layer
        self.grl = GradientReverseLayer()
        
    def train_dann(self, X_source, y_source, X_target, y_target_val, 
                  epochs=100, batch_size=32, lr=1e-3):
        """Train using Domain Adversarial Neural Network (DANN) from TLlib."""
        print("\n=== Training with DANN (TLlib) ===")
        start_time = time.time()
        
        # Convert to tensors
        X_s = torch.FloatTensor(X_source).to(self.device)
        y_s = torch.FloatTensor(y_source).to(self.device)
        X_t = torch.FloatTensor(X_target).to(self.device)
        
        # Create datasets
        source_dataset = TensorDataset(X_s, y_s, torch.zeros(len(X_s)))  # Domain 0
        target_dataset = TensorDataset(X_t, torch.zeros(len(X_t)), torch.ones(len(X_t)))  # Domain 1
        
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': self.feature_extractor.parameters()},
            {'params': self.predictor.parameters()},
            {'params': self.domain_discriminator.parameters()}
        ], lr=lr)
        
        # Loss functions
        regression_loss = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.feature_extractor.train()
            self.predictor.train()
            self.domain_discriminator.train()
            
            total_loss = 0
            
            for (X_s_batch, y_s_batch, d_s), (X_t_batch, _, d_t) in zip(source_loader, target_loader):
                # Ensure same batch size
                min_batch = min(len(X_s_batch), len(X_t_batch))
                X_s_batch = X_s_batch[:min_batch]
                y_s_batch = y_s_batch[:min_batch]
                d_s = d_s[:min_batch]
                X_t_batch = X_t_batch[:min_batch]
                d_t = d_t[:min_batch]
                
                optimizer.zero_grad()
                
                # Extract features
                feat_s = self.feature_extractor(X_s_batch)
                feat_t = self.feature_extractor(X_t_batch)
                
                # Regression loss on source
                pred_s = self.predictor(feat_s).squeeze()
                loss_reg = regression_loss(pred_s, y_s_batch)
                
                # Domain adversarial loss (DANN)
                feat_s_rev = self.grl(feat_s)
                feat_t_rev = self.grl(feat_t)
                
                domain_loss_adv = dann.DomainAdversarialLoss(
                    self.domain_discriminator
                )(feat_s_rev, feat_t_rev)
                
                # Total loss
                loss = loss_reg + 0.1 * domain_loss_adv
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(source_loader):.4f}")
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        return self
    
    def predict(self, X_test):
        """Predict with the trained model."""
        self.feature_extractor.eval()
        self.predictor.eval()
        
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(X_test_t)
            predictions = self.predictor(features).squeeze()
        
        return predictions.cpu().numpy()


class MultiFidelityBayesianOptimization:
    """Multi-fidelity Bayesian Optimization using mfpml."""
    
    def __init__(self, bounds, n_init=10):
        self.bounds = bounds
        self.n_init = n_init
        self.optimizer = None
        
    def optimize_acquisition(self, X_lf, y_lf, X_hf, y_hf, n_iter=20):
        """Run multi-fidelity Bayesian optimization."""
        print("\n=== Multi-Fidelity Bayesian Optimization (mfpml) ===")
        
        # Initialize multi-fidelity BO
        self.optimizer = MultiFidelityBO(
            func_lf=None,  # We'll use existing data
            func_hf=None,
            bounds=self.bounds,
            n_init=self.n_init,
            kernel='rbf',
            acquisition='EI'  # Expected Improvement
        )
        
        # Set initial data
        self.optimizer.X_lf = X_lf
        self.optimizer.y_lf = y_lf
        self.optimizer.X_hf = X_hf
        self.optimizer.y_hf = y_hf
        
        # Find next points to evaluate
        best_points = []
        for i in range(n_iter):
            # Get next point via acquisition function
            x_next = self.optimizer.suggest_next_point()
            best_points.append(x_next)
            
            if i % 5 == 0:
                print(f"  Iteration {i+1}/{n_iter}: Suggested point shape: {x_next.shape}")
        
        return np.array(best_points)


def evaluate_model(y_true, y_pred, y_std=None, model_name=""):
    """Evaluate model performance."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    metrics = {
        f"{model_name}_rmse": rmse,
        f"{model_name}_mae": mae,
        f"{model_name}_r2": r2
    }
    
    if y_std is not None:
        # Coverage at 95% confidence
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        metrics[f"{model_name}_coverage_95"] = coverage
    
    return metrics


def plot_results(results_dict, save_path="results/mfpml_tllib_results.png"):
    """Visualize results from different models."""
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, data) in enumerate(results_dict.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_std = data.get('y_std', None)
        
        ax = axes[idx]
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        if y_std is not None:
            # Add error bars
            ax.errorbar(y_true, y_pred, yerr=1.96*y_std, fmt='none', 
                       alpha=0.2, color='gray', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Calculate R²
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        ax.set_title(f'{model_name}\nR² = {r2:.3f}')
        ax.set_xlabel('True NMR Rate')
        ax.set_ylabel('Predicted NMR Rate')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Fidelity Learning: mfpml + TLlib', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main execution using the actual libraries from the document."""
    print("=" * 70)
    print("MULTI-FIDELITY LEARNING WITH mfpml AND TLlib")
    print("Using the actual libraries from the analysis document")
    print("=" * 70)
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Load data
    data_loader = MultiFidelityDataLoader()
    X_lf, y_lf, X_hf, y_hf = data_loader.load_and_prepare_data()
    
    # Step 2: Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_hf)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*50}")
        
        # Split data
        X_hf_train, X_hf_test = X_hf[train_idx], X_hf[test_idx]
        y_hf_train, y_hf_test = y_hf[train_idx], y_hf[test_idx]
        
        # Model 1: Multi-fidelity Kriging (mfpml)
        print("\n--- Multi-Fidelity Kriging (mfpml) ---")
        try:
            mf_kriging = MFPMLMultiFidelityModel()
            mf_kriging.train_multi_kriging(X_lf, y_lf, X_hf_train, y_hf_train)
            
            y_pred_kriging, y_std_kriging = mf_kriging.predict(X_hf_test)
            
            kriging_metrics = evaluate_model(y_hf_test, y_pred_kriging, 
                                            y_std_kriging, "mf_kriging")
            print(f"MF-Kriging R²: {kriging_metrics['mf_kriging_r2']:.4f}")
            
            results['MF-Kriging (mfpml)'] = {
                'y_true': y_hf_test,
                'y_pred': y_pred_kriging,
                'y_std': y_std_kriging
            }
        except Exception as e:
            print(f"MF-Kriging failed: {e}")
        
        # Model 2: Co-Kriging (mfpml)
        print("\n--- Co-Kriging (mfpml) ---")
        try:
            co_kriging = MFPMLMultiFidelityModel()
            co_kriging.train_co_kriging(X_lf, y_lf, X_hf_train, y_hf_train)
            
            y_pred_cokriging, y_std_cokriging = co_kriging.predict(X_hf_test)
            
            cokriging_metrics = evaluate_model(y_hf_test, y_pred_cokriging,
                                              y_std_cokriging, "co_kriging")
            print(f"Co-Kriging R²: {cokriging_metrics['co_kriging_r2']:.4f}")
            
            results['Co-Kriging (mfpml)'] = {
                'y_true': y_hf_test,
                'y_pred': y_pred_cokriging,
                'y_std': y_std_cokriging
            }
        except Exception as e:
            print(f"Co-Kriging failed: {e}")
        
        # Model 3: Transfer Learning with DANN (TLlib)
        print("\n--- Transfer Learning DANN (TLlib) ---")
        try:
            transfer_model = TLlibTransferLearning(
                input_dim=X_lf.shape[1],
                hidden_dim=128
            )
            
            # Train with domain adaptation
            transfer_model.train_dann(
                X_lf, y_lf,  # Source domain (HTE)
                X_hf_train, y_hf_train,  # Target domain (NMR)
                epochs=50,
                batch_size=min(32, len(X_hf_train))
            )
            
            y_pred_dann = transfer_model.predict(X_hf_test)
            
            dann_metrics = evaluate_model(y_hf_test, y_pred_dann, None, "dann")
            print(f"DANN R²: {dann_metrics['dann_r2']:.4f}")
            
            results['DANN (TLlib)'] = {
                'y_true': y_hf_test,
                'y_pred': y_pred_dann,
                'y_std': None
            }
        except Exception as e:
            print(f"DANN failed: {e}")
        
        # Model 4: Multi-fidelity Bayesian Optimization (mfpml)
        print("\n--- Multi-Fidelity Bayesian Optimization (mfpml) ---")
        try:
            # Define bounds for optimization
            bounds = [(X_lf[:, i].min(), X_lf[:, i].max()) 
                     for i in range(X_lf.shape[1])]
            
            mf_bo = MultiFidelityBayesianOptimization(bounds, n_init=5)
            best_points = mf_bo.optimize_acquisition(
                X_lf[:100], y_lf[:100],  # Use subset for speed
                X_hf_train, y_hf_train,
                n_iter=10
            )
            print(f"Found {len(best_points)} optimal points for evaluation")
        except Exception as e:
            print(f"MF-BO failed: {e}")
        
        # Use first fold only for POC
        break
    
    # Step 3: Visualize results
    if results:
        print("\n=== Creating Visualizations ===")
        plot_results(results)
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    for model_name, data in results.items():
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        print(f"{model_name:25s} - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Cost-benefit analysis
    print("\n=== Cost-Benefit Analysis ===")
    print(f"Low-fidelity (HTE) samples: {len(X_lf)} @ $1 = ${len(X_lf)}")
    print(f"High-fidelity (NMR) samples: {len(X_hf)} @ $100 = ${len(X_hf) * 100}")
    print(f"Total cost: ${len(X_lf) + len(X_hf) * 100}")
    print(f"NMR-only cost: ${len(X_hf) * 100}")
    print(f"Additional cost for multi-fidelity: {(len(X_lf) / (len(X_hf) * 100)) * 100:.1f}%")
    
    print("\n✅ Proof of Concept Complete using mfpml and TLlib!")
    
    return results


if __name__ == "__main__":
    results = main()
