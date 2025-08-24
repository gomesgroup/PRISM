#!/usr/bin/env python
"""
Evaluate Dataset Metrics and Compare with Current Model Performance

This script analyzes:
1. The actual HTE dataset used for training 
2. The true performance metrics the model should achieve
3. Comparison with current model performance
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns

# Add path for our system
sys.path.append('/home/passos/ml_measurable_hte_rates/gt4sd_hte_integration')
from final_optimized_hte_system import FinalOptimizedHTESystem


def load_and_analyze_hte_dataset():
    """Load and analyze the actual HTE dataset."""
    
    print("📊 HTE DATASET ANALYSIS")
    print("=" * 60)
    
    # Load the corrected HTE rates dataset
    data_path = "/home/passos/ml_measurable_hte_rates/data/rates/corrected_hte_rates.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded dataset: {len(df)} reactions")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {data_path}")
        return None, None
    
    # Filter for measurable data only
    measurable_df = df[
        (df['Fast_unmeasurable'] == False) & 
        (df['HTE_rate_corrected'] > 0)
    ].copy()
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total reactions: {len(df)}")
    print(f"   Measurable reactions: {len(measurable_df)}")
    print(f"   Coverage: {len(measurable_df)/len(df)*100:.1f}%")
    
    # Analyze HTE rates
    hte_rates = measurable_df['HTE_rate_corrected']
    print(f"\n⚗️  HTE Rate Statistics:")
    print(f"   Mean: {hte_rates.mean():.3f}")
    print(f"   Median: {hte_rates.median():.3f}")
    print(f"   Std: {hte_rates.std():.3f}")
    print(f"   Range: {hte_rates.min():.3f} - {hte_rates.max():.3f}")
    print(f"   Log10 mean: {np.log10(hte_rates).mean():.3f}")
    print(f"   Log10 std: {np.log10(hte_rates).std():.3f}")
    
    # Check train/test split
    train_df = measurable_df[measurable_df['test splits'] == 'TRAIN']
    test_df = measurable_df[
        (measurable_df['test splits'] == 'TEST1') | 
        (measurable_df['test splits'] == 'TEST2')
    ]
    
    print(f"\n🔄 Train/Test Split:")
    print(f"   Train samples: {len(train_df)} ({len(train_df)/len(measurable_df)*100:.1f}%)")
    print(f"   Test samples: {len(test_df)} ({len(test_df)/len(measurable_df)*100:.1f}%)")
    
    return measurable_df, (train_df, test_df)


def baseline_model_performance(train_df, test_df):
    """Establish baseline performance using simple models."""
    
    print(f"\n🤖 BASELINE MODEL PERFORMANCE")
    print("=" * 60)
    
    # Simple feature engineering - use basic molecular properties
    # For this analysis, we'll use the numerical indices as proxy features
    # In reality, you'd use proper molecular descriptors
    
    feature_cols = ['acyl_chlorides', 'amines']  # Simple ID-based features
    target_col = 'HTE_rate_corrected'
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values  
    y_test = test_df[target_col].values
    
    # Log transform targets for better modeling
    y_train_log = np.log10(y_train)
    y_test_log = np.log10(y_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"🔧 Training baseline models...")
    
    # Model 1: Random Forest on original scale
    rf_orig = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_orig.fit(X_train_scaled, y_train)
    y_pred_rf_orig = rf_orig.predict(X_test_scaled)
    
    # Model 2: Random Forest on log scale
    rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_log.fit(X_train_scaled, y_train_log)
    y_pred_rf_log = 10 ** rf_log.predict(X_test_scaled)
    
    # Calculate metrics
    models = {
        'RF_Original': (y_test, y_pred_rf_orig),
        'RF_LogScale': (y_test, y_pred_rf_log),
    }
    
    baseline_results = {}
    
    for model_name, (y_true, y_pred) in models.items():
        # Ensure positive predictions
        y_pred = np.maximum(y_pred, 0.001)
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Log space metrics
        y_true_log = np.log10(y_true)
        y_pred_log = np.log10(y_pred)
        r2_log = r2_score(y_true_log, y_pred_log)
        mae_log = mean_absolute_error(y_true_log, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
        
        baseline_results[model_name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'r2_log': r2_log,
            'mae_log': mae_log,
            'rmse_log': rmse_log
        }
        
        print(f"\n📈 {model_name} Results:")
        print(f"   R²: {r2:.4f}")
        print(f"   MAE: {mae:.3f}")
        print(f"   RMSE: {rmse:.3f}")
        print(f"   R² (log): {r2_log:.4f}")
        print(f"   MAE (log): {mae_log:.3f}")
        print(f"   RMSE (log): {rmse_log:.3f}")
    
    return baseline_results


def evaluate_rt_model_performance(test_df):
    """Evaluate our current Regression Transformer model."""
    
    print(f"\n🤖 REGRESSION TRANSFORMER EVALUATION")
    print("=" * 60)
    
    try:
        # Initialize our optimized system
        print("🚀 Loading final optimized RT system...")
        rt_system = FinalOptimizedHTESystem()
        
        # Convert test data to RT format
        test_inputs = []
        true_values = []
        
        for _, row in test_df.iterrows():
            # Create input in RT format 
            # Note: This is a simplified format - the actual trained model 
            # might expect different descriptor tokens
            input_text = f"<d0>{row['acyl_chlorides']:.3f} <d1>{row['amines']:.3f} <hte> |"
            test_inputs.append(input_text)
            true_values.append(row['HTE_rate_corrected'])
        
        print(f"📊 Evaluating on {len(test_inputs)} test samples...")
        
        # Get predictions from RT model
        predictions = []
        generated_texts = []
        confidences = []
        
        for i, input_text in enumerate(test_inputs):
            if i % 20 == 0:
                print(f"   Processing {i}/{len(test_inputs)}...")
            
            try:
                result = rt_system.final_predict_hte_rate(input_text, max_new_tokens=8)
                
                # Extract prediction
                pred_value = result.get('hte_rate', -1.0)  # -1.0 is fallback
                confidence = result.get('confidence', 0.0)
                generated = result.get('generated_text', '')
                
                predictions.append(pred_value if pred_value != -1.0 else np.nan)
                confidences.append(confidence)
                generated_texts.append(generated)
                
            except Exception as e:
                print(f"   Error on sample {i}: {e}")
                predictions.append(np.nan)
                confidences.append(0.0)
                generated_texts.append('')
        
        # Convert to arrays and handle NaNs
        y_true = np.array(true_values)
        y_pred = np.array(predictions)
        valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true) & (y_pred > 0)
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        print(f"\n📊 RT Model Results:")
        print(f"   Total test samples: {len(test_inputs)}")
        print(f"   Valid predictions: {len(y_true_valid)} ({len(y_true_valid)/len(test_inputs)*100:.1f}%)")
        print(f"   Failed predictions: {len(test_inputs) - len(y_true_valid)}")
        
        if len(y_true_valid) > 5:  # Need minimum samples for meaningful metrics
            # Calculate metrics
            r2 = r2_score(y_true_valid, y_pred_valid)
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            
            # Log space metrics
            y_true_log = np.log10(y_true_valid)
            y_pred_log = np.log10(y_pred_valid)
            r2_log = r2_score(y_true_log, y_pred_log)
            mae_log = mean_absolute_error(y_true_log, y_pred_log)
            rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
            
            rt_results = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'r2_log': r2_log,
                'mae_log': mae_log,
                'rmse_log': rmse_log,
                'valid_predictions': len(y_true_valid),
                'success_rate': len(y_true_valid) / len(test_inputs) * 100
            }
            
            print(f"\n📈 Performance Metrics:")
            print(f"   R²: {r2:.4f}")
            print(f"   MAE: {mae:.3f}")
            print(f"   RMSE: {rmse:.3f}")
            print(f"   R² (log): {r2_log:.4f}")
            print(f"   MAE (log): {mae_log:.3f}")
            print(f"   RMSE (log): {rmse_log:.3f}")
            print(f"   Success rate: {rt_results['success_rate']:.1f}%")
            
            # Show some example predictions
            print(f"\n🔍 Example Predictions:")
            for i in range(min(5, len(y_true_valid))):
                idx = np.where(valid_mask)[0][i]
                print(f"   {i+1}. True: {y_true[idx]:.3f}, Pred: {y_pred[idx]:.3f}, "
                      f"Generated: {generated_texts[idx][:50]}...")
                
        else:
            print("❌ Insufficient valid predictions for meaningful evaluation")
            rt_results = {
                'r2': 0.0,
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2_log': 0.0,
                'mae_log': float('inf'),
                'rmse_log': float('inf'),
                'valid_predictions': len(y_true_valid),
                'success_rate': len(y_true_valid) / len(test_inputs) * 100
            }
        
        return rt_results
        
    except Exception as e:
        print(f"❌ RT model evaluation failed: {e}")
        return None


def comprehensive_evaluation():
    """Run comprehensive evaluation of dataset and models."""
    
    print("🎯 COMPREHENSIVE HTE MODEL EVALUATION")
    print("=" * 80)
    
    # Load dataset
    df, (train_df, test_df) = load_and_analyze_hte_dataset()
    if df is None:
        return
    
    # Baseline performance
    baseline_results = baseline_model_performance(train_df, test_df)
    
    # RT model performance  
    rt_results = evaluate_rt_model_performance(test_df)
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("🏆 FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Measurable samples: {len(df[df['Fast_unmeasurable'] == False])}")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    print(f"\n📈 Model Performance (R² scores):")
    for model_name, results in baseline_results.items():
        print(f"   {model_name}: R² = {results['r2']:.4f}, R²(log) = {results['r2_log']:.4f}")
    
    if rt_results:
        print(f"   Regression Transformer: R² = {rt_results['r2']:.4f}, R²(log) = {rt_results['r2_log']:.4f}")
        print(f"   RT Success Rate: {rt_results['success_rate']:.1f}%")
    else:
        print(f"   Regression Transformer: EVALUATION FAILED")
    
    # Expected vs Actual
    best_baseline_r2 = max(results['r2'] for results in baseline_results.values())
    rt_r2 = rt_results['r2'] if rt_results else 0.0
    
    print(f"\n🎯 Performance Analysis:")
    print(f"   Best baseline R²: {best_baseline_r2:.4f}")
    print(f"   RT model R²: {rt_r2:.4f}")
    
    if rt_r2 > 0.5:
        print(f"   Status: ✅ RT model performs well")
    elif rt_r2 > 0.1:
        print(f"   Status: ⚠️  RT model shows some correlation")
    else:
        print(f"   Status: ❌ RT model needs significant improvement")
    
    print(f"\n💡 Key Insights:")
    if rt_results and rt_results['success_rate'] < 50:
        print(f"   • Low RT success rate ({rt_results['success_rate']:.1f}%) indicates generation issues")
    if rt_r2 < 0.1:
        print(f"   • Poor R² suggests the model is not learning meaningful patterns")
        print(f"   • This confirms the property generation problems identified earlier")
    
    return {
        'dataset': df,
        'baseline': baseline_results,
        'regression_transformer': rt_results
    }


if __name__ == "__main__":
    results = comprehensive_evaluation()
