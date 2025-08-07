#!/usr/bin/env python3
"""
Model Loading Utility
====================
Simple utility for loading and using saved models for predictions.
"""

import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class BiasPredictor:
    """Simple interface for using trained bias correction models."""
    
    def __init__(self, classifier_suffix="", regressor_suffix="", features_suffix=""):
        """
        Initialize by loading saved models.
        
        Parameters:
        -----------
        classifier_suffix : str
            Suffix for classifier and classification scaler files
        regressor_suffix : str
            Suffix for regressor and regression scaler files
        features_suffix : str
            Suffix for features file. If empty, uses classifier_suffix
        """
        self.classifier_suffix = classifier_suffix
        self.regressor_suffix = regressor_suffix
        self.features_suffix = features_suffix if features_suffix else classifier_suffix
        self.classifier = None
        self.regressor = None
        self.scaler_class = None
        self.scaler_reg = None
        self.features = None
        self.load_models()
    
    def load_models(self):
        """Load all model components."""
        try:
            # Load classifier and classification scaler
            with open(f'models/best_classifier{self.classifier_suffix}.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            
            with open(f'models/scaler_class{self.classifier_suffix}.pkl', 'rb') as f:
                self.scaler_class = pickle.load(f)
            
            # Load regressor and regression scaler
            with open(f'models/best_regressor{self.regressor_suffix}.pkl', 'rb') as f:
                self.regressor = pickle.load(f)
            
            with open(f'models/scaler_reg{self.regressor_suffix}.pkl', 'rb') as f:
                self.scaler_reg = pickle.load(f)
            
            # Load features
            with open(f'models/features{self.features_suffix}.pkl', 'rb') as f:
                self.features = pickle.load(f)
            
            print(f"Successfully loaded models:")
            print(f"  Classifier suffix: {self.classifier_suffix}")
            print(f"  Regressor suffix: {self.regressor_suffix}")
            print(f"  Features suffix: {self.features_suffix}")
            print(f"  Features: {len(self.features)}")
            
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Make sure you've trained and saved models first")
            print(f"Expected files:")
            print(f"  models/best_classifier{self.classifier_suffix}.pkl")
            print(f"  models/scaler_class{self.classifier_suffix}.pkl")
            print(f"  models/best_regressor{self.regressor_suffix}.pkl")
            print(f"  models/scaler_reg{self.regressor_suffix}.pkl")
            print(f"  models/features{self.features_suffix}.pkl")
    
    def predict_bias(self, feature_data):
        """
        Predict bias for given feature data.
        
        Parameters:
        -----------
        feature_data : dict or pd.DataFrame
            Dictionary with feature names as keys and values, or DataFrame with features as columns
        
        Returns:
        --------
        float : Predicted bias value
        """
        if self.classifier is None or self.regressor is None:
            print("Models not loaded. Cannot make predictions.")
            return 0.0
        
        # Convert to DataFrame if needed
        if isinstance(feature_data, dict):
            df = pd.DataFrame([feature_data])
        else:
            df = feature_data.copy()
        
        # Check if all required features are present
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return 0.0
        
        # Extract features in correct order
        X = df[self.features].iloc[0:1]
        
        # Scale features
        X_class_scaled = self.scaler_class.transform(X)
        X_reg_scaled = self.scaler_reg.transform(X)
        
        # Predict class first (0 = measurable, 1 = fast)
        class_pred = self.classifier.predict(X_class_scaled)[0]
        
        if class_pred == 1:  # Fast unmeasurable
            return 0.0
        
        # Predict bias magnitude
        bias_pred = self.regressor.predict(X_reg_scaled)[0]
        return max(0.0, bias_pred)  # Ensure non-negative
    
    def predict_rate_class(self, feature_data):
        """
        Predict rate class (0 = measurable, 1 = fast unmeasurable).
        
        Parameters:
        -----------
        feature_data : dict or pd.DataFrame
            Feature data
        
        Returns:
        --------
        int : Predicted class (0 or 1)
        """
        if self.classifier is None:
            print("Classifier not loaded. Cannot make predictions.")
            return 0
        
        # Convert to DataFrame if needed
        if isinstance(feature_data, dict):
            df = pd.DataFrame([feature_data])
        else:
            df = feature_data.copy()
        
        # Check features
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return 0
        
        # Extract and scale features
        X = df[self.features].iloc[0:1]
        X_scaled = self.scaler_class.transform(X)
        
        return self.classifier.predict(X_scaled)[0]
    
    def get_model_info(self):
        """Get information about loaded models."""
        if self.classifier is None:
            return "No models loaded"
        
        info = {
            'classifier_type': type(self.classifier).__name__,
            'regressor_type': type(self.regressor).__name__,
            'n_features': len(self.features),
            'features': self.features
        }
        return info

def load_models_simple(model_suffix=""):
    """
    Convenience function for loading models with the same suffix (backward compatibility).
    
    Parameters:
    -----------
    model_suffix : str
        Common suffix for all model files
    
    Returns:
    --------
    BiasPredictor : Initialized predictor with loaded models
    """
    return BiasPredictor(classifier_suffix=model_suffix, regressor_suffix=model_suffix, features_suffix=model_suffix)

def load_and_predict_batch(csv_file, classifier_suffix="", regressor_suffix="", features_suffix="", save_results=True):
    """
    Load a CSV file and predict bias for all rows.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with feature data
    classifier_suffix : str
        Suffix for classifier files
    regressor_suffix : str
        Suffix for regressor files
    features_suffix : str
        Suffix for features file
    save_results : bool
        Whether to save results to file
    
    Returns:
    --------
    pd.DataFrame : Original data with predictions added
    """
    
    # Load data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows from {csv_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Initialize predictor
    predictor = BiasPredictor(classifier_suffix, regressor_suffix, features_suffix)
    
    if predictor.classifier is None:
        print("Failed to load models")
        return None
    
    # Make predictions
    print("Making predictions...")
    df['predicted_bias'] = 0.0
    df['predicted_class'] = 0
    
    for idx in df.index:
        row_data = df.loc[idx:idx]  # Get single row as DataFrame
        
        try:
            bias_pred = predictor.predict_bias(row_data)
            class_pred = predictor.predict_rate_class(row_data)
            
            df.loc[idx, 'predicted_bias'] = bias_pred
            df.loc[idx, 'predicted_class'] = class_pred
            
        except Exception as e:
            print(f"Error predicting for row {idx}: {e}")
            continue
    
    print(f"Predictions complete")
    print(f"  Average predicted bias: {df['predicted_bias'].mean():.3f}")
    print(f"  Predicted classes: {df['predicted_class'].value_counts().to_dict()}")
    
    # Save results
    if save_results:
        output_file = csv_file.replace('.csv', '_with_predictions.csv')
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df

def example_usage():
    """Example of how to use the loaded models."""
    print("=== Example Usage ===")
    
    # Example 1: Load models with same suffix for both classifier and regressor
    print("\n--- Option 1: Same suffix for both models ---")
    predictor = BiasPredictor(classifier_suffix="_each_27", regressor_suffix="_each_27")
    
    # Example 2: Load models with different suffixes
    print("\n--- Option 2: Different suffixes ---")
    # predictor = BiasPredictor(
    #     classifier_suffix="_classification_model", 
    #     regressor_suffix="_regression_model",
    #     features_suffix="_each_27"  # Optional, defaults to classifier_suffix
    # )
    
    if predictor.classifier is None:
        print("No models found. Train models first using main.py")
        print("\nExample file naming:")
        print("  models/best_classifier_each_27.pkl")
        print("  models/best_regressor_each_27.pkl") 
        print("  models/scaler_class_each_27.pkl")
        print("  models/scaler_reg_each_27.pkl")
        print("  models/features_each_27.pkl")
        return
    
    # Print model info
    print("\nModel Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        if key != 'features':
            print(f"  {key}: {value}")
    
    print(f"\nRequired features ({len(info['features'])}):")
    for i, feature in enumerate(info['features']):
        print(f"  {i+1:2d}. {feature}")
    
    # Example prediction (you would need to provide actual feature values)
    print("\nExample prediction:")
    print("To make predictions, provide a dictionary with all required features")
    print("Example: predictor.predict_bias({'acyl_feature1': 1.0, 'amine_feature1': 2.0, ...})")
    
    # Example batch prediction
    print("\nExample batch prediction:")
    print("load_and_predict_batch('data.csv', '_class_model', '_reg_model', '_features')")

if __name__ == "__main__":
    example_usage() 