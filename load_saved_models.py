#!/usr/bin/env python3
"""
Load Saved Models Utility
=========================
Simple utility to load and use trained models from the main directory.
"""

import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the model loading utilities
from src.load_models import BiasPredictor, load_models_simple, load_and_predict_batch

def main():
    """Example usage of loading saved models."""
    print("=== Load Saved Models Utility ===")
    print("\nThis utility demonstrates how to load and use trained models.")
    print("Make sure you have trained models saved in the models/ directory first.")
    
    # Try to load models with common suffix
    print("\nAttempting to load models with suffix '_each_27'...")
    try:
        predictor = load_models_simple("_each_27")
        
        if predictor.classifier is not None:
            print("✅ Models loaded successfully!")
            
            # Display model information
            info = predictor.get_model_info()
            print(f"\nModel Information:")
            print(f"  Classifier: {info['classifier_type']}")
            print(f"  Regressor: {info['regressor_type']}")
            print(f"  Number of features: {info['n_features']}")
            
            print(f"\nRequired features:")
            for i, feature in enumerate(info['features'], 1):
                print(f"  {i:2d}. {feature}")
            
            print(f"\nTo make predictions, use:")
            print(f"  predicted_bias = predictor.predict_bias(feature_dict)")
            print(f"  predicted_class = predictor.predict_rate_class(feature_dict)")
            
        else:
            print("❌ Failed to load models")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"\nCheck that you have model files in the models/ directory:")
        print(f"  - models/best_classifier_each_27.pkl")
        print(f"  - models/best_regressor_each_27.pkl")
        print(f"  - models/scaler_class_each_27.pkl")
        print(f"  - models/scaler_reg_each_27.pkl")
        print(f"  - models/features_each_27.pkl")
    
    print(f"\n" + "="*50)
    print("Usage Examples:")
    print("="*50)
    
    print("""
# Option 1: Same suffix for all models
from src.load_models import load_models_simple
predictor = load_models_simple("_each_27")

# Option 2: Different suffixes
from src.load_models import BiasPredictor
predictor = BiasPredictor(
    classifier_suffix="_classification_model",
    regressor_suffix="_regression_model", 
    features_suffix="_features"
)

# Make predictions
feature_data = {
    'acyl_feature1': 1.0,
    'amine_feature1': 2.0,
    # ... all required features
}
predicted_bias = predictor.predict_bias(feature_data)
predicted_class = predictor.predict_rate_class(feature_data)

# Batch prediction from CSV
from src.load_models import load_and_predict_batch
results = load_and_predict_batch(
    'new_data.csv',
    classifier_suffix='_class_model',
    regressor_suffix='_reg_model'
)
""")

if __name__ == "__main__":
    main() 