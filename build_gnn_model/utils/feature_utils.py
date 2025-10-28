#!/usr/bin/env python3
"""
Utility functions for handling molecular features in the hyperparameter optimization workflow.
"""

import json
from typing import List


def get_additional_features(json_file_path: str) -> List[str]:
    """
    Extract all feature keys that are not 'hte_lnk' from the JSON data.
    
    This function dynamically reads the JSON file and extracts all keys except 'hte_lnk'
    which represents the target variable. All other keys are considered additional features
    that can be used as input to the model.
    
    Args:
        json_file_path: Path to the JSON file containing the split data
        
    Returns:
        Sorted list of additional feature names
        
    Example:
        >>> features = get_additional_features("data/splits_train_val_tests_lnk.json")
        >>> print(f"Found {len(features)} features")
        Found 79 features
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Get features from first train sample
        if 'train' not in data or not data['train']:
            raise ValueError(f"No training data found in {json_file_path}")
            
        first_sample = next(iter(data['train'].values()))
        addn_features = [key for key in first_sample.keys() if key != 'hte_lnk']
        
        # Sort for consistency
        addn_features = sorted(addn_features)
        
        print(f"✓ Found {len(addn_features)} additional features")
        print(f"  First 5: {addn_features[:5]}")
        if len(addn_features) > 5:
            print(f"  Last 5:  {addn_features[-5:]}")
            
        return addn_features
        
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {json_file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading features from {json_file_path}: {e}")


def get_feature_info(json_file_path: str) -> dict:
    """
    Get detailed information about features in the JSON data.
    
    Args:
        json_file_path: Path to the JSON file containing the split data
        
    Returns:
        Dictionary with feature information including counts and types
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        if 'train' not in data or not data['train']:
            raise ValueError(f"No training data found in {json_file_path}")
            
        first_sample = next(iter(data['train'].values()))
        
        # Categorize features
        molecular_features = []
        control_features = []
        other_features = []
        
        for key in first_sample.keys():
            if key == 'hte_lnk':
                continue
            elif key == 'Control':
                control_features.append(key)
            elif any(suffix in key for suffix in ['_acyl', '_amine']):
                molecular_features.append(key)
            else:
                other_features.append(key)
        
        info = {
            'total_features': len(first_sample) - 1,  # Exclude hte_lnk
            'molecular_features': sorted(molecular_features),
            'control_features': sorted(control_features),
            'other_features': sorted(other_features),
            'feature_counts': {
                'molecular': len(molecular_features),
                'control': len(control_features),
                'other': len(other_features)
            }
        }
        
        return info
        
    except Exception as e:
        raise RuntimeError(f"Error analyzing features from {json_file_path}: {e}")


if __name__ == "__main__":
    # Test the functions
    import sys
    import os
    
    # Try to find a JSON file to test with
    test_files = [
        "data/hte-all-corrected_splits_train_val_tests_lnk.json",
        "../data/hte-all-corrected_splits_train_val_tests_lnk.json",
        "data/hte-all-corrected_splits_train_val_tests_lnk_with_separate_tests.json",
        "../data/hte-all-corrected_splits_train_val_tests_lnk_with_separate_tests.json"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Testing with file: {test_file}")
            
            # Test get_additional_features
            features = get_additional_features(test_file)
            print(f"\nTotal additional features: {len(features)}")
            
            # Test get_feature_info
            info = get_feature_info(test_file)
            print(f"\nFeature breakdown:")
            print(f"  Molecular features: {info['feature_counts']['molecular']}")
            print(f"  Control features: {info['feature_counts']['control']}")
            print(f"  Other features: {info['feature_counts']['other']}")
            print(f"  Total: {info['total_features']}")
            
            break
    else:
        print("No test JSON file found. Please run from the project directory.")
