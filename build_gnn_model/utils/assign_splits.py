#!/usr/bin/env python3
"""
Create ChemProp-compatible splits file from dataset with 'test splits' column.

This script reads a CSV file with a 'test splits' column containing TRAIN/TEST1/TEST2 labels,
creates a validation set by splitting the training data, and saves the result in JSON format.
The script also includes Controls data mapping acyl chloride IDs to their control rates.
The script can be run from any directory using absolute paths.

Usage:
    python assign_splits.py input.csv --output-file output_splits.json
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import sys
import argparse
import os


def load_and_merge_features(file_paths, index_col, suffix):
    """Load, merge, clean and rename molecular descriptor files efficiently."""
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path).set_index(index_col)
            # Drop frequency columns immediately
            df = df.drop(columns=[col for col in df.columns if 'FREQS_' in col or 'FCS_' in col])
            df = df.drop(columns=[col for col in df.columns if 'name' in col or 'smiles' in col])
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {os.path.basename(path)}: {e}")
    
    if not dfs:
        return None, {}
    
    # Merge all dataframes
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, how='outer', rsuffix='_dup')
    
    # Create binary column for acyl pKa features (only for acyl suffix)
    if suffix == '_acyl':
        pka_cols = [col for col in merged.columns if 'pka_aHs' in col]
        for pka_col in pka_cols:
            binary_col = f"{pka_col}_x_has_acidic_H"
            merged[binary_col] = merged[pka_col].apply(lambda x: 0.0 if pd.isna(x) else float(x))
    
    # Process categorical variables - create binary columns for each class
    if suffix == '_amine' and 'class' in merged.columns:
        unique_classes = ["1_aliphatic", "1_aromatic", "1_mixture", "2_mixture", "2_aliphatic"]
        for class_val in unique_classes:
            merged[f'amine_class_{class_val}'] = (merged['class'] == class_val).astype(int)
        merged = merged.drop(columns=['class'])
        print(f"🏷️  Created {len(unique_classes)} binary columns for amine classes")

    if suffix == '_acyl' and 'class' in merged.columns:
        unique_classes = ["aromatic", "aliphatic", "hetero", "mixture"]
        for class_val in unique_classes:
            merged[f'acyl_class_{class_val}'] = (merged['class'] == class_val).astype(int)
        merged = merged.drop(columns=['class'])
        print(f"🏷️  Created {len(unique_classes)} binary columns for acyl classes")
    
    # Remove any remaining non-numeric columns (except identifiers we want to keep)
    categorical_cols = merged.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_drop = [col for col in categorical_cols if col not in ['name', 'smiles']]
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)
        print(f"🗑️  Dropped remaining categorical columns: {cols_to_drop}")
    
    # Keep only numerical columns (including newly encoded ones)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    merged = merged[numeric_cols]
    merged = merged.dropna(axis=1)
    
    # Add suffix to all remaining columns
    rename_dict = {col: f"{col}{suffix}" for col in merged.columns}
    
    return merged.rename(columns=rename_dict), rename_dict


def add_features(df, save_path=None, max_features=None):
    """Add molecular features efficiently with minimal code."""
    # Get absolute paths for feature files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from utils/ to updated/
    features_dir = os.path.join(project_root, 'data', 'features')
    
    # Define feature file paths using absolute paths
    acyl_paths = [os.path.join(features_dir, 'descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv'),
                  os.path.join(features_dir, 'descriptors_acyl_chlorides.csv')]
    amine_paths = [os.path.join(features_dir, 'descriptors_amines_morfeus_addn_w_xtb.csv'),
                   os.path.join(features_dir, 'descriptors_amines.csv')]
    
    # Load and process features
    acyl_features, acyl_dict = load_and_merge_features(acyl_paths, 'acyl_chlorides', '_acyl')
    amine_features, amine_dict = load_and_merge_features(amine_paths, 'amines', '_amine')
    
    if acyl_features is None and amine_features is None:
        print("❌ No features loaded")
        return df
    
    # Apply feature selection if max_features is specified
    if max_features and max_features > 0:
        all_features = pd.concat([acyl_features, amine_features], axis=1)
        if len(all_features.columns) > max_features:
            # Select features with highest variance (most informative)
            feature_vars = all_features.var().sort_values(ascending=False)
            selected_cols = feature_vars.head(max_features).index.tolist()
            
            # Split back into acyl and amine
            acyl_selected = [col for col in selected_cols if col in acyl_features.columns]
            amine_selected = [col for col in selected_cols if col in amine_features.columns]
            
            if acyl_selected:
                acyl_features = acyl_features[acyl_selected]
            if amine_selected:
                amine_features = amine_features[amine_selected]
            
            print(f"🎯 Selected {len(selected_cols)} most informative features (max: {max_features})")
    
    # Merge with main dataframe
    result = df.copy()
    total_features = 0
    
    if acyl_features is not None and 'acyl_chlorides' in df.columns:
        result = result.merge(acyl_features, left_on='acyl_chlorides', right_index=True, how='left')
        total_features += len(acyl_dict)
        print(f"✓ Added {len(acyl_dict)} acyl features")
    
    if amine_features is not None and 'amines' in df.columns:
        result = result.merge(amine_features, left_on='amines', right_index=True, how='left')
        total_features += len(amine_dict)
        print(f"✓ Added {len(amine_dict)} amine features")
    
    
    print(f"📊 Final shape: {result.shape} (+{total_features} features)")
    
    if save_path:
        result.to_csv(save_path, index=False)
        print(f"💾 Saved to: {save_path}")
    
    return result


def create_splits(csv_file, output_file, val_ratio=0.2, random_state=42, add_features_flag=True, y_column='corrected_HTE_rate_all'):
    """
    Create splits from a CSV with 'test splits' column.
    Creates reaction keys in format 'rxn_'+acyl_chloride+'_'+amine paired with target y values.
    Also includes Controls data and molecular features for each reaction.
    
    Args:
        csv_file: Path to CSV file with 'test splits' and 'Controls' columns
        output_file: Path to save JSON splits file
        val_ratio: Fraction of training data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        add_features_flag: Whether to add molecular features (default: True)
        y_column: Name of the target column to use as y values (default: 'corrected_HTE_rate_all')
    """
    
    # Load dataset
    print(f"Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")

    # Load molecular features for inclusion in JSON
    features_dict = {}
    if add_features_flag:
        if 'acyl_chlorides' in df.columns and 'amines' in df.columns:
            print(f"\nLoading molecular features for JSON inclusion...")
            df_with_features = add_features(df.copy())
            
            # Extract feature columns (exclude original columns)
            original_cols = set(df.columns)
            feature_cols = [col for col in df_with_features.columns if col not in original_cols]
            
            # Create features dictionary for each reaction
            for idx, row in df_with_features.iterrows():
                rxn_key = 'rxn_' + str(row['acyl_chlorides']) + '_' + str(row['amines'])
                features_dict[rxn_key] = {col: row[col] for col in feature_cols if pd.notna(row[col])}
            
            print(f"✓ Loaded features for {len(features_dict)} reactions with {len(feature_cols)} feature columns")
            
            # Optionally save CSV with features
            base_name = os.path.splitext(csv_file)[0]
            feats_file = f"{base_name}_feats.csv"
            df_with_features.to_csv(feats_file, index=False)
            print(f"💾 Also saved features to CSV: {feats_file}")
        else:
            print("Warning: 'acyl_chlorides' or 'amines' columns not found. Skipping feature addition.")
    else:
        print("Skipping feature addition (use --add-features to include molecular descriptors)")

    # Check for required columns
    required_columns = ['test splits', 'acyl_chlorides', 'amines', y_column, 'Controls']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns not found: {missing_columns}. Available columns: {list(df.columns)}")
    
    # Create reaction keys in the format 'rxn_'+acyl_chloride+'_'+amine
    df['reaction_key'] = 'rxn_' + df['acyl_chlorides'].astype(str) + '_' + df['amines'].astype(str)
    print(f"Created {len(df['reaction_key'].unique())} unique reaction keys")
    
    # Create controls mapping: acyl_chloride -> control_rate
    controls_mapping = {}
    for idx, row in df.iterrows():
        acyl_id = row['acyl_chlorides']
        control_value = row['Controls']
        if acyl_id not in controls_mapping:
            controls_mapping[acyl_id] = control_value
    
    print(f"Created controls mapping for {len(controls_mapping)} acyl chlorides")
    
    # Get split labels
    data_splits = df['test splits']
    split_types = data_splits.unique()
    print(f"Split types found: {split_types}")
    
    # Create dictionaries for each split type with reaction_key: y_column pairs
    train_data = {}
    test1_data = {}
    test2_data = {}
    
    for idx, row in df.iterrows():
        rxn_key = row['reaction_key']
        y_value = row[y_column]
        split_type = row['test splits']
        
        if split_type == 'TRAIN':
            train_data[rxn_key] = y_value
        elif split_type == 'TEST1':
            test1_data[rxn_key] = y_value
        elif split_type == 'TEST2':
            test2_data[rxn_key] = y_value
    
    # Combine TEST1 and TEST2 into a single test set
    test_data = {**test1_data, **test2_data}
    
    print(f"\nOriginal split sizes:")
    print(f"  TRAIN: {len(train_data)}")
    print(f"  TEST1: {len(test1_data)}")
    print(f"  TEST2: {len(test2_data)}")
    print(f"  Combined TEST: {len(test_data)} (TEST1 + TEST2)")
    
    # Create validation split from training data
    train_keys = list(train_data.keys())
    train_values = [train_data[key] for key in train_keys]
    
    train_size = 1.0 - val_ratio
    train_keys_split, val_keys_split = train_test_split(
        train_keys,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )
    
    # Create final split dictionaries with combined structure
    final_train_data = {}
    final_val_data = {}
    final_test_data = {}
    
    # Create train split with rxn_, Control, and features
    for key in train_keys_split:
        acyl_id = int(key.split('_')[1])
        control_value = controls_mapping[acyl_id]
        final_train_data[key] = {
            "hte_lnk": train_data[key],
            "Control": control_value
        }
        # Add features if available (flatten features to same level as Control)
        if key in features_dict:
            final_train_data[key].update(features_dict[key])
    
    # Create val split with rxn_, Control, and features
    for key in val_keys_split:
        acyl_id = int(key.split('_')[1])
        control_value = controls_mapping[acyl_id]
        final_val_data[key] = {
            "hte_lnk": train_data[key],
            "Control": control_value
        }
        # Add features if available (flatten features to same level as Control)
        if key in features_dict:
            final_val_data[key].update(features_dict[key])
    
    # Create test split with rxn_, Control, and features
    for key in test_data.keys():
        acyl_id = int(key.split('_')[1])
        control_value = controls_mapping[acyl_id]
        final_test_data[key] = {
            "hte_lnk": test_data[key],
            "Control": control_value
        }
        # Add features if available (flatten features to same level as Control)
        if key in features_dict:
            final_test_data[key].update(features_dict[key])
    
    print(f"\nAfter creating validation split:")
    print(f"  Train: {len(final_train_data)} ({len(final_train_data)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(final_val_data)} ({len(final_val_data)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(final_test_data)} ({len(final_test_data)/len(df)*100:.1f}%)")
    
    # Create the final splits dictionary with combined structure
    splits_dict = {
        'train': final_train_data,
        'val': final_val_data,
        'test': final_test_data
    }
    
    print(f"Number of train reactions: {len(final_train_data)}")
    print(f"Number of val reactions: {len(final_val_data)}")
    print(f"Number of test reactions: {len(final_test_data)}")

    # Save to JSON file
    print(f"\nSaving splits to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(splits_dict, f, indent=2)
    
    print("✅ Splits file created successfully!")
    
    # Optional: Save separate TEST1/TEST2 info if needed for reference
    if len(test2_data) > 0:
        # Create separate test1 and test2 data with combined structure
        final_test1_data = {}
        final_test2_data = {}
        
        for key in test1_data.keys():
            acyl_id = int(key.split('_')[1])
            control_value = controls_mapping[acyl_id]
            final_test1_data[key] = {
                "hte_lnk": test1_data[key],
                "Control": control_value
            }
            # Add features if available (flatten features to same level as Control)
            if key in features_dict:
                final_test1_data[key].update(features_dict[key])
        
        for key in test2_data.keys():
            acyl_id = int(key.split('_')[1])
            control_value = controls_mapping[acyl_id]
            final_test2_data[key] = {
                "hte_lnk": test2_data[key],
                "Control": control_value
            }
            # Add features if available (flatten features to same level as Control)
            if key in features_dict:
                final_test2_data[key].update(features_dict[key])
        
        separate_file = output_file.replace('.json', '_with_separate_tests.json')
        splits_with_separate = {
            'train': final_train_data,
            'val': final_val_data,
            'test': final_test_data,  # Combined test set
            'test1_only': final_test1_data,  # Separate TEST1 for reference
            'test2_only': final_test2_data   # Separate TEST2 for reference
        }
        
        with open(separate_file, 'w') as f:
            json.dump(splits_with_separate, f, indent=2)
        
        print(f"📝 Also saved separate test info to: {separate_file}")
    
    return splits_dict


def main():
    parser = argparse.ArgumentParser(description='Create ChemProp splits from CSV with test splits column')
    parser.add_argument('csv_file', help='Input CSV file with "test splits" column')
    parser.add_argument('--output-file', '-o', help='Output JSON file for splits')
    parser.add_argument('--val-ratio', type=float, default=0.1, 
                       help='Fraction of training data for validation (default: 0.1)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--add-features', action='store_true', default=True,
                       help='Add molecular features (acyl chlorides and amines descriptors) to the JSON splits (default: True)')
    parser.add_argument('--no-features', action='store_true',
                       help='Skip adding molecular features to the JSON splits')
    parser.add_argument('--y-column', type=str, default='corrected_HTE_rate_all',
                       help='Name of the target column to use as y values (default: corrected_HTE_rate_all)')
    
    args = parser.parse_args()
    
    # Handle feature flags - default to True unless --no-features is specified
    add_features_flag = args.add_features and not args.no_features
    
    # Handle default output file path using absolute paths
    if args.output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, 'data')
        args.output_file = os.path.join(data_dir, 'hte-all-corrected_splits_train_val_tests_lnk.json')
    
    # Convert relative paths to absolute paths if needed
    if not os.path.isabs(args.output_file):
        args.output_file = os.path.abspath(args.output_file)
    
    if not os.path.isabs(args.csv_file):
        args.csv_file = os.path.abspath(args.csv_file)
    
    try:
        create_splits(
            csv_file=args.csv_file,
            output_file=args.output_file,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
            add_features_flag=add_features_flag,
            y_column=args.y_column
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
