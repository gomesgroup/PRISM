#!/usr/bin/env python3
"""
Data Processing Module
=====================
Handles data loading, feature processing, and correlation analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import polars as pl
except Exception:
    pl = None
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_hte_data(analysis_type='bias_correction', target_col='bias'):
    """Load HTE data and calculate bias metrics."""
    
    if analysis_type == 'bias_correction':
        if pl is not None:
            df_pl = pl.read_csv('data/rates/hte_rates_raw_split_into_2tests.csv')
            df = df_pl.to_pandas()
            df.set_index(df.columns[0], inplace=True)
        else:
            df = pd.read_csv('data/rates/hte_rates_raw_split_into_2tests.csv', index_col=0)
        
        #### Calculate bias for only True Slow Unreliable
        df['is_biased'] = df['Slow_unreliable'] == True
        df['bias'] = np.where(df['is_biased'], df['Controls']*1.5 - df['HTE_rate'], 0)
        
        #### Calculate the delta between the HTE rate and NMR rate 
        # df['is_biased'] = (df['Fast_unmeasurable'] == False) & (df['Slow_unreliable'] == False) & (df['nmr_rate_2'] > 0)
        # df['bias'] = df['HTE_rate'] - df['nmr_rate_2']
        print(f"Number of biased reactions: {df['is_biased'].sum()}")
        
    elif analysis_type == 'hte_prediction':
        if pl is not None:
            df = pl.read_csv('data/rates/corrected_hte_rates.csv').to_pandas()
        else:
            df = pd.read_csv('data/rates/corrected_hte_rates.csv')
        
        #### Only measurable data
        df = df[df['Fast_unmeasurable'] == False]
        df = df[df['HTE_rate_corrected'] > 0]
        df['HTE_lnk_corrected'] = np.log10(df['HTE_rate_corrected'])
        df = df[['acyl_chlorides','amines',target_col,'test splits']]
        print(f"Number of HTE rates: {len(df)}")
    
    return df

def load_descriptors(files, index_col_name):
    """Load descriptor CSVs and handle duplicates."""
    desc_list = []
    for file in files:
        try:
            # Prefer pandas for maximum compatibility; fallback to polars
            try:
                desc = pd.read_csv(file)
            except Exception:
                if pl is None:
                    raise
                dpl = pl.read_csv(file)
                if index_col_name in dpl.columns:
                    dpl = dpl.unique(subset=[index_col_name], keep='first')
                desc = dpl.to_pandas()
            if index_col_name in desc.columns:
                desc.set_index(index_col_name, inplace=True)
                dup_count = desc.index.duplicated().sum()
                if dup_count:
                    print(f"Warning: Removed {dup_count} duplicate indices in {file}")
                    desc = desc[~desc.index.duplicated(keep='first')]
            desc_list.append(desc)
        except Exception as e:
            print(f"Could not load {file}: {e}")
    return desc_list

def load_and_process_features(df, target_col='bias'):
    """Load and process molecular descriptors for acyl chlorides and amines."""
    
    # Define file paths
    acid_files = [
        'data/features/descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv',
        'data/features/descriptors_acyl_chlorides.csv'
    ]
    amine_files = [
        'data/features/descriptors_amines_morfeus_addn_w_xtb.csv',
        'data/features/descriptors_amines.csv'
    ]
    
    # Load descriptors
    acid_desc_list = load_descriptors(acid_files, 'acyl_chlorides')
    amine_desc_list = load_descriptors(amine_files, 'amines')
    if not acid_desc_list or not amine_desc_list:
        # fallback to base loader without smiles
        return *load_and_process_features(df, target_col), {}, {}
    
    # Merge acid descriptors
    acid_descriptors = acid_desc_list[0]
    for desc in acid_desc_list[1:]:
        acid_descriptors = acid_descriptors.join(desc, how='outer', rsuffix='_dup')
    
    # Merge amine descriptors
    amine_descriptors = amine_desc_list[0]
    for desc in amine_desc_list[1:]:
        amine_descriptors = amine_descriptors.join(desc, how='outer', rsuffix='_dup')
    
    # Process conditional features
    acid_descriptors = preprocess_conditional_features(acid_descriptors, 'acyl')
    amine_descriptors = preprocess_conditional_features(amine_descriptors, 'amine')
    
    # Create feature dataframes with bias information
    acid_feature_data = create_feature_dataframe(df, acid_descriptors, 'acyl_chlorides', 'acyl_', target_col=target_col)
    amine_feature_data = create_feature_dataframe(df, amine_descriptors, 'amines', 'amine_', target_col=target_col)
    
    return acid_feature_data, amine_feature_data

def load_and_process_features_with_smiles(df, target_col='bias'):
    """Load and process descriptors plus return SMILES maps for acyl/amine.

    Returns:
      acid_feature_data, amine_feature_data, acid_smiles_map, amine_smiles_map
    """
    # Define file paths
    acid_files = [
        'data/features/descriptors_acyl_chlorides_morfeus_addn_w_xtb.csv',
        'data/features/descriptors_acyl_chlorides.csv'
    ]
    amine_files = [
        'data/features/descriptors_amines_morfeus_addn_w_xtb.csv',
        'data/features/descriptors_amines.csv'
    ]

    acid_desc_list = load_descriptors(acid_files, 'acyl_chlorides')
    amine_desc_list = load_descriptors(amine_files, 'amines')

    # Capture SMILES from the first descriptor file where available
    acid_smiles_map = {}
    for d in acid_desc_list:
        if 'smiles' in d.columns:
            acid_smiles_map = d['smiles'].astype(str).to_dict()
            break
    amine_smiles_map = {}
    for d in amine_desc_list:
        if 'smiles' in d.columns:
            amine_smiles_map = d['smiles'].astype(str).to_dict()
            break

    # Merge acid descriptors
    acid_descriptors = acid_desc_list[0]
    for desc in acid_desc_list[1:]:
        acid_descriptors = acid_descriptors.join(desc, how='outer', rsuffix='_dup')

    # Merge amine descriptors
    amine_descriptors = amine_desc_list[0]
    for desc in amine_desc_list[1:]:
        amine_descriptors = amine_descriptors.join(desc, how='outer', rsuffix='_dup')

    acid_descriptors = preprocess_conditional_features(acid_descriptors, 'acyl')
    amine_descriptors = preprocess_conditional_features(amine_descriptors, 'amine')

    acid_feature_data = create_feature_dataframe(df, acid_descriptors, 'acyl_chlorides', 'acyl_', target_col=target_col)
    amine_feature_data = create_feature_dataframe(df, amine_descriptors, 'amines', 'amine_', target_col=target_col)

    return acid_feature_data, amine_feature_data, acid_smiles_map, amine_smiles_map

def load_reaction_energy_features(filepath='data/reaction_energies/reaction_TSB_w_aimnet2.csv'):
    """Load AIMNet2-derived reaction energy/TS features and prepare for merging.

    Returns a dataframe indexed by ['acyl_chlorides','amines'] with selected numeric
    and boolean features. Column names are prefixed with 'rxn_'.
    """
    try:
        rxn = pd.read_csv(filepath)
    except Exception as e:
        print(f"Could not load reaction energy features: {e}")
        return None

    # Harmonize column names
    rename_map = {
        'acid_chlorides': 'acyl_chlorides',
        'acid_chloride': 'acyl_chlorides',
        'amines': 'amines',
    }
    for old, new in rename_map.items():
        if old in rxn.columns:
            rxn = rxn.rename(columns={old: new})

    # Keep only rows with the necessary keys
    core_cols = ['acyl_chlorides', 'amines']
    missing_keys = [c for c in core_cols if c not in rxn.columns]
    if missing_keys:
        print(f"Reaction energy file missing keys {missing_keys}; skipping integration.")
        return None

    # Selected numeric features if present
    numeric_candidates = [
        'lowest_barrier_dGTS_path_B',
        'barriers_dGTS_from_RXTS_B',
        'barriers_dGTS_from_INT1_B',
        'barriers_dGTS_from_PRDS_B',
        'rxn_dG_B',
        'imag_freq'
    ]
    present_numeric = [c for c in numeric_candidates if c in rxn.columns]

    # Boolean/status features
    bool_cols = []
    if 'used_constrained' in rxn.columns:
        # Convert to int 0/1
        rxn['used_constrained'] = rxn['used_constrained'].astype(str).str.upper().isin(['TRUE', '1', 'T']).astype(int)
        bool_cols.append('used_constrained')

    # Status one-hot or simplified valid flag
    status_cols = []
    if 'status' in rxn.columns:
        rxn['status'] = rxn['status'].astype(str)
        rxn['ts_status_valid'] = rxn['status'].str.contains('Valid TS', case=False).astype(int)
        status_cols.append('ts_status_valid')

    keep_cols = core_cols + present_numeric + bool_cols + status_cols
    rxn_small = rxn[keep_cols].copy()

    # Prefix
    feat_cols = [c for c in rxn_small.columns if c not in core_cols]
    rxn_small = rxn_small.rename(columns={c: f"rxn_{c}" for c in feat_cols})

    # Set index for efficient merge
    rxn_small = rxn_small.set_index(['acyl_chlorides', 'amines'])

    print(f"Loaded reaction energy features: {len(rxn_small)} rows, {len(feat_cols)} features")
    return rxn_small

def preprocess_conditional_features(descriptors, molecule_type='acyl'):
    # Make a copy to avoid modifying original data
    desc = descriptors.copy()
    
    # Identify conditional feature patterns
    conditional_features = {}
    
    # Features that depend on hydrogen atoms (num_Hs > 0)
    h_dependent_features = [col for col in desc.columns if 'Hs_' in col or '_Hs' in col]
    
    # Features that depend on acidic hydrogens (pKa features)
    pka_dependent_features = [col for col in desc.columns if 'pka' in col.lower() or 'pk' in col.lower()]
    
    if h_dependent_features:
        print(f"Found {len(h_dependent_features)} hydrogen-dependent features: {h_dependent_features}")
        
        # Check if we have num_Hs column
        if 'num_Hs' in desc.columns:
            conditional_features['num_Hs'] = {
                'condition_col': 'num_Hs',
                'features': h_dependent_features,
                'condition': lambda x: x > 0,
                'fill_value': 0.0
            }
        else:
            print("Warning: num_Hs column not found, using heuristic for H-dependent features")
    
    # Handle pKa features separately - they should not be filled with 0
    if pka_dependent_features:
        print(f"Found {len(pka_dependent_features)} pKa-dependent features: {pka_dependent_features}")
        
        if 'num_Hs' in desc.columns:
            # For pKa features, we'll use a different strategy
            conditional_features['pKa_features'] = {
                'condition_col': 'num_Hs',
                'features': pka_dependent_features,
                'condition': lambda x: x > 0,
                'fill_value': None,  # Special handling - don't fill with 0
                'special_handling': 'pKa'
            }
        else:
            print("Warning: num_Hs column not found, cannot properly handle pKa features")
    
    # Features that depend on specific functional groups or structural elements
    # You can extend this based on your domain knowledge
    
    # Features that depend on aromatic systems
    aromatic_features = [col for col in desc.columns if 'aromatic' in col.lower() or 'ring' in col.lower()]
    if aromatic_features:
        print(f"Found {len(aromatic_features)} aromatic-dependent features: {aromatic_features}")
    
    # Features that depend on heteroatoms
    hetero_features = [col for col in desc.columns if any(atom in col.lower() for atom in ['n_', 'o_', 's_', 'hal'])]
    if hetero_features:
        print(f"Found {len(hetero_features)} heteroatom-dependent features: {hetero_features}")
    
    # Process conditional features
    for condition_name, config in conditional_features.items():
        condition_col = config['condition_col']
        features = config['features']
        condition_func = config['condition']
        fill_value = config['fill_value']
        
        if condition_col in desc.columns:
            print(f"\nProcessing {condition_name} conditional features...")
            
            # Get molecules that meet the condition
            condition_mask = desc[condition_col].apply(condition_func)
            meets_condition = condition_mask.sum()
            total_molecules = len(desc)
            
            print(f"  {meets_condition}/{total_molecules} molecules meet condition {condition_name}")
            
            # Check if this requires special handling
            special_handling = config.get('special_handling', None)
            
            for feature in features:
                if feature in desc.columns:
                    # Count NA values before processing
                    na_before = desc[feature].isna().sum()
                    
                    if special_handling == 'pKa':
                        # Special handling for pKa features
                        print(f"    Processing pKa feature {feature} with special handling...")
                        
                        # Create binary indicator for whether molecule has acidic hydrogens
                        has_acidic_h_col = f"has_acidic_H_for_{feature}"
                        desc[has_acidic_h_col] = condition_mask.astype(int)
                        
                        # For molecules WITHOUT acidic hydrogens, create a special "no_acidic_H" value
                        # We'll use a large positive value (e.g., 50) to indicate "no acidic protons"
                        # This is chemically meaningful: no acidic H = very high pKa (very basic)
                        no_acidic_h_value = 50.0  # Very high pKa indicating no acidic protons
                        desc.loc[~condition_mask, feature] = desc.loc[~condition_mask, feature].fillna(no_acidic_h_value)
                        
                        # For molecules WITH acidic hydrogens but still have NA, use median imputation
                        condition_subset = desc.loc[condition_mask, feature]
                        if condition_subset.isna().any():
                            median_val = condition_subset.median()
                            if pd.notna(median_val):
                                desc.loc[condition_mask, feature] = condition_subset.fillna(median_val)
                            else:
                                # If all values are NA even for molecules with acidic H, use a default acyl value
                                desc.loc[condition_mask, feature] = condition_subset.fillna(10.0)  # Moderate pKa
                        
                        # Create interaction feature
                        interaction_col = f"{feature}_x_has_acidic_H"
                        desc[interaction_col] = desc[feature] * desc[has_acidic_h_col]
                        
                        print(f"      Created indicator: {has_acidic_h_col}")
                        print(f"      Created interaction: {interaction_col}")
                        print(f"      Used {no_acidic_h_value} for molecules without acidic H")
                        
                    else:
                        # Standard handling for non-pKa features
                        # For molecules that don't meet condition, fill with the specified value
                        desc.loc[~condition_mask, feature] = desc.loc[~condition_mask, feature].fillna(fill_value)
                        
                        # For molecules that meet condition but still have NA, use more sophisticated imputation
                        condition_subset = desc.loc[condition_mask, feature]
                        if condition_subset.isna().any():
                            # Use median imputation for molecules that should have this feature
                            median_val = condition_subset.median()
                            if pd.notna(median_val):
                                desc.loc[condition_mask, feature] = condition_subset.fillna(median_val)
                            else:
                                # If all values are NA even for molecules with the condition, use fill_value
                                desc.loc[condition_mask, feature] = condition_subset.fillna(fill_value)
                    
                    na_after = desc[feature].isna().sum()
                    print(f"    {feature}: {na_before} -> {na_after} NA values")
    
    # Handle remaining NA values with domain-appropriate strategies
    print(f"\nHandling remaining NA values...")
    
    for col in desc.columns:
        if desc[col].isna().any():
            na_count = desc[col].isna().sum()
            
            if pd.api.types.is_numeric_dtype(desc[col]):
                # For numeric features, use median imputation
                median_val = desc[col].median()
                if pd.notna(median_val):
                    desc[col] = desc[col].fillna(median_val)
                else:
                    # If all values are NA, fill with 0
                    desc[col] = desc[col].fillna(0.0)
                print(f"  {col}: Filled {na_count} NA values with median/zero")
            else:
                # For categorical features, use mode or 'unknown'
                mode_val = desc[col].mode()
                if len(mode_val) > 0:
                    desc[col] = desc[col].fillna(mode_val[0])
                else:
                    desc[col] = desc[col].fillna('unknown')
                print(f"  {col}: Filled {na_count} NA values with mode/unknown")
    
    # Encode categorical features
    print(f"\nEncoding categorical features...")
    categorical_features = []
    identifier_features = []
    
    # Define identifier patterns that should be excluded from modeling
    identifier_patterns = ['name', 'smiles', 'id', 'identifier', 'index']
    
    for col in desc.columns:
        if not pd.api.types.is_numeric_dtype(desc[col]):
            # Check if this is an identifier feature
            is_identifier = any(pattern in col.lower() for pattern in identifier_patterns)
            
            if is_identifier:
                identifier_features.append(col)
                # Remove identifier features completely
                desc = desc.drop(columns=[col])
                print(f"  Removed identifier feature: {col}")
            else:
                categorical_features.append(col)
                unique_values = desc[col].unique()
                print(f"  Found categorical feature {col} with {len(unique_values)} unique values: {unique_values}")
                
                # One-hot encode categorical features if they have few categories
                if len(unique_values) <= 10 and len(unique_values) > 1:
                    # Create dummy variables
                    dummies = pd.get_dummies(desc[col], prefix=col, prefix_sep='_')
                    
                    # Drop the original column and add dummy columns
                    desc = desc.drop(columns=[col])
                    desc = pd.concat([desc, dummies], axis=1)
                    print(f"    One-hot encoded {col} -> {len(dummies.columns)} features: {list(dummies.columns)}")
                
                elif len(unique_values) > 10:
                    # For high-cardinality categorical features, use label encoding
                    le = LabelEncoder()
                    desc[col + '_encoded'] = le.fit_transform(desc[col].astype(str))
                    desc = desc.drop(columns=[col])
                    print(f"    Label encoded {col} -> {col}_encoded")
                
                else:
                    # Single value - create a constant feature (will be dropped later due to zero variance)
                    desc[col + '_constant'] = 1
                    desc = desc.drop(columns=[col])
                    print(f"    Constant feature {col} -> {col}_constant")
    
    print(f"  Processed {len(categorical_features)} categorical features")
    print(f"  Removed {len(identifier_features)} identifier features: {identifier_features}")
    
    # Create feature quality indicators
    print(f"\nCreating feature quality indicators...")
    
    # Add binary indicators for whether conditional features were originally present
    for condition_name, config in conditional_features.items():
        condition_col = config['condition_col']
        features = config['features']
        
        if condition_col in desc.columns:
            # Create indicator for whether molecule has the structural element
            indicator_name = f"has_{condition_name}"
            desc[indicator_name] = desc[condition_col].apply(config['condition']).astype(int)
            print(f"  Added indicator: {indicator_name}")
            
            # Optionally create interaction features (feature * indicator)
            for feature in features[:3]:  # Limit to first 3 to avoid too many features
                if feature in desc.columns:
                    interaction_name = f"{feature}_x_{indicator_name}"
                    desc[interaction_name] = desc[feature] * desc[indicator_name]
    
    print(f"Preprocessing complete. Shape: {desc.shape}")
    print(f"Remaining NA values: {desc.isna().sum().sum()}")
    
    return desc

def create_feature_dataframe(df, descriptors, id_col, prefix, target_col='bias'):
    """Create feature dataframe with bias information."""
    if target_col == 'bias':
        feature_data = df[[id_col, target_col]].copy()
        feature_data['max_bias'] = df.groupby(id_col)[target_col].transform('max')
    else:
        feature_data = df[[id_col, target_col]].copy()
    
    # Merge with descriptors
    feature_data = feature_data.merge(descriptors, left_on=id_col, right_index=True, how='left')
    
    # Add prefix to feature columns
    desc_cols = [col for col in feature_data.columns if col not in [id_col, target_col, 'max_bias']]
    rename_dict = {col: f"{prefix}{col}" for col in desc_cols}
    feature_data = feature_data.rename(columns=rename_dict)
    
    # Drop duplicates and keep only unique molecules
    feature_data = feature_data.drop_duplicates(subset=[id_col]).set_index(id_col)
    
    return feature_data

def process_feature_correlations(df, target_col='bias', features=[], 
                                correlation_threshold=0.95, top_n_to_plot=6,
                                corr_csv_filename=None, plot_filename=None,
                                plot_title='Feature Correlations vs. Bias'):
    """Process feature correlations and remove highly correlated features."""
    
    # Select relevant features
    if features:
        available_features = [f for f in features if f in df.columns]
    else:
        available_features = [col for col in df.columns if col not in [target_col, 'max_bias']]
    
    if not available_features:
        return [], pd.DataFrame()
    
    # Calculate correlations with target
    correlations = df[available_features + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != target_col]
    
    # Remove highly correlated features
    reduced_features = []
    feature_corr_matrix = df[available_features].corr().abs()
    
    for feature in correlations.index:
        if feature in reduced_features:
            continue
        
        # Check correlation with already selected features
        is_highly_correlated = False
        for selected_feature in reduced_features:
            if feature_corr_matrix.loc[feature, selected_feature] > correlation_threshold:
                is_highly_correlated = True
                break
        
        if not is_highly_correlated:
            reduced_features.append(feature)
    
    print(f"Reduced from {len(available_features)} to {len(reduced_features)} features")
    
    # Create correlation dataframe
    corr_df = pd.DataFrame({
        'feature': reduced_features,
        'correlation': [correlations[f] for f in reduced_features]
    }).sort_values('correlation', ascending=False)
    
    # Save correlations
    if corr_csv_filename:
        corr_df.to_csv(corr_csv_filename, index=False)
    
    # Plot top correlations
    if plot_filename and len(reduced_features) > 0:
        plot_top_correlations(df, reduced_features[:top_n_to_plot], target_col, 
                            plot_filename, plot_title)
    
    return reduced_features, corr_df

def plot_top_correlations(df, features, target_col, plot_filename, plot_title):
    """Plot correlations for top features."""
    if not features:
        return
    
    n_features = min(len(features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features[:n_features]):
        if i < len(axes):
            axes[i].scatter(df[feature], df[target_col], alpha=0.6)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target_col)
            corr = df[feature].corr(df[target_col])
            axes[i].set_title(f'r = {corr:.3f}')
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

def select_features_sequentially(df, target_col='max_bias', n_top=5, direction="forward"):
    """Select features using sequential feature selection."""
    feature_cols = [col for col in df.columns if col != target_col]
    
    if len(feature_cols) == 0:
        return []
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Use RandomForest for feature selection (parallelized)
    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=max(1, os.cpu_count() or 1))
    
    n_features = min(n_top, len(feature_cols))
    selector = SequentialFeatureSelector(
        estimator, 
        n_features_to_select=n_features,
        direction=direction,
        scoring='r2',
        cv=3,
        n_jobs=-1
    )
    
    selector.fit(X, y)
    selected_features = list(X.columns[selector.get_support()])
    
    print(f"Selected {len(selected_features)} features using sequential selection")
    return selected_features

def select_features_by_correlation_fast(df, target_col: str, n_top: int = 32):
    """Select top-n features by absolute Pearson correlation with target.

    This is a fast, vectorized selector to avoid slow sequential feature selection.
    Assumes `df` contains only numeric feature columns plus the target column.
    """
    if target_col not in df.columns:
        return []

    feature_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        return []

    # Convert to numpy for vectorized correlations (optionally via jax)
    try:
        import jax.numpy as jnp  # type: ignore
        use_jax = True
    except Exception:
        jnp = None
        use_jax = False
    X_np = df[feature_cols].to_numpy(dtype=float)
    y_np = df[target_col].to_numpy(dtype=float)

    # Center and compute std
    if use_jax:
        y_mean = float(jnp.mean(y_np))
        y_center = y_np - y_mean
        y_std = float(jnp.std(y_center, ddof=1))
    else:
        y_mean = y_np.mean()
        y_center = y_np - y_mean
        y_std = y_center.std(ddof=1)
    if y_std == 0 or np.isnan(y_std):
        return feature_cols[:min(n_top, len(feature_cols))]

    if use_jax:
        X_mean = np.nanmean(X_np, axis=0)
        X_center = X_np - X_mean
        X_std = X_center.std(axis=0, ddof=1)
    else:
        X_mean = np.nanmean(X_np, axis=0)
        X_center = X_np - X_mean
        X_std = X_center.std(axis=0, ddof=1)
    # Avoid divide by zero
    X_std[X_std == 0] = 1.0

    # Compute Pearson correlation for each feature
    # corr_j = sum_i (x_ij_center * y_i_center) / ((n-1) * std_x_j * std_y)
    numerators = np.nansum(X_center * y_center[:, None], axis=0)
    denoms = (X_np.shape[0] - 1) * X_std * y_std
    corrs = numerators / denoms
    abs_corrs = np.abs(corrs)

    # Select top-n indices
    n_select = min(n_top, abs_corrs.size)
    top_idx = np.argpartition(-abs_corrs, kth=n_select - 1)[:n_select]
    # Sort selected by correlation
    top_idx_sorted = top_idx[np.argsort(-abs_corrs[top_idx])]
    top_features = [feature_cols[i] for i in top_idx_sorted]

    return top_features

def select_top_features_combined(acid_corr, amine_corr, n_features=5, include_features=[]):
    """Select top features from combined correlation analysis."""
    # Combine and sort by correlation
    combined_corr = pd.concat([acid_corr, amine_corr]).sort_values('correlation', ascending=False)
    
    # Select top features
    top_features = combined_corr['feature'].head(n_features).tolist()
    
    # Add any specifically included features
    final_features = list(set(top_features + include_features))
    
    return final_features

def analyze_bias_patterns(df, save_plot=False):
    """Analyze bias patterns in the data."""
    print("=== Bias Pattern Analysis ===")
    
    # Calculate bias statistics by acyl chloride
    acid_bias_freq = df.groupby('acyl_chlorides').agg({
        'is_biased': ['sum', 'count'],
        'bias': ['mean', 'max', 'std']
    }).round(3)
    
    acid_bias_freq.columns = ['biased_count', 'total_count', 'mean_bias', 'max_bias', 'std_bias']
    acid_bias_freq['bias_frequency'] = acid_bias_freq['biased_count'] / acid_bias_freq['total_count']
    
    print(f"Number of acyl chlorides with bias: {(acid_bias_freq['biased_count'] > 0).sum()}")
    print(f"Average bias frequency: {acid_bias_freq['bias_frequency'].mean():.3f}")
    
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(acid_bias_freq['total_count'], acid_bias_freq['bias_frequency'])
        plt.xlabel('Total Reactions')
        plt.ylabel('Bias Frequency')
        plt.title('Bias Patterns by Acyl Chloride')
        plt.savefig('plots/bias_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return acid_bias_freq 