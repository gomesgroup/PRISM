#!/usr/bin/env python3
"""
Data Processing Module
=====================
Handles data loading, feature processing, and correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_hte_data(analysis_type='bias_correction', target_col='bias'):
    """Load HTE data and calculate bias metrics."""
    
    if analysis_type == 'bias_correction':
        df = pd.read_csv('data/hte_rates_raw_split_into_2tests.csv', index_col=0)
        
        #### Calculate bias for only True Slow Unreliable
        df['is_biased'] = df['Slow_unreliable'] == True
        df['bias'] = np.where(df['is_biased'], df['Controls']*1.5 - df['HTE_rate'], 0)
        
        #### Calculate the delta between the HTE rate and NMR rate 
        # df['is_biased'] = (df['Fast_unmeasurable'] == False) & (df['Slow_unreliable'] == False) & (df['nmr_rate_2'] > 0)
        # df['bias'] = df['HTE_rate'] - df['nmr_rate_2']
        print(f"Number of biased reactions: {df['is_biased'].sum()}")
        
    elif analysis_type == 'hte_prediction':
        df = pd.read_csv('data/corrected_hte_rates.csv')
        
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
            desc = pd.read_csv(file)
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
    
    # Use RandomForest for feature selection
    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    
    n_features = min(n_top, len(feature_cols))
    selector = SequentialFeatureSelector(
        estimator, 
        n_features_to_select=n_features,
        direction=direction,
        scoring='r2',
        cv=3
    )
    
    selector.fit(X, y)
    selected_features = list(X.columns[selector.get_support()])
    
    print(f"Selected {len(selected_features)} features using sequential selection")
    return selected_features

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