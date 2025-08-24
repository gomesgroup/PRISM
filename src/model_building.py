#!/usr/bin/env python3
"""
Model Building Module
====================
Handles classification and regression model training with hyperparameter optimization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, GroupKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector as SklearnSFS
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
try:
    # For catching GPU-not-enabled errors and falling back to CPU
    from lightgbm.basic import LightGBMError  # type: ignore
except Exception:  # pragma: no cover
    class LightGBMError(Exception):
        pass
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import StackingRegressor
import optuna
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features, save_df=False,
                            rxn_features=None, add_interactions=True):
    """Create combined dataset for modeling."""
    # Get feature columns for each molecule type
    acid_features = [f for f in selected_features if f.startswith('acyl_')]
    amine_features = [f for f in selected_features if f.startswith('amine_')]
    
    # Merge features with main dataframe
    combined_df = df.copy()
    
    # Add acid features
    if acid_features:
        acid_data = acid_feature_data[acid_features]
        combined_df = combined_df.merge(acid_data, left_on='acyl_chlorides', right_index=True, how='left')
    
    # Add amine features
    if amine_features:
        amine_data = amine_feature_data[amine_features]
        combined_df = combined_df.merge(amine_data, left_on='amines', right_index=True, how='left')
    
    # Optionally merge reaction-energy features (prefixed 'rxn_') on the reaction pair
    if rxn_features is not None and isinstance(rxn_features, (pd.DataFrame,)) and len(rxn_features) > 0:
        try:
            combined_df = combined_df.merge(
                rxn_features,
                left_on=['acyl_chlorides', 'amines'],
                right_index=True,
                how='left'
            )
        except Exception as e:
            print(f"Warning: could not merge reaction features: {e}")

    # Add a few simple interaction descriptors if inputs exist
    if add_interactions:
        try:
            # Use columns only if present in the combined dataframe after merges
            if 'amine_pka_basic' in combined_df.columns and 'acyl_pka_lowest' in combined_df.columns:
                combined_df['int_pka_basic_minus_pka_lowest'] = (
                    combined_df['amine_pka_basic'] - combined_df['acyl_pka_lowest']
                )
            if 'amine_pka_basic' in combined_df.columns and 'acyl_pka_aHs' in combined_df.columns:
                combined_df['int_pka_basic_minus_pka_aHs'] = (
                    combined_df['amine_pka_basic'] - combined_df['acyl_pka_aHs']
                )
            if 'amine_global_N' in combined_df.columns and 'acyl_global_E' in combined_df.columns:
                combined_df['int_globalN_minus_globalE'] = (
                    combined_df['amine_global_N'] - combined_df['acyl_global_E']
                )
            if 'amine_BV_secondary_avg' in combined_df.columns and 'acyl_BV_secondary_2' in combined_df.columns:
                combined_df['int_BV_secondary_prod'] = (
                    combined_df['amine_BV_secondary_avg'] * combined_df['acyl_BV_secondary_2']
                )
        except Exception as e:
            print(f"Warning: could not create interaction features: {e}")

    # Handle missing values
    # Gather all model feature candidates that are present in dataframe
    feature_cols = [c for c in (acid_features + amine_features) if c in combined_df.columns]
    # Add any rxn_ or int_ features
    feature_cols += [c for c in combined_df.columns if c.startswith('rxn_') or c.startswith('int_')]
    # Ensure uniqueness
    feature_cols = list(dict.fromkeys(feature_cols))
    # Impute using TRAIN-only medians to avoid leakage if 'test splits' exists
    if 'test splits' in combined_df.columns and len(feature_cols) > 0:
        train_mask = combined_df['test splits'] == 'TRAIN'
        train_medians = combined_df.loc[train_mask, feature_cols].median(numeric_only=True)
        combined_df[feature_cols] = combined_df[feature_cols].fillna(train_medians)
    else:
        for col in feature_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    if save_df:
        combined_df.to_csv(f'results/combined_df_{len(feature_cols)}_features.csv', index=False)
    
    return combined_df, feature_cols

def build_classification_model(combined_df, features, hyperparameter_optimization=False,
                               use_gpu=False, early_stopping=False, group_by=None,
                               train_only_feature_selection=False, train_sfs_n=0,
                               allowed_classifiers=None, n_jobs=None): 
    """Build classification model to predict if reaction has measurable rate."""
    print("=== Building Classification Model ===")
    
    # Create binary target: 0 = measurable, 1 = fast unmeasurable
    y = combined_df['Fast_unmeasurable'].astype(int)
    X = combined_df[features]
    
    print(f"Classification dataset: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split data based on "test splits" column
    train_mask = combined_df['test splits'] == 'TRAIN'
    test_mask = (combined_df['test splits'] == 'TEST1') | (combined_df['test splits'] == 'TEST2')

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\n\nTraining data: {len(y_train)} samples")
    print(f"Test data: {len(y_test)} samples\n\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optional train-only feature selection to avoid leakage
    selected_train_features = list(features)
    if train_only_feature_selection:
        try:
            num_feats = len(features)
            if num_feats < 2:
                print("SFS skipped for classification (insufficient features)")
            else:
                # Ensure n_features_to_select < n_features
                default_cap = max(1, num_feats - 1)
                n_select_raw = train_sfs_n if train_sfs_n and train_sfs_n > 0 else min(default_cap, 30)
                n_select = max(1, min(n_select_raw, default_cap))
                clf_for_sfs = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
                sfs = SklearnSFS(
                    clf_for_sfs,
                    n_features_to_select=n_select,
                    direction='forward',
                    scoring='f1',
                    cv=3,
                    n_jobs=-1
                )
                sfs.fit(X_train, y_train)
                mask = sfs.get_support()
                selected_train_features = list(X.columns[mask])
                X = combined_df[selected_train_features]
                X_train = X[train_mask]
                X_test = X[test_mask]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                print(f"Train-only SFS selected {len(selected_train_features)} features for classification")
        except Exception as e:
            print(f"SFS (classification) failed, using provided features. Error: {e}")

    # Build model
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)
    if hyperparameter_optimization:
        print("Running hyperparameter optimization for classification...")
        best_model, best_model_name = optuna_classification_optimization(
            X_train_scaled, y_train, X_test_scaled, y_test, use_gpu=use_gpu
        )
    else:
        # Try different classifiers and select best based on CV F1 score
        xgb_kwargs = {'random_state': 42, 'eval_metric': 'logloss'}
        lgbm_kwargs = {'random_state': 42, 'class_weight': 'balanced', 'verbose': -1, 'n_jobs': n_jobs}
        cat_kwargs = {'random_state': 42, 'verbose': 0, 'auto_class_weights': 'Balanced', 'loss_function': 'Logloss'}
        if use_gpu:
            xgb_kwargs['tree_method'] = 'gpu_hist'
            xgb_kwargs['predictor'] = 'gpu_predictor'
            # Keep LightGBM on CPU to avoid GPU build issues
        xgb_kwargs['n_jobs'] = n_jobs

        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=n_jobs),
            'XGBoost': XGBClassifier(**xgb_kwargs),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
            'CatBoost': CatBoostClassifier(**cat_kwargs, thread_count=n_jobs),
            'LightGBM': LGBMClassifier(**lgbm_kwargs)
        }
        if allowed_classifiers:
            classifiers = {k: v for k, v in classifiers.items() if k in allowed_classifiers}
        
        best_cv_score = -1
        best_model = None
        best_model_name = None
        
        print("Evaluating different classifiers...")
        # Prepare grouping for CV once
        groups_train = None
        if group_by in ['acyl', 'amine', 'pair']:
            if group_by == 'acyl':
                groups_train = combined_df[train_mask]['acyl_chlorides']
            elif group_by == 'amine':
                groups_train = combined_df[train_mask]['amines']
            else:
                groups_train = combined_df[train_mask]['acyl_chlorides'].astype(str) + "_" + combined_df[train_mask]['amines'].astype(str)
        # Optional early stopping split
        if early_stopping:
            try:
                from sklearn.model_selection import train_test_split as _tts
                X_tr, X_val, y_tr, y_val = _tts(X_train_scaled, y_train, test_size=0.15, random_state=42, stratify=y_train)
            except Exception:
                X_tr, X_val, y_tr, y_val = X_train_scaled, X_test_scaled, y_train, y_test
        else:
            X_tr, X_val, y_tr, y_val = X_train_scaled, X_test_scaled, y_train, y_test

        for name, model in classifiers.items():
            try:
                # Set GPU params for LGBM / CatBoost if requested
                if use_gpu:
                    if isinstance(model, LGBMClassifier):
                        try:
                            model.set_params(device='gpu')
                        except Exception:
                            pass
                    if isinstance(model, CatBoostClassifier):
                        try:
                            model.set_params(task_type='GPU', devices='0,1,2,3')
                        except Exception:
                            pass
                # Fit model
                if early_stopping and isinstance(model, (XGBClassifier, LGBMClassifier, CatBoostClassifier)):
                    if isinstance(model, XGBClassifier):
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                    elif isinstance(model, LGBMClassifier):
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='logloss')
                    else:  # CatBoost
                        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Cross-validation F1 score
                if groups_train is not None:
                    cv = GroupKFold(n_splits=5)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1', groups=groups_train, n_jobs=n_jobs)
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1', n_jobs=n_jobs)
                cv_mean = cv_scores.mean()
                
                print(f"  {name}: CV F1 = {cv_mean:.3f} ± {cv_scores.std():.3f}")
                
                if cv_mean > best_cv_score:
                    best_cv_score = cv_mean
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"  {name}: Failed - {str(e)}")
                continue
        
        print(f"Best classifier: {best_model_name} (CV F1 = {best_cv_score:.3f})")
    
    # Evaluate model
    if best_model is None:
        print("No valid classifier was trained.")
        return None, scaler, {}
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    
    # Get predictions for visualization
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='f1')
    
    print(f"Classification Results:")
    print(f"  Train Accuracy: {train_score:.3f}")
    print(f"  Test Accuracy: {test_score:.3f}")
    print(f"  CV F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    results = {
        'model': best_model_name,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return best_model, scaler, results

def build_regression_model(combined_df, features, y_col='bias', hyperparameter_optimization=False,
                           use_gpu=False, early_stopping=False, group_by=None, enable_stacking=False,
                           train_only_feature_selection=False, train_sfs_n=0,
                           allowed_regressors=None, n_jobs=None,
                           use_bagging_ensemble=False, bagging_estimators=0, bagging_max_samples=1.0,
                           bagging_random_state=42):
    """Build regression model to predict bias magnitude."""
    print("=== Building Regression Model ===")

    #### For Filter to only biased cases for regression
    # biased_df = combined_df[combined_df['is_biased'] == True].copy()
    #### Instead, use all data to learn from not biased cases
    #### But, Filter for all measurable cases (rate class = 0)
    if y_col == 'bias':
        measurable_df = combined_df[combined_df['Fast_unmeasurable'] == False].copy()
    else:
        measurable_df = combined_df.copy()
    
    #### For NMR rates
    # measurable_df = combined_df[(combined_df['Fast_unmeasurable'] == False) & 
    #                             (combined_df['Slow_unreliable'] == False) & 
    #                             (combined_df['nmr_rate_2'] > 0)].copy()
    
    if len(measurable_df) == 0:
        print("No measurable biased cases found for regression")
        return None, None, {}
    
    y = measurable_df[y_col]
    X = measurable_df[features]
    
    print(f"Regression dataset: {X.shape}")
    print(f"Target range: {y.min():.3f} to {y.max():.3f}")
    
    # Split data based on "test splits" column
    train_mask = measurable_df['test splits'] == 'TRAIN'
    test_mask = (measurable_df['test splits'] == 'TEST1') | (measurable_df['test splits'] == 'TEST2')

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\n\nTraining data: {len(y_train)} samples")
    print(f"Test data: {len(y_test)} samples\n\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optional train-only feature selection to avoid leakage
    selected_train_features = list(features)
    if train_only_feature_selection:
        try:
            num_feats = len(features)
            if num_feats < 2:
                print("SFS skipped for regression (insufficient features)")
            else:
                # Ensure n_features_to_select < n_features
                default_cap = max(1, num_feats - 1)
                n_select_raw = train_sfs_n if train_sfs_n and train_sfs_n > 0 else min(default_cap, 40)
                n_select = max(1, min(n_select_raw, default_cap))
                reg_for_sfs = RandomForestRegressor(n_estimators=300, random_state=42)
                sfs = SklearnSFS(
                    reg_for_sfs,
                    n_features_to_select=n_select,
                    direction='forward',
                    scoring='r2',
                    cv=3,
                    n_jobs=-1
                )
                sfs.fit(X_train, y_train)
                mask = sfs.get_support()
                selected_train_features = list(X.columns[mask])
                X = measurable_df[selected_train_features]
                X_train = X[train_mask]
                X_test = X[test_mask]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                print(f"Train-only SFS selected {len(selected_train_features)} features for regression")
        except Exception as e:
            print(f"SFS (regression) failed, using provided features. Error: {e}")

    # Build model
    if n_jobs is None:
        n_jobs = max(1, os.cpu_count() or 1)
    if hyperparameter_optimization:
        print("Running hyperparameter optimization for regression...")
        best_model, best_model_name = optuna_regression_optimization(
            X_train_scaled, y_train, X_test_scaled, y_test, target_col=y_col, use_gpu=use_gpu
        )
    else:
        # Try different regressors and select best based on CV R2 score
        xgb_kwargs = {'n_estimators': 800, 'random_state': 42, 'objective': 'reg:squaredlogerror' if y_col == 'bias' else 'reg:squarederror'}
        lgbm_kwargs = {'n_estimators': 1500, 'random_state': 42, 'verbose': -1, 'n_jobs': n_jobs}
        cat_kwargs = {'random_state': 42, 'verbose': 0}
        if use_gpu:
            xgb_kwargs['tree_method'] = 'gpu_hist'
            xgb_kwargs['predictor'] = 'gpu_predictor'
            # Keep LightGBM on CPU to avoid GPU build issues
            cat_kwargs['task_type'] = 'GPU'
        xgb_kwargs['n_jobs'] = n_jobs

        regressors = {
            'Bayesian Ridge': BayesianRidge(),
            'Random Forest': RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=n_jobs),
            'Extra Trees': ExtraTreesRegressor(n_estimators=600, random_state=42, n_jobs=n_jobs),
            'XGBoost': XGBRegressor(**xgb_kwargs),
            'CatBoost': CatBoostRegressor(**cat_kwargs, thread_count=n_jobs),
            'LightGBM': LGBMRegressor(**lgbm_kwargs),
            'SVR': SVR(kernel='rbf', C=3.0, epsilon=0.1, gamma='scale'),
            'Kernel Ridge': KernelRidge(alpha=1.0, kernel='rbf', gamma=None),
            'MLP': MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                                random_state=42, max_iter=300, early_stopping=True),
            'Ridge': Ridge(random_state=42),
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42)
        }

        # Optionally include Gaussian Process (skip for large datasets for tractability)
        try:
            # Always allow GP if explicitly requested via allowed_regressors; otherwise, be conservative on size
            allow_gp = True
            if allowed_regressors is None and len(y_train) > 1200:
                allow_gp = False
            if allow_gp:
                gp_kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-2)
                regressors['Gaussian Process'] = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-6, normalize_y=True, random_state=42)
            else:
                print("Skipping Gaussian Process (dataset too large)")
        except Exception:
            pass
        if allowed_regressors:
            regressors = {k: v for k, v in regressors.items() if k in allowed_regressors}
        
        best_cv_score = -float('inf')
        best_model = None
        best_model_name = None
        
        print("Evaluating different regressors...")
        groups_train = None
        if group_by in ['acyl', 'amine', 'pair']:
            if group_by == 'acyl':
                groups_train = measurable_df['acyl_chlorides']
            elif group_by == 'amine':
                groups_train = measurable_df['amines']
            else:
                groups_train = measurable_df['acyl_chlorides'].astype(str) + "_" + measurable_df['amines'].astype(str)
        # Early stopping split
        if early_stopping:
            try:
                from sklearn.model_selection import train_test_split as _tts
                X_tr, X_val, y_tr, y_val = _tts(X_train_scaled, y_train, test_size=0.15, random_state=42)
            except Exception:
                X_tr, X_val, y_tr, y_val = X_train_scaled, X_test_scaled, y_train, y_test
        else:
            X_tr, X_val, y_tr, y_val = X_train_scaled, X_test_scaled, y_train, y_test

        for name, model in regressors.items():
            try:
                # Set GPU params for CatBoost if requested
                if use_gpu:
                    if isinstance(model, CatBoostRegressor):
                        try:
                            model.set_params(task_type='GPU')
                        except Exception:
                            pass
                # Fit model
                if early_stopping and isinstance(model, (XGBRegressor, LGBMRegressor, CatBoostRegressor)):
                    if isinstance(model, XGBRegressor):
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                    elif isinstance(model, LGBMRegressor):
                        try:
                            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                        except Exception:
                            # GPU not enabled; retry CPU
                            try:
                                model.set_params(device_type='cpu')
                            except Exception:
                                try:
                                    model.set_params(device='cpu')
                                except Exception:
                                    pass
                            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                    else:  # CatBoost
                        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Cross-validation R2 score
                if groups_train is not None:
                    cv = GroupKFold(n_splits=5)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2', groups=groups_train[train_mask], n_jobs=n_jobs)
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=n_jobs)
                cv_mean = cv_scores.mean()
                
                print(f"  {name}: CV R² = {cv_mean:.3f} ± {cv_scores.std():.3f}")
                
                if cv_mean > best_cv_score:
                    best_cv_score = cv_mean
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"  {name}: Failed - {str(e)}")
                continue
        
        print(f"Best regressor: {best_model_name} (CV R² = {best_cv_score:.3f})")

        # Optional stacking for potential boost
        if enable_stacking:
            try:
                base_estimators = []
                # Prefer heterogeneous strong learners
                if 'LightGBM' in regressors:
                    base_estimators.append(('lgbm', regressors['LightGBM']))
                if 'XGBoost' in regressors:
                    base_estimators.append(('xgb', regressors['XGBoost']))
                if 'CatBoost' in regressors:
                    base_estimators.append(('cat', regressors['CatBoost']))
                if 'Extra Trees' in regressors:
                    base_estimators.append(('etr', regressors['Extra Trees']))

                final = Ridge(random_state=42)
                stacker = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=final,
                    passthrough=False,
                    n_jobs=None
                )
                # Evaluate stacking
                if groups_train is not None:
                    cv = GroupKFold(n_splits=5)
                    stack_cv = cross_val_score(
                        stacker, X_train_scaled, y_train, cv=cv, scoring='r2',
                        groups=groups_train[train_mask], n_jobs=n_jobs
                    )
                else:
                    stack_cv = cross_val_score(stacker, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=n_jobs)
                print(f"Stacking CV R² = {stack_cv.mean():.3f} ± {stack_cv.std():.3f}")
                if stack_cv.mean() > best_cv_score:
                    best_model = stacker
                    best_model_name = 'Stacking(LGBM+XGB+Cat+ETR→Ridge)'
                    best_cv_score = stack_cv.mean()
            except Exception as e:
                print(f"Stacking failed: {e}")
    
    # Optionally wrap best model in a bagging ensemble for UQ and robustness
    if use_bagging_ensemble and bagging_estimators and bagging_estimators >= 2:
        base_estimator = best_model
        try:
            # Use single-threaded bagging to avoid multiprocessing crashes with GPU libs
            best_model = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=bagging_estimators,
                max_samples=bagging_max_samples,
                bootstrap=True,
                random_state=bagging_random_state,
                n_jobs=1
            )
            best_model.fit(X_train_scaled, y_train)
            best_model_name = f"Bagging({best_model_name} x{bagging_estimators})"
            print(f"Wrapped best model with Bagging ensemble: {best_model_name}")
        except Exception as e:
            print(f"Bagging ensemble failed; using single model. Error: {e}")
            best_model = base_estimator

    # Persist simple per-split predictions for downstream analysis will be added after evaluation

    # Evaluate model
    if best_model is None:
        print("No valid regressor was trained.")
        return None, scaler, {}
    # Compute scores explicitly to avoid edge cases in overridden .score
    from sklearn.metrics import r2_score as _r2
    try:
        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, scaler, {}
    train_score = _r2(y_train, train_pred)
    test_score = _r2(y_test, test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"Regression Results:")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    results = {
        'model': best_model_name,
        'train_r2': train_score,
        'test_r2': test_score,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }
    # Attach predictions for downstream saving
    try:
        results['y_train_pred'] = train_pred
        results['y_test_pred'] = test_pred
    except Exception:
        pass
    
    return best_model, scaler, results

def build_models(acid_feature_data, amine_feature_data, df, selected_features, 
                single_run=False, hyperparameter_optimization=False,
                use_gpu=False, early_stopping=False, group_by=None, enable_stacking=False,
                rxn_features=None, train_only_feature_selection=False, train_sfs_n=0,
                n_jobs=None):
    """Build both classification and regression models."""
    print(f"=== Building Models with {len(selected_features)} features ===")
    print(f"Selected features: {selected_features}")
    
    # Create combined dataset
    combined_df, valid_features = create_combined_dataset(
        acid_feature_data, amine_feature_data, df, selected_features, save_df=single_run,
        rxn_features=rxn_features, add_interactions=True
    )

    if not valid_features:
        print("No valid features found")
        return None, None, None, None, [], {}, None, None, None
    
    print(f"Valid features: {len(valid_features)}")
    
    # Build classification model
    classifier, scaler_class, class_results = build_classification_model(
        combined_df, valid_features, hyperparameter_optimization,
        use_gpu=use_gpu, early_stopping=early_stopping, group_by=group_by,
        train_only_feature_selection=train_only_feature_selection, train_sfs_n=train_sfs_n,
        n_jobs=n_jobs
    )
    
    # Build regression model
    regressor, scaler_reg, reg_results = build_regression_model(
        combined_df, valid_features, y_col='bias', hyperparameter_optimization=hyperparameter_optimization,
        use_gpu=use_gpu, early_stopping=early_stopping, group_by=group_by, enable_stacking=enable_stacking,
        train_only_feature_selection=train_only_feature_selection, train_sfs_n=train_sfs_n,
        n_jobs=n_jobs
    )
    
    # Create prediction functions
    def predict_bias_func(acid_id, amine_id):
        """Predict bias for a given acid-amine pair."""
        try:
            # Get features for this pair
            row_data = combined_df[
                (combined_df['acyl_chlorides'] == acid_id) & 
                (combined_df['amines'] == amine_id)
            ]
            
            if len(row_data) == 0:
                return 0.0
            
            X = row_data[valid_features].iloc[0:1]
            X_scaled = scaler_reg.transform(X)
            
            # First predict if it needs bias correction
            class_pred = classifier.predict(scaler_class.transform(X))[0]
            
            if class_pred == 1:  # Fast unmeasurable
                return 0.0
            
            # Predict bias magnitude
            bias_pred = regressor.predict(X_scaled)[0]
            return max(0.0, bias_pred)  # Ensure non-negative
            
        except Exception as e:
            print(f"Error predicting bias for {acid_id}-{amine_id}: {e}")
            return 0.0
    
    def predict_class_func(acid_id, amine_id):
        """Predict rate class for a given acid-amine pair."""
        try:
            row_data = combined_df[
                (combined_df['acyl_chlorides'] == acid_id) & 
                (combined_df['amines'] == amine_id)
            ]
            
            if len(row_data) == 0:
                return 0
            
            X = row_data[valid_features].iloc[0:1]
            X_scaled = scaler_class.transform(X)
            
            return classifier.predict(X_scaled)[0]
            
        except Exception as e:
            print(f"Error predicting class for {acid_id}-{amine_id}: {e}")
            return 0
    
    # Combine results with proper structure
    combined_results = {
        'classification': class_results,
        'regression': reg_results
    }
    
    return (classifier, regressor, scaler_class, scaler_reg, valid_features, 
            combined_results, predict_bias_func, combined_df, predict_class_func)

def build_hte_prediction_models(acid_feature_data, amine_feature_data, df, selected_features, 
                target_col, single_run=False, hyperparameter_optimization=False,
                use_gpu=False, early_stopping=False, group_by=None, enable_stacking=False,
                rxn_features=None, train_only_feature_selection=False, train_sfs_n=0,
                n_jobs=None, allowed_regressors=None,
                use_bagging_ensemble=False, bagging_estimators=0, bagging_max_samples=1.0):
    """Build regression model for HTE prediction."""
    print(f"=== Building HTE Prediction Models with {len(selected_features)} features ===")
    print(f"Selected features: {selected_features}")
    
    # Create combined dataset
    combined_df, valid_features = create_combined_dataset(
        acid_feature_data, amine_feature_data, df, selected_features, save_df=single_run,
        rxn_features=rxn_features, add_interactions=True
    )

    if not valid_features:
        print("No valid features found")
        return None, None, None, None, [], {}, None, None, None
    
    print(f"Valid features: {len(valid_features)}")

    # Build regression model
    regressor, scaler_reg, reg_results = build_regression_model(
        combined_df, valid_features, y_col=target_col, hyperparameter_optimization=hyperparameter_optimization,
        use_gpu=use_gpu, early_stopping=early_stopping, group_by=group_by, enable_stacking=enable_stacking,
        train_only_feature_selection=train_only_feature_selection, train_sfs_n=train_sfs_n,
        allowed_regressors=allowed_regressors, n_jobs=n_jobs,
        use_bagging_ensemble=use_bagging_ensemble, bagging_estimators=bagging_estimators,
        bagging_max_samples=bagging_max_samples
    )
    final_results = {
        'regression': reg_results
    }
    
    return (regressor, scaler_reg, valid_features, combined_df, final_results)


def optuna_classification_optimization(X_train, y_train, X_val, y_val, n_trials=50, use_gpu=False):
    """Optimize classification model hyperparameters using Optuna."""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgbm'])
        
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                random_state=42
            )
        elif model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                eval_metric='logloss'
            )
            if use_gpu:
                model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
        else:  # lgbm
            model = LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                verbose=-1
            )
            if use_gpu:
                try:
                    model.set_params(device_type='gpu')
                except Exception:
                    try:
                        model.set_params(device='gpu')
                    except Exception:
                        pass
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            # Fallback for LightGBM GPU builds not available
            if model_type == 'lgbm' and use_gpu:
                try:
                    model.set_params(device_type='cpu')
                except Exception:
                    try:
                        model.set_params(device='cpu')
                    except Exception:
                        pass
                model.fit(X_train, y_train)
            else:
                raise
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Train best model
    best_params = study.best_params
    model_type = best_params.pop('model_type')
    
    if model_type == 'rf':
        best_model_name = 'Random Forest'
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_type == 'xgb':
        best_model_name = 'XGBoost'
        best_model = XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        if use_gpu:
            best_model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
    else:
        best_model_name = 'LightGBM'
        best_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
        if use_gpu:
            try:
                best_model.set_params(device_type='gpu')
            except Exception:
                try:
                    best_model.set_params(device='gpu')
                except Exception:
                    pass
    
    try:
        best_model.fit(X_train, y_train)
    except Exception as e:
        if model_type == 'lgbm' and use_gpu:
            try:
                best_model.set_params(device_type='cpu')
            except Exception:
                try:
                    best_model.set_params(device='cpu')
                except Exception:
                    pass
            best_model.fit(X_train, y_train)
        else:
            raise
    return best_model, best_model_name

def optuna_regression_optimization(X_train, y_train, X_val, y_val, target_col='bias', n_trials=50, use_gpu=False):
    """Optimize regression model hyperparameters using Optuna."""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgbm', 'cat', 'bayesian_ridge'])
        
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                random_state=42
            )
        elif model_type == 'xgb':
            # Use reg:squaredlogerror for 'bias' target, reg:squarederror for log-transformed targets
            objective = 'reg:squaredlogerror' if target_col == 'bias' else 'reg:squarederror'
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                objective=objective
            )
            if use_gpu:
                model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
        elif model_type == 'lgbm':
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                verbose=-1
            )
            # LightGBM stays on CPU regardless of use_gpu
        elif model_type == 'cat':
            model = CatBoostRegressor(
                iterations=trial.suggest_int('iterations', 200, 800),
                depth=trial.suggest_int('depth', 4, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.02, 0.3),
                random_state=42,
                loss_function='RMSE',
                verbose=False
            )
            if use_gpu:
                try:
                    model.set_params(task_type='GPU')
                except Exception:
                    pass
        else:  # bayesian ridge
            model = BayesianRidge(
                alpha_1=trial.suggest_float('alpha_1', 1e-6, 1e-3),
                alpha_2=trial.suggest_float('alpha_2', 1e-6, 1e-3),
                lambda_1=trial.suggest_float('lambda_1', 1e-6, 1e-3),
                lambda_2=trial.suggest_float('lambda_2', 1e-6, 1e-3)
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return r2_score(y_val, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Train best model
    best_params = study.best_params
    model_type = best_params.pop('model_type')
    
    if model_type == 'rf':
        best_model_name = 'Random Forest'
        best_model = RandomForestRegressor(**best_params, random_state=42)
    elif model_type == 'xgb':
        best_model_name = 'XGBoost'
        best_model = XGBRegressor(**best_params, random_state=42)
        if use_gpu:
            best_model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
    elif model_type == 'lgbm':
        best_model_name = 'LightGBM'
        best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
        # LightGBM stays on CPU regardless of use_gpu
    elif model_type == 'cat':
        best_model_name = 'CatBoost'
        best_model = CatBoostRegressor(**best_params, random_state=42, loss_function='RMSE', verbose=False)
        if use_gpu:
            try:
                best_model.set_params(task_type='GPU')
            except Exception:
                pass
    else:
        best_model_name = 'Bayesian Ridge'
        best_model = BayesianRidge(**best_params)
    
    best_model.fit(X_train, y_train)
    return best_model, best_model_name

def save_models(classifier, regressor, scaler_class, scaler_reg, features, suffix=""):
    """Save trained models and scalers."""
    import os
    os.makedirs('models', exist_ok=True)
    
    model_files = {
        f'models/best_classifier{suffix}.pkl': classifier,
        f'models/best_regressor{suffix}.pkl': regressor,
        f'models/scaler_class{suffix}.pkl': scaler_class,
        f'models/scaler_reg{suffix}.pkl': scaler_reg,
        f'models/features{suffix}.pkl': features
    }
    
    for filename, obj in model_files.items():
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    
    print(f"Models saved with suffix: {suffix}")

def load_models(suffix=""):
    """Load trained models and scalers."""
    model_files = [
        f'models/best_classifier{suffix}.pkl',
        f'models/best_regressor{suffix}.pkl', 
        f'models/scaler_class{suffix}.pkl',
        f'models/scaler_reg{suffix}.pkl',
        f'models/features{suffix}.pkl'
    ]
    
    loaded_objects = []
    for filename in model_files:
        try:
            with open(filename, 'rb') as f:
                loaded_objects.append(pickle.load(f))
        except FileNotFoundError:
            print(f"File {filename} not found")
            return None
    
    return loaded_objects 