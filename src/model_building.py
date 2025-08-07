#!/usr/bin/env python3
"""
Model Building Module
====================
Handles classification and regression model training with hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_combined_dataset(acid_feature_data, amine_feature_data, df, selected_features, save_df=False):
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
    
    # Handle missing values
    feature_cols = acid_features + amine_features
    for col in feature_cols:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(combined_df[col].median())

    if save_df:
        combined_df.to_csv(f'results/combined_df_{len(feature_cols)}_features.csv', index=False)
    
    return combined_df, feature_cols

def build_classification_model(combined_df, features, hyperparameter_optimization=False): 
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
    
    # Build model
    if hyperparameter_optimization:
        print("Running hyperparameter optimization for classification...")
        best_model, best_model_name = optuna_classification_optimization(X_train_scaled, y_train, X_test_scaled, y_test)
    else:
        # Try different classifiers and select best based on CV F1 score
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced', loss_function='Logloss'),
            'LightGBM': LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        }
        
        best_cv_score = -1
        best_model = None
        best_model_name = None
        
        print("Evaluating different classifiers...")
        for name, model in classifiers.items():
            try:
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Cross-validation F1 score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
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

def build_regression_model(combined_df, features, hyperparameter_optimization=False):
    """Build regression model to predict bias magnitude."""
    print("=== Building Regression Model ===")

    #### For Filter to only biased cases for regression
    # biased_df = combined_df[combined_df['is_biased'] == True].copy()
    #### Instead, use all data to learn from not biased cases
    #### But, Filter for all measurable cases (rate class = 0)
    measurable_df = combined_df[combined_df['Fast_unmeasurable'] == False].copy()
    
    #### For NMR rates
    # measurable_df = combined_df[(combined_df['Fast_unmeasurable'] == False) & 
    #                             (combined_df['Slow_unreliable'] == False) & 
    #                             (combined_df['nmr_rate_2'] > 0)].copy()
    
    if len(measurable_df) == 0:
        print("No measurable biased cases found for regression")
        return None, None, {}
    
    y = measurable_df['bias']
    X = measurable_df[features]
    
    print(f"Regression dataset: {X.shape}")
    print(f"Bias range: {y.min():.3f} to {y.max():.3f}")
    
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
    
    # Build model
    if hyperparameter_optimization:
        print("Running hyperparameter optimization for regression...")
        best_model, best_model_name = optuna_regression_optimization(X_train_scaled, y_train, X_test_scaled, y_test)
    else:
        # Try different regressors and select best based on CV R2 score
        regressors = {
            'Bayesian Ridge': BayesianRidge(),
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=200, random_state=42, objective='reg:squaredlogerror'), #
            'Ridge': Ridge(random_state=42),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0),
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'LightGBM': LGBMRegressor(random_state=42, verbose=-1)
        }
        
        best_cv_score = -float('inf')
        best_model = None
        best_model_name = None
        
        print("Evaluating different regressors...")
        for name, model in regressors.items():
            try:
                # Fit model
                model.fit(X_train_scaled, y_train)
                
                # Cross-validation R2 score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
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
    
    # Evaluate model
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    
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
    
    return best_model, scaler, results

def build_models(acid_feature_data, amine_feature_data, df, selected_features, 
                single_run=False, hyperparameter_optimization=False):
    """Build both classification and regression models."""
    print(f"=== Building Models with {len(selected_features)} features ===")
    print(f"Selected features: {selected_features}")
    
    # Create combined dataset
    combined_df, valid_features = create_combined_dataset(
        acid_feature_data, amine_feature_data, df, selected_features, save_df=single_run
    )

    if not valid_features:
        print("No valid features found")
        return None, None, None, None, [], {}, None, None, None
    
    print(f"Valid features: {len(valid_features)}")
    
    # Build classification model
    classifier, scaler_class, class_results = build_classification_model(
        combined_df, valid_features, hyperparameter_optimization
    )
    
    # Build regression model
    regressor, scaler_reg, reg_results = build_regression_model(
        combined_df, valid_features, hyperparameter_optimization
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

def optuna_classification_optimization(X_train, y_train, X_val, y_val, n_trials=50):
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
        else:  # lgbm
            model = LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                verbose=-1
            )
        
        model.fit(X_train, y_train)
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
    else:
        best_model_name = 'LightGBM'
        best_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
    
    best_model.fit(X_train, y_train)
    return best_model, best_model_name

def optuna_regression_optimization(X_train, y_train, X_val, y_val, n_trials=50):
    """Optimize regression model hyperparameters using Optuna."""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'xgb', 'lgbm', 'bayesian_ridge'])
        
        if model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                random_state=42
            )
        elif model_type == 'xgb':
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                objective='reg:squaredlogerror'
            )
        elif model_type == 'lgbm':
            model = LGBMRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                random_state=42,
                verbose=-1
            )
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
    elif model_type == 'lgbm':
        best_model_name = 'LightGBM'
        best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
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