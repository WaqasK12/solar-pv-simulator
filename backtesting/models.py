# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:21:51 2024

@author: user
"""
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .preprocessing import prepare_features
import numpy as np
import pandas as pd

def train_xgboost_with_best_params(X, y, best_params):
    """
    Train an XGBoost model using the best hyperparameters obtained from RandomizedSearchCV.
    """
    # Initialize XGBoost model with the best parameters
    xgb_model = xgb.XGBRegressor(**best_params, verbosity=1)
    
    # Fit the model
    xgb_model.fit(X, y)
    
    return xgb_model

def get_best_hyperparameters(X, y):
    """
    Perform RandomizedSearchCV to get the best hyperparameters for XGBoost, supporting both 'gbtree' and 'gblinear' boosters.
    """
    # Define hyperparameter grid
    params = {
        'booster': ['gbtree'],  # Include both boosters

        'objective': ['reg:squarederror'],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'n_estimators': [10, 25, 50, 100, 300, 500],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 10, 15, 20],
        # # Tree-specific parameters
        # 'max_depth': [3, 4, 5, 6, 10],  # Only used for 'gbtree'
        # 'min_child_weight': [2, 3, 4, 5, 6, 7],  # Only used for 'gbtree'
        # 'subsample': [0.6, 0.8, 1.0],  # Only used for 'gbtree'
        # 'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],  # Only used for 'gbtree'
        # 'gamma': [0, 0.01, 0.1],  # Only used for 'gbtree'
        # Linear-specific parameters
        'max_delta_step': [0, 1, 5],  # Can be used for 'gblinear'
    }

    # Initialize model for RandomizedSearchCV
    xgb_model = xgb.XGBRegressor(verbosity=1)
    
    # Perform RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=xgb_model, 
        param_distributions=params, 
        n_iter=20,
        cv=5, 
        scoring='neg_root_mean_squared_error', 
        verbose=1, 
        n_jobs=-1
    )
    randomized_search.fit(X, y)
    
    # Return the best parameters
    return randomized_search.best_params_

def rolling_backtest(
    combined_data,
    target_col,
    train_window=365 * 96,
    forecast_horizon=96,
    data_lag_steps=96
):
    """
    Perform a rolling backtest with a generic data lag between training and test periods.
    
    Parameters:
        combined_data (DataFrame): Full dataset with features and target.
        target_col (str): Name of the target column.
        train_window (int): Number of time steps to use for training (e.g., 1 year).
        forecast_horizon (int): Number of steps to forecast in each iteration (e.g., 1 day = 96).
        data_lag_steps (int): How many steps after training end to wait before forecasting (e.g., 1 day = 96).
    """
    results = []

    for forecast_start in range(
        train_window,
        len(combined_data) - forecast_horizon - data_lag_steps + 1,
        forecast_horizon
    ):
        # Define train and test ranges with lag between
        train_start_idx = forecast_start - train_window
        train_end_idx = forecast_start
        test_start_idx = forecast_start + data_lag_steps
        test_end_idx = test_start_idx + forecast_horizon

        train_data = combined_data.iloc[train_start_idx:train_end_idx]
        test_data = combined_data.iloc[test_start_idx:test_end_idx]

        print(f"\nTrain until     : {train_data.index[-1]}")
        print(f"Forecasting from: {test_data.index[0]} to {test_data.index[-1]}")

        # Prepare features
        train_features = prepare_features(train_data, target_col)
        test_features = prepare_features(test_data, target_col)

        if train_features.empty or test_features.empty:
            print("Skipping due to empty features.")
            continue

        X_train, y_train = train_features.drop(columns=['target']), train_features['target']
        X_test, y_test = test_features.drop(columns=['target']), test_features['target']

        best_params = get_best_hyperparameters(X_train, y_train)
        model = train_xgboost_with_best_params(X_train, y_train, best_params)

        y_pred = model.predict(X_test)
        y_pred = pd.Series(y_pred, index=test_data.index)

        rmse, mae, nrmse = calculate_error_metrics(y_test, y_pred)

        results.append({
            'iteration': forecast_start,
            'train_end': train_data.index[-1],
            'forecast_start': test_data.index[0],
            'forecast_end': test_data.index[-1],
            'y_true': y_test.values,
            'y_pred': y_pred,
            'test_indices': test_data.index,  # ✅ restore this line
            'RMSE': rmse,
            'MAE': mae,
            'NRMSE': nrmse
        })


        print(f"RMSE = {rmse:.4f}, MAE = {mae:.4f}, NRMSE = {nrmse:.4f}")

    return results


# def rolling_backtest_old(combined_data, target_col,train_window=365 * 96, forecast_horizon=192, data_lag_train=96):
#     """
#     Perform rolling backtest using an XGBoost model with hyperparameter tuning and retraining in each iteration.
#     """
#     results = []


#     for start_idx in range(train_window, len(combined_data) - forecast_horizon, forecast_horizon):
#         # Train data should grow by forecast_horizon each time
#         end_train_idx = start_idx + forecast_horizon
#         train_data = combined_data.iloc[0:end_train_idx]
#         test_data = combined_data.iloc[end_train_idx:end_train_idx + forecast_horizon]

#         # Debugging: Check train and test data size
#         print(f"Iteration {start_idx}: Train size: {len(train_data)}, Test size: {len(test_data)}")
        
#         # Prepare features for train and test data
#         train_features = prepare_features(train_data, target_col)
#         test_features = prepare_features(test_data, target_col)
        
#         if train_features.empty or test_features.empty:
#             print(f"Empty features at iteration {start_idx}. Skipping.")
#             continue

#         X_train, y_train = train_features.drop(columns=['target']), train_features['target']
#         X_test, y_test = test_features.drop(columns=['target']), test_features['target']

#         # Retrain the model using the best hyperparameters
#         # Get best hyperparameters for the current training window
#         best_params = get_best_hyperparameters(X_train, y_train)
        
#         # Retrain the model using the best hyperparameters
#         model = train_xgboost_with_best_params(X_train, y_train, best_params)


#         # Make predictions using the trained model
#         y_pred = model.predict(X_test)
        
#         y_pred = pd.Series(y_pred, index=test_data.index)

        
#         # Calculate error metrics
#         rmse, mae, nrmse = calculate_error_metrics(y_test, y_pred)

#         # Store the results including error metrics
#         results.append({
#             'iteration': start_idx,
#             'y_true': y_test.values,
#             'y_pred': y_pred,
#             'test_indices': test_data.index,  # Store test indices
#             'RMSE': rmse,
#             'MAE': mae,
#             'NRMSE': nrmse
#         })

#         # Print error metrics for this iteration
#         print(f"Iteration {start_idx}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, NRMSE = {nrmse:.4f}")

#     return results



def calculate_error_metrics(y_true, y_pred):
    """
    Calculate and return common error metrics for regression models.
    """
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate NRMSE (Normalized RMSE)
    nrmse = rmse / np.mean(y_true) if np.mean(y_true) != 0 else np.nan  # Avoid division by zero
    
    return rmse, mae, nrmse
