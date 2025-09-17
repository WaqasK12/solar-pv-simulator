# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:16:28 2024

@author: user
"""
import pandas as pd
import holidays


def prepare_features(combined_data, target_col):
    features = pd.DataFrame()

    weather_features = ["temperature_2m", "relative_humidity_2m", "direct_normal_irradiance_instant"]
    for feature in weather_features:
        features[feature] = combined_data[feature]
    
    # Time-based features
    features["hour"] = combined_data.index.hour
    features["day_of_week"] = combined_data.index.dayofweek
    features["month"] = combined_data.index.month
    
    # # Holiday indicator for the Netherlands
    # nl_holidays = holidays.Netherlands(years=combined_data.index.year.unique())  
    # features["is_holiday"] = combined_data.index.date.isin(nl_holidays.keys()).astype(int)
        
    # Target and lag features
    features["target"] = combined_data[target_col]
    # features["lag_1_week"] = combined_data[target_col].shift(96 * 2)  # 96 intervals per day * 7 days
    
    features = features.dropna()
    if features.empty:
        print("Warning: No rows after dropna in prepare_features.")
    return features
