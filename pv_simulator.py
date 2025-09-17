# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:02:51 2024

@author: 190908
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
import pytz
from datetime import datetime
import numpy as np
import os

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def round_up_to_next_15_minutes(dt):
    # Round up to the next 15-minute interval
    minute = (dt.minute // 15 + 1) * 15
    if minute == 60:
        minute = 0
        dt += timedelta(hours=1)
    return dt.replace(minute=minute, second=0, microsecond=0)

def get_weather_data(latitude, longitude,tilt, azimuth, start_date="2024-01-01", end_date="2025-01-01"):

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
    	"latitude": latitude,
    	"longitude": longitude,
    	"start_date": start_date,
    	"end_date": end_date,
    	# "hourly": ["temperature_2m", "wind_speed_10m", "global_tilted_irradiance"],
    	"minutely_15": ["temperature_2m", "wind_speed_10m", "global_tilted_irradiance"],
    	"wind_speed_unit": "ms",
    	"timezone": "auto",
    	"tilt":tilt,
    	"azimuth": azimuth,
    	"models": "best_match",
        "timezone": "GMT",
    }
    responses = openmeteo.weather_api(url, params=params)
    
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    
    
    # Processing minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_wind_speed_10m = minutely_15.Variables(1).ValuesAsNumpy()
    minutely_15_global_tilted_irradiance = minutely_15.Variables(2).ValuesAsNumpy()

    
    minutely_15_data = {"date": pd.date_range(
    	start = pd.to_datetime(minutely_15.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(minutely_15.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = minutely_15.Interval()),
    	inclusive = "left"
    )}
    minutely_15_data["Temp"] = minutely_15_temperature_2m
    minutely_15_data["WS"] = minutely_15_wind_speed_10m
    minutely_15_data["GTI"] = minutely_15_global_tilted_irradiance

    
    minutely_15_dataframe = pd.DataFrame(data = minutely_15_data)

    # Converting the 'date' column to datetime and set it as the index
    minutely_15_dataframe["date"] = pd.to_datetime(minutely_15_dataframe["date"])
    minutely_15_dataframe.set_index("date", inplace=True)
    
    return minutely_15_dataframe



def map_azimuth_to_api_angle(conventional_angle):
    """
    Converting conventional compass direction (0° = North, 90° = East, 180° = South, 270° = West)
    to the azimuth system (0° = South, -90° = East, 90° = West, 180° or -180° = North) defined by the open meteo api.

    """
    # Apply the conversion formula
    azimuth_angle = (conventional_angle - 180) % 360
    
    # Normalize the result to keep it within the range of -180° to 180°
    if azimuth_angle > 180:
        azimuth_angle -= 360
    
    return azimuth_angle
        

def handle_special_cases(tilt_list, azimuth_list):
    corrected_tilts = []
    corrected_azimuths = []
    for tilt, azimuth in zip(tilt_list, azimuth_list):
        if pd.isna(tilt) and pd.isna(azimuth):
            corrected_tilts.append(30.0)  # default values
            corrected_azimuths.append(180.0)
        else:
            corrected_tilts.append(tilt)
            corrected_azimuths.append(azimuth)
    return corrected_tilts, corrected_azimuths



def calculate_pv_yield(df_weather, years_operational, kWp_DC, name, tilt_angle):
    U0 = 25
    U1 = 6.84
    DCAE = 0.98
    CSD = 0.99
    YYD = 0.005
    PPCTE = 0.003

    df_weather['PCT'] = df_weather['Temp'] + (df_weather['GTI'] / (U0 + (U1 * df_weather['WS'])))
    YDP = 1 - (years_operational * YYD)
    df_weather[name] = ((1 - ((df_weather['PCT'] - 25) * PPCTE)) * (df_weather['GTI'] * kWp_DC) / 1000) * DCAE * CSD * YDP

    return df_weather



def calculate_combined_pv_yield(df_weathers, years_operational, kWp_DCs, kWp_AC, kWp_ACs=None):
    combined_df = None

    # Check if kWp_ACs is provided and is a list
    if kWp_ACs is not None and isinstance(kWp_ACs, list):
        assert len(kWp_ACs) == len(df_weathers), "Length of kWp_ACs must match the number of weather dataframes."

    for i, (df_weather, kWp_DC) in enumerate(zip(df_weathers, kWp_DCs)):
        if df_weather is None or df_weather.empty:
            print(f"Warning: Weather dataframe {i} is empty or None.")
            continue  
        orientation_name = f"PV_yield_DC{i}"
        df_weather = calculate_pv_yield(df_weather, years_operational, kWp_DC, name=orientation_name, tilt_angle=None)  # tilt is handled within df_weather

        if combined_df is None:
            combined_df = df_weather.copy()
            combined_df[f"Global_Tilted_Irradiance_{i}"] = df_weather['GTI']
            # combined_df[f"PCT_{i}"] = df_weather['PCT']
            combined_df[f"PV_yield_DC{i}"] = df_weather[orientation_name]
            combined_df['Total_PV_yield_DC'] = df_weather[orientation_name]
        else:
            combined_df[f"Global_Tilted_Irradiance_{i}"] = df_weather['GTI']
            # combined_df[f"PCT_{i}"] = df_weather['PCT']
            combined_df[f"PV_yield_DC{i}"] = df_weather[orientation_name]
            combined_df['Total_PV_yield_DC'] += df_weather[orientation_name]
    
    

    # Clip individual yields if kWp_ACs is provided and contains non-zero values
    if kWp_ACs and any(float(kWp) > 0 for kWp in kWp_ACs):
        for i, kWp_AC_I in enumerate(kWp_ACs):
            if i < len(df_weathers):  # Ensure we don't go out of bounds
                combined_df[f"PV_yield_AC{i}"] = combined_df[f"PV_yield_DC{i}"].clip(upper=kWp_AC_I)
        combined_df['Total_PV_yield_AC'] = combined_df[[f"PV_yield_AC{i}" for i in range(len(df_weathers))]].sum(axis=1)
    # Skip clipping if both kWp_AC and kWp_ACs are zero or None
    elif (kWp_ACs is None or all(float(kWp) == 0 for kWp in kWp_ACs)) and (kWp_AC is None or kWp_AC == 0):
        print("Skipping clipping for Total_PV_yield_AC as both individual inverter capacities and total inverter capacities are zero or None.")
    # Clip total yield if kWp_AC is provided and no individual clipping is done
    elif kWp_AC is not None and kWp_AC > 0:
        combined_df['Total_PV_yield_AC'] = combined_df['Total_PV_yield_DC'].clip(upper=kWp_AC)

    # List of columns to keep, excluding 'Total_PV_yield_DC' and 'Total_PV_yield_AC'
    column_order = [col for col in combined_df.columns if col not in ['Total_PV_yield_DC', 'Total_PV_yield_AC']]
    
    # Always append 'Total_PV_yield_DC'
    column_order.append('Total_PV_yield_DC')
    
    # Append 'Total_PV_yield_AC' only if it exists in the DataFrame
    if 'Total_PV_yield_AC' in combined_df.columns:
        column_order.append('Total_PV_yield_AC')
    
    # Reorder the DataFrame columns
    combined_df = combined_df[column_order]
    
    combined_df = combined_df.drop(columns=['PCT', 'GTI'], errors='ignore')

    

    # combined_df.to_csv("combined_df.csv")
    return combined_df




def save_to_excel(result_df):
    """Save data to Excel inside the 'data' folder with current datetime in the filename."""
    if result_df is None:
        print("No data to save. Please generate results first.")
        return
    
    
    # Check if the index is a timezone-aware datetime and convert it to timezone-unaware
    if isinstance(result_df.index, pd.DatetimeIndex) and result_df.index.tz is not None:
        result_df.index = result_df.index.tz_localize(None)
    
    # Ensure that any timezone-aware datetime columns are converted to timezone-unaware
    for col in result_df.select_dtypes(include=['datetime']):
        if result_df[col].dt.tz is not None:
            result_df[col] = result_df[col].dt.tz_localize(None)
    
    # Get the current datetime in a string format (e.g., '2025-03-28_1530')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    # Create a directory called 'data' if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Create a filename with the timestamp and specify the 'data' folder
    filename = f"data/PVsimulation_data_{timestamp}.xlsx"
    
    # Save to Excel
    result_df.to_excel(filename, index=True)
    print(f"Data saved to {filename}")




    