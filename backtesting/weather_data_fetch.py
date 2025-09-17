# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:12:50 2024

@author: user
"""
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd


# Historical and forecast weather data
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data( longitude, latitude, data_type="forecast", start_date=None, end_date=None):

    # Set parameters based on the data type
    if data_type == "historical":
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required for historical data.")
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT",
            "minutely_15": ["temperature_2m", "relative_humidity_2m", "direct_normal_irradiance_instant"]
        }
    elif data_type == "forecast":
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "minutely_15": ["temperature_2m", "relative_humidity_2m", "direct_normal_irradiance_instant"],
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "GMT"

            # "forecast_minutely_15": 384 ,
        }
    else:
        raise ValueError("Data type not supported. Please choose either 'historical' or 'forecast'.")

    responses = openmeteo.weather_api(url, params=params)

    # Process the response
    response = responses[0]

    # Process minutely_15 data. The order of variables needs to be the same as requested.
    minutely_15 = response.Minutely15()
    minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
    minutely_15_relative_humidity_2m = minutely_15.Variables(1).ValuesAsNumpy()
    minutely_15_direct_normal_irradiance_instant = minutely_15.Variables(2).ValuesAsNumpy()
        
    minutely_15_data = {
        "date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=False),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=False),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left"
        )
    }
    minutely_15_data["temperature_2m"] = minutely_15_temperature_2m.round().astype(int)
    minutely_15_data["relative_humidity_2m"] = minutely_15_relative_humidity_2m.round().astype(int)
    minutely_15_data["direct_normal_irradiance_instant"] = minutely_15_direct_normal_irradiance_instant.round().astype(int)
    
    minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)
    
    # Set 'tijd_nl' column as index
    minutely_15_dataframe.set_index("date", inplace=True)
    minutely_15_dataframe = minutely_15_dataframe.apply(pd.to_numeric, errors='coerce')
    
    return minutely_15_dataframe

