# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:13:07 2024

@author: user
"""

from entsoe import EntsoePandasClient
import pandas as pd
from pathlib import Path

def imbalance_entsoe(start, end):

    api_key = '3b124f3f-d64e-43af-be0e-a3433d07722f'  
    client = EntsoePandasClient(api_key=api_key)
    
    # Specify the country code for the Netherlands
    country_code = 'NL'
    
    # Fetch imbalance prices for the defined period
    try:
        imbalance_prices = client.query_imbalance_prices(country_code, start=start, end=end)
    
        return imbalance_prices
    except Exception as e:
        print(f"An error occurred: {e}")




# start = pd.Timestamp('2023-01-01', tz='Europe/Amsterdam')
# end = pd.Timestamp('2024-07-08', tz='Europe/Amsterdam')
# imbalance_data =  imbalance_entsoe(start, end)  





# def load_local_imbalance_data(start, end, data_folder='files/entsoe_data'):
#     folder = Path(data_folder)
#     csv_files = sorted(folder.glob("*.csv"))

#     df_list = []
#     for file in csv_files:
#         df = pd.read_csv(file)

#         # Extract start time from "31/12/2022 23:00:00 - 01/01/2023 00:00:00"
#         df['datetime'] = df.iloc[:, 0].str.extract(r'^(.*?)\s+-')[0]
#         df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S')

#         df = df.set_index('datetime')
#         df_list.append(df)

#     full_df = pd.concat(df_list)
#     full_df = full_df.sort_index()

#     # Convert input timestamps to UTC if they aren't already
#     if start.tzinfo is not None:
#         start = start.tz_convert('UTC').tz_localize(None)
#         end = end.tz_convert('UTC').tz_localize(None)

#     # Filter
#     filtered_df = full_df.loc[start:end]

#     return filtered_df

        

# start = pd.Timestamp('2023-01-01', tz='Europe/Amsterdam')
# end = pd.Timestamp('2024-07-08', tz='Europe/Amsterdam')
# imbalance_data =  load_local_imbalance_data(start, end)  