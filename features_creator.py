#!/usr/bin/env python3

'''
Create dataframe with all features created by calling features functions written in the
feature_functions files
'''

import pandas as pd
import numpy as np
from feature_functions import *


# Load our CSV Data
data = pd.read_csv('data/EURUSDhours.csv')

# Rename columns
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'AskVol']

# Change all date data
data.set_index(pd.to_datetime(data.Date),inplace=True)

# Drop date column
data.drop(columns='Date', axis=1, inplace=True)

# Drop all the duplicates data for better fitting our algorithm
prices = data.drop_duplicates(keep=False)

# Create lists for each period required by our functions (cf the research paper)

momentum_key = [3, 4, 5, 8, 9, 10]
stochastic_key = [3, 4, 5, 8, 9, 10]
williams_key = [6, 7, 8, 9, 10]
proc_key = [12, 13, 14, 15]
wadl_key = [15]
adosc_key = [2, 3, 4, 5]
macd_key = [15, 30]
cci_key = [15]
bollinger_key = [15]
heikenashi_key = [15]
paverage_key = [2]
slope_key = [3, 4, 5, 10, 20, 30]
fourier_key = [10, 20, 30]
sine_key = [5, 6]

key_list = [momentum_key, stochastic_key, williams_key, proc_key, wadl_key, adosc_key,
            macd_key, cci_key, bollinger_key, heikenashi_key, paverage_key, slope_key,
            fourier_key, sine_key]

# Create all the features (prints are for debugging)

momentum_dict = momentum(prices, momentum_key)
print('1')
stochastic_dict = stochastic(prices, stochastic_key)
print('2')
williams_dict = williams(prices, williams_key)
print('3')
proc_dict = proc(prices, proc_key)
print('4')
wadl_dict = wadl(prices, wadl_key)
print('5')
adosc_dict = adosc(prices, adosc_key)
print('6')
macd_dict = macd(prices, macd_key)
print('7')
cci_dict = cci(prices, cci_key)
print('8')
bollinger_dict = bollinger(prices, bollinger_key, 2)
print('9')

# Resample data for heikenashi
hka_prices = prices.copy()
hka_prices['Symbol'] = 'SYMB'
hka = ohlcresample(hka_prices, '15H')
heikenashi_dict = heikenashi(hka, heikenashi_key)
print('10')
paverage_dict = paverage(prices, paverage_key)
print('11')
slope_dict = slopes(prices, slope_key)
print('12')
fourier_dict = fourier(prices, fourier_key)
print('13')
sine_dict = sine(prices, sine_key)
print('14')

# Create list of dictionnaries

dict_list = [momentum_dict.close, stochastic_dict.stochclose, williams_dict.willclose,
             proc_dict.proc, wadl_dict.wadl, adosc_dict.ad, macd_dict.line, cci_dict.cci,
             bollinger_dict.bands, heikenashi_dict.candles, paverage_dict.avs,
             slope_dict.slope, fourier_dict.coeffs, sine_dict.coeffs]

# List of 'base' columns names

column_feat = ['Momentum', 'Stochastic', 'Williams', 'PROC', 'Wadl', 'ADOSC', 'MACD',
               'CCI', 'Bollinger', 'Heikenashi', 'Paverage', 'Slope', 'Fourier', 'Sine']

# Populate the DATAFRAME

dataframe = pd.DataFrame(index=prices.index)

for i, dictionnary in enumerate(dict_list):
    if column_feat[i] == 'MACD':
        col_id = column_feat[i] + str(key_list[6][0]) + str(key_list[6][1])

        dataframe[col_id] = dictionnary

    else:

        for hours in key_list[i]: # iterate through periods
            for column_name in list(dict_list[i][hours]): # iterate through all the columns of the dictionnary for the periods given
                col_id = column_feat[i] + str(hours) + column_name
                dataframe[col_id] = dict_list[i][hours][column_name]

threshold = round(0.7*len(dataframe))

dataframe[['Open', 'High', 'Low', 'Close']] = prices[['Open', 'High', 'Low', 'Close']]

# Heikenashi is resampled ==> empty data in between


dataframe.Heikenashi15Open.fillna(method='bfill', inplace=True)
dataframe.Heikenashi15High.fillna(method='bfill', inplace=True)
dataframe.Heikenashi15Low.fillna(method='bfill', inplace=True)
dataframe.Heikenashi15Close.fillna(method='bfill', inplace=True)

# Drop columns that have 30% or more NAN data

dataframe_cleaned = dataframe.copy()
dataframe_cleaned.dropna(axis=1, thresh=threshold, inplace=True)
dataframe_cleaned.dropna(axis=0)

dataframe_cleaned.to_csv('data/dataframe.csv')
