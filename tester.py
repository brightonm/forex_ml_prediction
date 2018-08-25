#!/usr/bin/env python3
'''
Getting and plotting FOREX data from Dukascopy website
Using FFN (not working with Yahoo or Google)
'''

import pandas as pd
import plotly as py
from plotly import tools
import plotly.graph_objs as go
from feature_functions import *

############## Step 1 - Load up our data & create moving average ################

# df = pd.read_csv('EURUSDhours.csv', index_col='Gmt time')
df = pd.read_csv('data/dataframe.csv')
print(df)
# df.index.names = ['Date']
# df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M:%S.%f')
#
# # Keep unique values in the dataframe, drop all the duplicates
# df.drop_duplicates(keep=False, inplace=True)
#
# # See the candles better
# df = df.iloc[:200]
#
# # moving average of the opening
# ma = df.Close.rolling(center=False, window=30).mean()

############ Step 2 - Get function data from selected function: ################

## Heiken Ashi candles
# ha_results = heikenashi(df, [1])
# ha = ha_results.candles[1]

## Detrended
# detrended = detrend(df, method='difference')

## Fourier & Sine series
# f = fourier(df, [10, 15], method='difference') # Decommented the step 3 below while testing this line and set plot to True in the function
# f = sine(df, [10, 15], method='difference') # Decommented the step 3 below while testing this line and set plot to True in the function

## Williams Accumulation Distribution
# WADL = wadl(df, [15])
# line = WADL.wadl[15]

## Data Resampling Function
# resampled = ohlcresample(df, '15H')
# resampled.index = resampled.index.droplevel(0)

## Momentum Function
# m = momentum(df, [10])
# res = m.close[10]

## Stochastic Oscillator Function
# s = stochastic(df, [14, 15])
# res = s.stochclose[14]

## Williams Oscillator Function
# w = williams(df,[15])
# res = w.willclose[15]

## PROC Function (Price Rate of Change)
# p = proc(df, [30])
# res = p.proc[30]

## Accumulation Distribution Oscillator
# ad = adosc(df, [30])
# res = ad.ad[30]

# MACD (Moving Average Convergence Divergence)
# macd = macd(df, [15, 30])
# res = macd.signal

## CCI (Commodity Channel Index)
# c = cci(df, [30])
# res = c.cci[30]

## Bollinger Bands
# boll = bollinger(df, [30], 2)
# res = boll.bands[30]

## Price Averages
# average = paverage(df, [30])
# res = average.avs[30]

## Slope Functions
# slope_f = slopes(df, [30])
# res = slope_f.slope[30]


############################## Step 3 - Plot ###################################

# trace0 = go.Ohlc(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name='Currency Quote')
# trace1 = go.Scatter(x=df.index, y=ma, name='Moving Average')
# trace2 = go.Bar(x=df.index, y=df.Volume, name='Volume')
# trace2 = go.Ohlc(x=ha.index, open=ha.Open, high=ha.High, low=ha.Low, close=ha.Close, name='Heiken Ashi')
# trace2 = go.Scatter(x=df.index, y=detrended, name='Linear Detrended')
# trace2 = go.Scatter(x=df.index, y=detrended, name='Difference Detrended')
# trace2 = go.Scatter(x=df.index, y=detrended, name='Difference Detrended')
# trace2 = go.Scatter(x=line.index, y=line.Close, name='Williams Accumulation Distribution')
# trace2 = go.Ohlc(x=resampled.index, open=resampled.Open, high=resampled.High, low=resample.Low, close=resample.Close, name='Data Resampling')
# trace2 = go.Scatter(x=res.index, y=res.Close, name='Momentum')
# trace2 = go.Scatter(x=res.index, y=res.K, name='Stochastic Oscillator')
# trace2 = go.Scatter(x=res.index, y=res.R, name='Williams Oscillator')
# trace2 = go.Scatter(x=res.index, y=res.Close, name='PROC')
# trace2 = go.Scatter(x=res.index, y=res.ad, name='Accumulation Distribution Oscillator')
# trace2 = go.Scatter(x=res.index, y=res.Close, name='Commodity Channel Index')
# trace2 = go.Scatter(x=res.index, y=res.Upper, name='Bollinger Bands')
# trace2 = go.Scatter(x=res.index, y=res.Close, name='Price Averages')
# trace2 = go.Scatter(x=res.index, y=res.High, name='Slopes')

# data = [trace0, trace1, trace2]
#
# fig = tools.make_subplots(rows=2, cols=1, shared_xaxes=True)
# fig.append_trace(trace0, 1, 1)
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 2, 1)
#
# py.offline.plot(fig, filename='test_data_resampling.html')
# py.offline.plot(fig, filename='data/test_slopes.html')
