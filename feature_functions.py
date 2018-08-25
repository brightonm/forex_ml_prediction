#!/usr/bin/env python3

'''
Script of all features functions described in this Reasearch Paper :
http://www.wseas.us/e-library/conferences/2011/Penang/ACRE/ACRE-05.pdf
'''

import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime

class Holder:
    '''
    Store the result of our functions
    '''
    pass


# Heiken Ashi Candles
def heikenashi(prices, periods):
    '''

    Those candles avoid the noise of normal candles (cf plotting)

    :param prices: pandas dataframe of OHLC & Volume data created in the file tester.py

    :param periods: list of the hours we want the feature to be applied on, for instance [1, 2, 3, 6]
                    (cf the research paper), period for which to create the candles, 1 = 1 hour

    :return:  Heiken Ashi OHCL candles an Holder object called results and containing the attribute
              called candles which is a dictionnary (its keys are the periods)

    '''

    # Create our object
    results = Holder()

    # Create an empty dictionnary to store our actual dataframe
    dict = {}

    # Initialize heinken ashi close (cf formulas)
    ha_close = prices[['Open', 'High', 'Low', 'Close']].sum(axis=1)/4

    # My way to create a empty pandas series that we'll update later
    ha_open = ha_close.copy()
    ha_open.iloc[0] = ha_close.iloc[0]

    # Initialize ha_high ha_low
    ha_high = ha_close.copy()
    ha_low = ha_close.copy()

    # Fill ha_open, ha_high and ha_low according to formulas
    for day in range(1, len(prices)):
        ha_open.iloc[day] = (ha_open.iloc[day-1] + ha_close.iloc[day-1])/2
        ha_high.iloc[day] = np.array([prices['High'].iloc[day], ha_open.iloc[day], ha_close.iloc[day]]).max()
        ha_low.iloc[day] = np.array([prices['Low'].iloc[day], ha_open.iloc[day], ha_close.iloc[day]]).min()

    # Concatenate them in order to create a panda dataframe
    df = pd.concat((ha_open, ha_high, ha_low, ha_close), axis=1)

    # Rename the columns
    df.columns = ['Open', 'High', 'Low', 'Close']

    df.index = df.index.droplevel(0)

    # Store our dataframe for this period in the periods dictionnary
    dict[periods[0]] = df

    # Store our periods dictionnary in the attribute candles of the class Holder
    results.candles = dict

    # Return the object
    return results


#### Fourier & Sine Series Fits
# An attempt to measure the relative vibration parameters of a price series.
# Including frequency, amplitude, phase angle, etc..

# Detrender
def detrend(prices, method='difference'):
    '''

    First we remove the major trend from the series in order to get a better fit

    Methods of de-trending : Difference De-trending, Linear De-trending
    Difference De-trending consists in subtracting the previous value of the time series from the current value of the time series
    Linear De-trending consists in substracting the value of the linear regression line from the current value of the times series


    :param prices: pandas dataframe of OHLC & Volume data

    :param method: method by which to detrend 'linear or 'difference'

    :return: the detrended price series

    '''

    if method=='difference':
        detrended = prices.Close[1:]-prices.Close[:-1].values

    elif method=='linear':
        x = np.arange(0, len(prices))
        y = prices.Close.values
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape(len(prices))
        detrended = prices.Close - trend

    else:
        print('You did not input a valid methos for detrending! Options are linear or difference')

    return detrended

# Fourier Fit of the De-trending series
# Use Scipy optimization library to best fit the trended series

# Fourier Series Expansion Fitting Functions

def fseries(x, a0, a1, b1, w):
    '''

    :param x: the hours (independant variable)

    :param a0: first fourier series coefficient

    :param a1: second fourier series coefficient

    :param b1: third fourier series coefficient

    :param w: fourier series frequency

    :return: the value of the fourier function

    '''

    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

    return f

# Sine Series Expansion Fitting Functions

def sseries(x, a0, b1, w):
    '''

    :param x: the hours (independant variable)

    :param a0: first sine series coefficient

    :param b1: third sine series coefficient

    :param w: sine series frequency

    :return: the value of the sine function

    '''

    f = a0 + b1 * np.sin(w * x)

    return f

# Fourier Series Coefficient Calculator Function

def fourier(prices, periods, method='difference'):
    '''

    :param prices: pandas dataframe of OHLC & Volume data

    :param periods: list of the hours we want the feature to be applied on, for instance [1, 2, 3, 6]
                    (cf the research paper), period for which to create the feature, 1 = 1 hour


    :param method: method by which to detrend the data

    :return: dict of dataframes containing coefficients for said periods

    '''

    # Create an object
    results = Holder()

    # Create an empty dictionnary
    dictio = {}

    # Option to plot the expansion fit for each iteration
    plot = False

    # Compute the coefficients of the series
    detrended = detrend(prices, method)
    # Iterate on the hours
    for hours in periods:
        # Compute each coeffs for each group of 3 hours
        coeffs = []
        for j in range(hours, len(prices)-hours):
            x = np.arange(0, hours)
            y = detrended.iloc[j-hours:j]

            ## Curve optimization
            # Catch the error
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                try:
                    # res i sour coefficients
                    res = scipy.optimize.curve_fit(fseries, x, y)
                except (RuntimeError, OptimizeWarning):
                    # Number of parameters
                    res = np.empty((1, 4))
                    res[0,:] = np.NAN

            if plot:
                xt = np.linspace(0, hours, 100)
                # Value of our function at each xt
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            # Fill our coefficient with the res variable
            coeffs = np.append(coeffs, res[0], axis=0)

        # Get rid of a warning we don't like
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)//4, 4)))

        # Store it in the pandas dataframe
        df = pd.DataFrame(coeffs, index=prices.iloc[hours:-hours].index)

        # Rename columns
        df.columns = ['a0', 'a1', 'b1', 'w']

        # Set nan value to the closest real numer value above/behind it
        df.fillna(method='bfill')

        # Store this dataframe in the dictionnary
        dictio[hours] = df

    results.coeffs = dictio

    return results


# Sine Series Coefficient Calculator Function

def sine(prices, periods, method='difference'):
    '''

    :param prices: pandas dataframe of OHLC & Volume data

    :param periods: list of the hours we want the feature to be applied on, for instance [1, 2, 3, 6]
                    (cf the research paper), period for which to create the feature, 1 = 1 hour


    :param method: method by which to detrend the data

    :return: dict of dataframes containing coefficients for said periods

    '''

    # Create an object
    results = Holder()

    # Create an empty dictionnary
    dict ={}

    # Option to plot the expansion fit for each iteration
    plot = False

    # Compute the coefficients of the series
    detrended = detrend(prices, method)
    # Iterate on the hours
    for hours in periods:
        # Compute each coeffs for each group of 3 hours
        coeffs = []
        for j in range(hours, len(prices)-hours):
            x = np.arange(0, hours)
            y = detrended.iloc[j-hours:j]

            ## Curve optimization
            # Catch the error
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                try:
                    # res i sour coefficients
                    res = scipy.optimize.curve_fit(sseries, x, y)
                except (RuntimeError, OptimizeWarning):
                    # Number of parameters
                    res = np.empty((1, 3))
                    res[0,:] = np.NAN

            if plot:
                xt = np.linspace(0, hours, 100)
                # Value of our function at each xt
                yt = sseries(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')

                plt.show()

            # Fill our coefficient with the res variable
            coeffs = np.append(coeffs, res[0], axis=0)

        # Get rid of a warning we don't like
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)//3, 3)))

        # Store it in the pandas dataframe
        df = pd.DataFrame(coeffs, index=prices.iloc[hours:-hours].index)

        # Rename columns
        df.columns = ['a0', 'b1', 'w']

        # Set nan value to the closest real numer value above/behind it
        df.fillna(method='bfill')

        # Store this dataframe in the dictionnary
        dict[hours] = df

    results.coeffs = dict

    return results

# Williams Accumulation Distribution

def wadl(prices, periods):
    '''
    A 'supply and demand' indicator that indicates the relative amount of buying
    and selling. Check online for the calculation.

    :param prices: dataframe of OHLC prices
    :param periods: (list) periods for which to calculate the function
    :return: williams accumulation distribution lines for each period
    '''

    # Create an object
    results = Holder()

    # Create an empty dictionnary
    dict ={}

    for hours in periods:
        # Store the wadl for the period
        wad = []
        for j in range(hours, len(prices)-hours):
            # Calculation of the True Range High & Low
            current_close = prices.Close.iloc[j]
            prec_close = prices.Close.iloc[j-1]
            tr_high = np.array([prices.High.iloc[j], prec_close]).max()
            tr_low = np.array([prices.Low.iloc[j], prec_close]).min()

            if current_close > prec_close:
                price_move = current_close - tr_low
            elif current_close < prec_close:
                price_move = current_close - tr_high
            elif current_close == prec_close:
                price_move = 0
            else:
                print('Unknown error occured, see administrator')

            # Calculate Accumulation Distribution
            acc_distrib = price_move * prices.AskVol.iloc[j]

            wad = np.append(wad, acc_distrib)

        # Calculate Total Accumulation Distribution
        wad = wad.cumsum()

        # Store it in the pandas dataframe and in the dictionnary
        wad = pd.DataFrame(wad, index=prices.iloc[hours:-hours].index)
        wad.columns = ['Close']


        dict[hours] = wad

    results.wadl = dict

    return results

# Data Resampling Function

def ohlcresample(df, timeframe, column='ask'):
    '''
    :param df: dataframe containing data that we want to resample
    :param timeframe: timeframe that we want for resampling
    :param column: which column we are resamlping (bid or ask) default='ask'
    :return: resampled OHLC data for the given timeframe
    '''

    grouped = df.groupby('Symbol')

    # Resample data and change the candles to different time period
    if np.any(df.columns == 'Ask'):

        if column == 'ask':
            ask = grouped['Ask'].resample(timeframe).ohlc()
            askvol = grouped['AskVol'].resample(timeframe).count()
            resampled = pd.DataFrame(ask)
            resampled['AskVol'] = askvol

        elif column == 'bid':
            bid = grouped['Bid'].resample(timeframe).ohlc()
            bidvol = grouped['BidVol'].resample(timeframe).count()
            resampled = pd.DataFrame(bid)
            resampled['BidVol'] = bidvol

        else:
           raise ValueError('Column must be a string. Either ask or bid')

    # Resampling data that are already in candles format
    elif np.any(df.columns == 'Close'):
        open = grouped['Open'].resample(timeframe).ohlc()
        close = grouped['Close'].resample(timeframe).ohlc()
        high = grouped['High'].resample(timeframe).ohlc()
        low = grouped['Low'].resample(timeframe).ohlc()
        askvol = grouped['AskVol'].resample(timeframe).count()

        resampled = pd.DataFrame(open)
        resampled.columns = ['Open', 'High', 'Low', 'Close']
        resampled['High'] = high['high']
        resampled['Low'] = low['low']
        resampled['Close'] = close['close']
        resampled['AskVol'] = askvol



    resampled.dropna(inplace=True)

    return resampled

# Momentum Function

def momentum(prices, periods):
    '''
    :param prices: dataframe of OHLC data
    :param periods: list of periods to calculate function value
    :return: momentum indicator
    '''

    results = Holder()

    # Initializing two dictionnaries open and close because we can have both momentum
    # for open and close
    open = {}
    close = {}

    for hours in periods:

        # Momentum measures the rate of acceleration of the price or volume (price here)
        # Difference between 2 values separate by a certain amount of time
        open[hours] = pd.DataFrame(prices.Open.iloc[hours:] - prices.Open.iloc[:-hours].values,
                                   index=prices.iloc[hours:].index)
        close[hours] = pd.DataFrame(prices.Close.iloc[hours:] - prices.Close.iloc[:-hours].values,
                                    index=prices.iloc[hours:].index)

        # Rename columns
        open[hours].columns = ['Open']
        close[hours].columns = ['Close']

    # Store data in the class
    results.open = open
    results.close = close

    return results

# Stochastic Oscillator Function

def stochastic(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: list of periods to calculate function value
    :return: Oscillator function values
    '''

    # Initialize class and dictionnary as usual
    results = Holder()
    close = {}

    for hours in periods:
        # See this website for calculation
        # https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:stochastic_oscillator_fast_slow_and_full
        Ks = []

        # Avoid the noise of the data
        for j in range(hours, len(prices)-hours):
            C = prices.Close.iloc[j+1]
            H = prices.High.iloc[j-hours:j].max()
            L = prices.Low.iloc[j-hours:j].min()

            if H == L:
                K = 0
            else:
                K = 100*(C-L)/(H-L)
            Ks = np.append(Ks, K)

        df = pd.DataFrame(Ks, index=prices.iloc[hours+1:-hours+1].index)
        df.columns = ['K']
        df['D'] = df.K.rolling(3).mean()
        df.dropna(inplace=True)

        close[hours] = df

    results.stochclose = close

    return results

# Williams Oscillator Function

def williams(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: list of periods to calculate function value
    :return: Williams oscillator function values
    '''
    # Initialize class and dictionnary as usual
    results = Holder()
    close = {}

    for hours in periods:
        # See this website for calculation
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
        Rs = []

        # Avoid the noise of the data
        for j in range(hours, len(prices)-hours):
            C = prices.Close.iloc[j+1]
            H = prices.High.iloc[j-hours:j].max()
            L = prices.Low.iloc[j-hours:j].min()

            if H == L:
                R = 0
            else:
                R = -100*(H-C)/(H-L)
            Rs = np.append(Rs, R)

        df = pd.DataFrame(Rs, index=prices.iloc[hours+1:-hours+1].index)
        df.columns = ['R']
        # df['D'] = df.R.rolling(3).mean()
        df.dropna(inplace=True)

        close[hours] = df

    results.willclose = close

    return results


# PROC Function (Price Rate of Change)

def proc(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: list of periods to calculate function value
    :return: PROC for indicated periods
    '''

    results = Holder()
    proc = {}

    for hours in periods:
        proc[hours] = pd.DataFrame((prices.Close.iloc[hours:]-prices.Close.iloc[:-hours].values)\
                                    /prices.Close.iloc[:-hours].values)
        proc[hours].columns = ['Close']

    results.proc = proc

    return results

# Accumulation Distribution Oscillator

def adosc(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: list of periods to calculate function value
    :return: indicator values for indicated periods
    '''

    results = Holder()
    accdist = {}

    for hours in periods:
        AD = []

        # Avoid the noise of the data
        for j in range(hours, len(prices)-hours):
            C = prices.Close.iloc[j+1]
            H = prices.High.iloc[j-hours:j].max()
            L = prices.Low.iloc[j-hours:j].min()
            V = prices.AskVol.iloc[j+1]

            if H == L:
                CLV = 0
            else:
                CLV = ((C-L)-(H-C))/(H-L)
            AD = np.append(AD, CLV*V)

        AD = AD.cumsum()
        df = pd.DataFrame(AD, index=prices.iloc[hours+1:-hours+1].index)
        df.columns = ['AD']
        # df['D'] = df.R.rolling(3).mean()
        # df.dropna(inplace=True)

        accdist[hours] = df

    results.ad = accdist

    return results

# MACD (Moving Average Convergence Divergence)

def macd(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: 1x2 array containing values for the EMAs
    :return: MACD for indicated periods
    '''

    # Initialize class and dictionnary as usual
    results = Holder()

    # Exponential Moving Averages
    ema1 = prices.Close.ewm(span=periods[0]).mean()
    ema2 = prices.Close.ewm(span=periods[1]).mean()

    macd = pd.DataFrame(ema1-ema2)
    macd.columns = ['L']

    # Return also signal : 3 period moving average of the MACD
    sig_macd = macd.rolling(3).mean()
    sig_macd.columns = ['SL']

    results.line = macd
    results.signal = sig_macd

    return results

# CCI (Commodity Channel Index)

def cci(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: periods for which to compute the indicator
    :return: CCI for indicated periods
    '''
    results = Holder()
    cci = {}

    for hours in periods:
        # Moving Average 
        ma = prices.Close.rolling(hours).mean()
        std = prices.Close.rolling(hours).std()

        # Mean Deviation
        md = (prices.Close-ma)/std

        cci[hours] = pd.DataFrame((prices.Close-ma)/(0.015*md))
        cci[hours].columns = ['Close']

    results.cci = cci

    return results

# Bollinger Bands

def bollinger(prices, periods, deviation):
    '''
    :params prices: dataframe of OHLC data
    :param periods: periods for which to compute the indicator
    :param deviation: deviations to use when calculating bands (upper&lower)
    :return: bollinger bands
    '''

    results = Holder()
    boll = {}

    for hours in periods:

        mid = prices.Close.rolling(hours).mean()
        std = prices.Close.rolling(hours).std()

        upper = mid + deviation * std
        lower = mid - deviation * std

        df = pd.concat((upper, mid, lower), axis=1)
        df.columns = ['Upper', 'Mid', 'Lower']

        boll[hours] = df

    results.bands = boll

    return results

# Price Averages

def paverage(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: periods for which to compute the indicator
    :return: averages of the given periods
    '''

    results = Holder()
    avs = {}

    for hours in periods:
        avs[hours] = pd.DataFrame(prices[['Open', 'High', 'Low', 'Close']].rolling(hours).mean())

    results.avs = avs

    return results

# Slope Functions

def slopes(prices, periods):
    '''
    :params prices: dataframe of OHLC data
    :param periods: periods for which to compute the indicator
    :return: slope over given periods
    '''

    results = Holder()
    slope = {}

    # Apply linear regression to some certain data points and store the slope

    for hours in periods:

        s = []

        for j in range(hours, len(prices)-hours):
            # We choose the feature High as indicated in the research paper
            y = prices.High.iloc[j-hours:j].values
            x = np.arange(len(y))
            res = stats.linregress(x, y=y)
            a = res.slope
            s = np.append(s, a)

        s = pd.DataFrame(s, index=prices.iloc[hours:-hours].index)
        s.columns = ['High']

        slope[hours] = s

    results.slope = slope

    return results
