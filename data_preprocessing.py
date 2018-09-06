#!/usr/bin/env python3
'''
Labeling the data (Buy Sell Hold)
Normalizing the data
Feature Selection
'''

from fastai.imports import *
from fastai.structured import proc_df, add_datepart
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



def create_label(df, stay_param=0):
    '''
    :param df: dataframe imported with all features alrready created
    :param stay_param: parameter that indicate when to Buy or Sell or Hold
    :return : nothing
    '''

    df['Price'] = df['Close'] - df['Open']
    def g(x):
        '''
        mapping function called right after
        :x : real value
        :return :trader action : 1 if Buy, - 1 if Sell, 0 if Hold
        '''
        if x > stay_param:
            return 1 # Buy
        elif x <= -stay_param:
            return -1 # Sell
        return 0 # Hold
    df['Actions'] = df['Price'].map(g)

    # Drop the Price feature
    df.drop(['Price'], axis=1, inplace=True)


def divide_dataframe(df):
    '''
    :param df: dataset we want to split into training and test data
    :return : X_train, X_test, y_train, y_test
    '''
    # Drop the date column
    # df.drop('Date', axis=1, inplace=True)

    # Converting date into int with fastai
    add_datepart(df, 'Date')

    # Preprocess with fastai
    X, y, nas, mapper = proc_df(df, y_fld='Actions', do_scale=True)
    # print(X.transpose()) # Debugging
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.4, shuffle=False)
    return {'X train': X_train, 'X test': X_test, 'y train':y_train, 'y test': y_test}


def calculate_strategy_returns(df, clf, X_test, y_test):
    '''
    We will predict the signal (buy or sell) for the test data set.
    Then, we will compute the strategy returns based on the signal predicted by the model in the test dataset.
    We save it in the column ‘Strategy_Return’ and then, plot the cumulative strategy returns.

    :param df: dataset for calculate the returns of the strategy
    :param clf: our classifier for predictions
    :param X_test: features we want to classify
    :param y_test: labels needed for computing the accuracy
    '''
    test_df = pd.DataFrame()
    test_df['Predicted_Signal'] = clf.predict(X_test)
    split = test_df.Predicted_Signal.count()
    test_df['Close'] = df.Close.loc[-split:].copy()
    # Calculate returns
    df['Price'] = ((df['Close'] - df['Open'])/df['Close'])*100
    test_df['Return'] = df.Price.loc[-split:].copy()
    test_df['Strategy_Return'] = test_df.Return * test_df.Predicted_Signal
    # print(test_df.head(30))
    test_df.Strategy_Return.cumsum().plot(figsize=(10,5))
    plt.ylabel('Strategy Returns (%)')
    plt.show()
    return test_df.Predicted_Signal

def main():
    '''
    Main function calling the functions written
    '''
    # Import the dataframe created
    df = pd.read_csv('data/dataframe.csv')

    # Label data : create an Buy (1) - Sell (-1) - Hold (0)
    create_label(df)

    # Divide dataframe into training and testing samples
    samples = divide_dataframe(df)

    # Use SVM
    clf = svm.SVC(kernel='rbf')

    # Train our model
    clf.fit(samples['X train'], samples['y train'])

    # Make prediction
    # y_pred = clf.predict(samples['X test'])
    y_pred = calculate_strategy_returns(df, clf, samples['X test'], samples['y test'])

    # Check accuracy
    print(accuracy_score(samples['y test'], y_pred))

main()
