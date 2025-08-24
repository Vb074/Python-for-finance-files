#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 19:24:25 2025

@author: vadimbodnarenko
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

url = 'https://lazyprogrammer.me/course_files/airline_passengers.csv'
df = pd.read_csv(url, index_col = 'Month', parse_dates = True)


Ntest = 12
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]

arima = ARIMA(train['Passengers'], order = (12,1,0))
arima_result_1210 = arima.fit()

#Creating a differencting function to shift the series back by d as d in this case is >1

def plot_fit_and_forecast_int(result, d, col = 'Passengers'):
    fig,ax = plt.subplots(figsize  = (10,5))
    ax.plot(df[col], label = 'data')
    
    #Plotting the curve fitted on training set
    train_pred = result.predict(start = train.index[d], end = train.index[-1], typ = 'levels')
    ax.plot(train.index[d:], train_pred, color = 'g', label = 'fitted')
    
    #Forecast the test set
    forecast_res = result.get_forecast(steps=Ntest)
    forecast = forecast_res.predicted_mean
    confint = forecast_res.conf_int()
    ax.plot(test.index, forecast, label = 'forecast')
    
    ax.fill_between(test.index, confint.iloc[:,0], confint.iloc[:,1], color = 'r', alpha = 0.3)
    ax.legend()
    
    
    
plot_fit_and_forecast_int(arima_result_1210, 1)


    