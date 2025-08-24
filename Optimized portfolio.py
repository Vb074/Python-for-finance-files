#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:25:23 2025

@author: vadimbodnarenko
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

asml = pd.read_csv('/Users/vadimbodnarenko/Downloads/Download Data - STOCK_US_XNAS_ASML-4.csv', index_col='Date', parse_dates=True)
meta = pd.read_csv('/Users/vadimbodnarenko/Downloads/Download Data - STOCK_US_XNAS_META.csv', index_col='Date', parse_dates=True)
apple = pd.read_csv('/Users/vadimbodnarenko/Downloads/Download Data - STOCK_US_XNAS_AAPL.csv', index_col='Date', parse_dates=True)
meta = meta.sort_index()
asml = asml.sort_index()
apple = apple.sort_index()

common_index = meta.index.intersection(asml.index)
common_index2 = meta.index.intersection(apple.index)

meta = meta.loc[common_index]
asml = asml.loc[common_index]
apple = apple.loc[common_index]


df_all = pd.concat([apple['Close'], meta['Close'], asml['Close']], axis = 1)

df_all.columns = ('AAPL', 'META', 'ASML')

names = ['AAPL', 'META', 'ASML']

returns = pd.DataFrame(index=df_all.index[1:])


for name in names:
    current_returns = df_all[name].pct_change()
    returns[name] = current_returns.iloc[1:] * 100
    
mean_return = returns.mean()
cov = returns.cov()
cov_np = cov.to_numpy()

#Monte Carlo Simulation for 10000 portfolios with all 3 assets

N = 10000
D = len(mean_return)
returns = np.zeros(N)
risks = np.zeros(N)

for i in range(N):
    rand_range = 1.0
    
    w = np.random.random(D)*rand_range - rand_range/2
    w[-1] = 1 - w[:-1].sum()
    np.random.shuffle(w)
    
    ret = mean_return.dot(w)
    risk = np.sqrt(w.dot(cov_np).dot(w))
    
    returns[i] = ret
    risks[i] = risk
    
#Monte Carlo for each asset individually 

single_asset_return = np.zeros(D)
single_assets_risk = np.zeros(D)

for i in range(D):
    ret = mean_return[i]
    risk = np.sqrt(cov_np[i,i])
    
    single_asset_return[i] = ret
    single_assets_risk[i] = risk
    

#plt.scatter(risks, returns, alpha = 0.1)
#plt.scatter(single_assets_risk, single_asset_return, color = 'red')

#Optimizing for max return (- sign as linprog minimizes by default)
from scipy.optimize import linprog

D = len(mean_return)

A_eq = np.ones((1,D))
b_eq = np.ones(1)

bounds = [(-0.5, None)] * D

res = linprog(-mean_return, A_eq=A_eq, b_eq=b_eq, bounds = bounds)

#Maximum return:
#print(res)
#print('Maximum Return in %: ', -res.fun * 100)
max_return = -res.fun

res = linprog(mean_return, A_eq=A_eq, b_eq=b_eq, bounds = bounds)
#Min return 
min_return = res.fun

#--------------------------------------------------------------------------

#Computing and plotting an efficient frontier

N = 100
target_returns = np.linspace(min_return, max_return, num = N)
from scipy.optimize import minimize

def get_portfolio_variance(weights):
    return weights.dot(cov.dot(weights))

def target_return_constraint(weights, target):
    return weights.dot(mean_return) - target

def portfolio_constraint(weights):
    return weights.sum() - 1

#Defining a constraint dictionary

constraints = [
    {
     'type': 'eq',
     'fun': target_return_constraint,
     'args': [target_returns[0]],
     },
    {'type': 'eq',
     'fun': portfolio_constraint,
     }
    ]

#Minimizing, limiting the magnitude of weights

res = minimize(
    fun = get_portfolio_variance,
    x0 = np.ones(D)/D,
    method = 'SLSQP',
    constraints = constraints,
    bounds = bounds,
    )

#Running through all the target returns, to generate an efficient frontier

optimized_risks = []
for target in target_returns:
    constraints[0]['args'] = [target] #Each iteration enforces a different required return
    #Pasting a minimizing fucntion from before
    res = minimize(
        fun = get_portfolio_variance,
        x0 = np.ones(D)/D,
        method = 'SLSQP',
        constraints = constraints,
        bounds = bounds,
        )
    optimized_risks.append(np.sqrt(res.fun))
    if res.status != 0:
        print(res)
        
plt.scatter(risks, returns, alpha = 0.1)
plt.plot(optimized_risks, target_returns, c = 'black') #<------ Efficient frontier

#Computing the Sharpe Ratio with risk-free rate being 3.9%
risk_free_rate = 0.039/252

#for maximization 
def neg_sharpe_ratio(weights): 
    mean = weights.dot(mean_return)
    sd = np.sqrt(weights.dot(cov).dot(weights))
    return -(mean-risk_free_rate)/sd

#Minimize function (to optimize SR, since it is negative)
res = minimize(
    fun = neg_sharpe_ratio,
    x0 = np.ones(D)/D,
    method = 'SLSQP',
    constraints = {
        'type': 'eq',
        'fun': portfolio_constraint,
        },
    bounds = bounds,
    )

#Assigning the best SR and corresponding weights W to variables:

best_sr, best_w = -res.fun, res.x

fig,ax = plt.subplots(figsize=(10,5))
plt.scatter(risks, returns, alpha = 0.1)
plt.plot(optimized_risks, target_returns, c='black')

opt_risk = np.sqrt(best_w.dot(cov).dot(best_w))
opt_ret = mean_return.dot(best_w)
plt.scatter([opt_risk], [opt_ret], c = 'r')

x1 = 0
y1 = risk_free_rate
x2 = opt_risk
y2 = opt_ret

plt.plot([x1,x2], [y1,y2])








    








    
    

    
    


    



                    
                    












