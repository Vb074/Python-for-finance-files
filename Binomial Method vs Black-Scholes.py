#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:49:29 2026

@author: vadimbodnarenko
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


#Binomial Model Valuation:
def american_option_bin(S, K, N, T, vol, r):
    dt = T/N
    u = np.exp(vol*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u-d)
    disc = np.exp(-r*dt)
    
    #Stock prices at maturity
    prices = [S * (u ** j) * (d ** (N-j)) for j in range(N+1)]
    
    #Option values at maturity
    values_call_a = [np.maximum(price - K, 0) for price in prices]
    values_put_a = [np.maximum(K - price, 0) for price in prices]
    values_call_eu = [np.maximum(price - K, 0) for price in prices]
    values_put_eu = [np.maximum(K - price, 0) for price in prices]
        
    
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            hold_call_a = disc * (p * values_call_a[j+1] + (1-p)*values_call_a[j])
            hold_put_a = disc * (p * values_put_a[j+1] + (1-p)*values_put_a[j])
            hold_call_eu = disc * (p * values_call_eu[j+1] + (1-p)*values_call_eu[j])
            hold_put_eu = disc * (p * values_put_eu[j+1] + (1-p)*values_put_eu[j])
            stock = S * (u ** j) * (d ** (i - j))
            
            ex_call = np.maximum(stock - K, 0)
            ex_put = np.maximum(K - stock, 0)
            
            
                
            values_call_a[j] = np.maximum(hold_call_a, ex_call)
            values_put_a[j] = np.maximum(hold_put_a, ex_put)
            values_call_eu[j] = hold_call_eu
            values_put_eu[j] = hold_put_eu
            
            
            
    return values_call_a[0], values_put_a[0], values_call_eu[0], values_put_eu[0]


V0 = american_option_bin(100, 100, 1500, 1/6, 0.25, 0.03)
print('American Call value using Binomial method valuation: ', (V0[0]))
print('American Put value using Binomial method valuation: ', (V0[1]))
print('EU Call value using Binomial method valuation: ', V0[2])
print('EU Put value using Binomial method valuation: ', V0[3])



#Black-Scholes Valuation:
r = 0.03
vol = 0.25
T = 1/6
K = 100
S0 = 100

def BS(S0, K, T, vol, r):

    d1 = (np.log(S0/K) + (r + 0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    call = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    put  = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    
    return call, put

call, put = BS(S0, K, T, vol, r)

print('EU Call value using Black-Scholes valuation: ', call)
print('EU Put value using Black-Scholes valuation: ', put)

print('---------------------------------------------------------------------------------')

print('Difference between Binomial and Black-Scholes for EU Call: ', abs(V0[2] - call))
print('Difference between Binomial and Black-Scholes for EU Put: ', abs(V0[3] - put))

error = []
Ns = range(100, 1500, 50)
for N in Ns:
    V0 = american_option_bin(100, 100, N, 1/6, 0.25, 0.03)
    error.append(abs(V0[2] - call))
    
plt.figure()
plt.plot(Ns, error)
plt.xlabel('Binomial Steps N')
plt.ylabel('Error')
plt.title('Convergence of Binomial European Call to Black-Scholes)')
plt.show()


    












        
        
    
    
    
    
    
    
    



            
            
    