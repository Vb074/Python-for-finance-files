#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 12:29:26 2026

@author: vadimbodnarenko
"""

#In my OVM course we had to manually set up the grid, BCs and Matrices for various option valuations. 
#Obviously you can't actually solve that on paper, hence wanting an actual numercial result 
#I have written a code that uses Crank-Nicolson method to value a Down-and-out barrier option. 
# Payoff is such that if a price of an underlying asset drops and touches the barrier price B,
#The option's value is automatically 0. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def down_out_call_CN_scipy(S0=10, E=10, r=0.01, q=0.005, sigma=0.3, T=1.0, B=9.0, Smax=40.0, M=300, N=600):
    #Numerical Grid
    S = np.linspace(B, Smax, M+1)
    dS = S[1] - S[0]
    dt = T/N
    
    V = np.maximum(S-E, 0)
    j = np.arange(1, M)
    Sj = S[j]
    
    #Forward Difference Operators:
    aj = 0.5 * sigma**2 * Sj**2 / dS**2 - (r - q) * Sj / (2 * dS)
    bj = -sigma**2 * Sj**2 / dS**2 - r
    cj = 0.5 * sigma**2 * Sj**2 / dS**2 + (r - q) * Sj / (2 * dS)
    
    #Crank Nicolson A V^n = B V^{n+1}
    A_sub  = -(dt / 2) * aj[1:]     
    A_diag = 1 - (dt / 2) * bj      
    A_sup  = -(dt / 2) * cj[:-1]
    
    B_sub  = +(dt / 2) * aj[1:]
    B_diag = 1 + (dt / 2) * bj
    B_sup  = +(dt / 2) * cj[:-1]
    
    #Setting up the solver
    ab = np.zeros((3, M - 1))
    ab[0, 1:]  = A_sup 
    ab[1, :]   = A_diag        
    ab[2, :-1] = A_sub
    
    #Marching back in time
    for n in range(N - 1, -1, -1):
        tau = T - n * dt
        
        #BCs
        V_left = 0.0  
        V_right = Smax * np.exp(-q * tau) - E * np.exp(-r * tau)
        
        V[0] = V_left
        V[-1] = V_right
        
        Vin = V[1:M]
        
        # RHS = B * Vin + BCs 
        rhs = B_diag * Vin
        rhs[1:]  += B_sub * Vin[:-1]
        rhs[:-1] += B_sup * Vin[1:]
        
        rhs[0]  += (dt / 2) * aj[0]  * V_left
        rhs[-1] += (dt / 2) * cj[-1] * V_right

    
        V[1:M] = solve_banded((1, 1), ab, rhs)

    price = np.interp(S0, S, V)
    return price, S, V

price, S, V = down_out_call_CN_scipy()
print("Down-and-out call price:", price)

plt.plot(S, V)
plt.xlabel("S")
plt.ylabel("V(S,0)")
plt.title("Down-and-out call via Crank–Nicolson")
plt.show()
    



        