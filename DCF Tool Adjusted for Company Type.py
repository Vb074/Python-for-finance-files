#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:24:38 2025

@author: vadimbodnarenko
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def choose_scenario(hist_fcf, comp):
    yoy_growth = []
    weights = []
    
    for i in range(len(hist_fcf) - 1):
        prev = hist_fcf[i]
        curr = hist_fcf[i+1]
        
        if prev <= 0:
            continue
        
        growth = (curr - prev)/prev
        
        yoy_growth.append(growth)
        weights.append(i+1)
    if comp == 1:
        weighted_avg = np.average(yoy_growth, weights = weights)
        weighted_avg = np.clip(weighted_avg, -0.20, 0.40)
    elif comp == 2:
        weighted_avg = np.average(yoy_growth, weights = weights)
        weighted_avg = np.clip(weighted_avg, -0.10, 0.12)
    else:
        weighted_avg = np.average(yoy_growth, weights = weights)
        weighted_avg = np.clip(weighted_avg, -0.5, 0.5)
    
    
    proj_growth_best = weighted_avg * 1.1
    proj_growth_wa = weighted_avg
    proj_growth_norm = weighted_avg * 0.8
    proj_growth_worst = weighted_avg * 0.5

    print("Choose a projection scenario:")
    print("1 - Best case   (weighted_avg * 1.1)")
    print("2 - Base case   (weighted_avg)")
    print("3 - Normalized  (weighted_avg * 0.8)")
    print("4 - Worst case  (weighted_avg * 0.5)")
   
    option = input("Enter option (1–4): ")
   
    if option == "1":
        return proj_growth_best        
    elif option == "2":
        return proj_growth_wa        
    elif option == "3":
        return proj_growth_norm        
    elif option == "4":
        return proj_growth_worst       
    else:
        print("Invalid selection. Using weighted_avg")
        return proj_growth_wa    
    
    

def CAPM(beta, rf, rm, price, S_outs, total_debt, interest_expense, tax_rate):
    re = rf + beta * (rm - rf)            
    rd = interest_expense / total_debt
    E = price * S_outs
    D = total_debt
    V = E + D
    w_e = E / V
    w_d = D / V
    WACC = w_e * re + w_d * rd * (1 - tax_rate)
    return WACC
 


def DCF(last_year_fcf, S_outs, debt, net_debt, proj, fcf_growth, r, g, comp):
    fcf_list_disc = []
    
    for n in range(1, proj+1):
        if comp == 1:
            if n > 5:
                fcf_growth *= 0.85
                fcf = last_year_fcf * (1 + fcf_growth)  
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
            else:
                fcf = last_year_fcf * (1 + fcf_growth) 
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
                
        elif comp == 2:
            if n > 4:
                fcf_growth *= 0.95
                fcf = last_year_fcf * (1 + fcf_growth)
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
            else:
                fcf = last_year_fcf * (1 + fcf_growth)
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
                
        else:
            if n > 2:
                fcf_growth *= 0.98
                fcf = last_year_fcf * (1 + fcf_growth) 
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
            else:
                fcf = last_year_fcf * (1 + fcf_growth) 
                disc = 1 / ((1 + r) ** n)
                fcf_list_disc.append(fcf * disc)
                last_year_fcf = fcf
                
        
    total_pv_fcf = np.sum(fcf_list_disc)

    TV = (last_year_fcf * (1 + g)) / (r - g)    
    PV_TV = TV / ((1 + r)**proj)

    Enterprise_v = total_pv_fcf + PV_TV
    Equity_v = Enterprise_v - net_debt
    Fair_Value = Equity_v / S_outs

    return Fair_Value
           

def mc_dcf(n, last_year_fcf, S_outs, debt, net_debt,
                    proj, fcf_growth, r, g, comp):
    results = []
    
    for _ in range(n):
        growth_sim = np.random.normal(fcf_growth, 0.02)
        r_sim = np.random.normal(r, 0.01)
        g_sim = np.random.uniform(g - 0.005, g + 0.005)
        g_sim = min(g_sim, r_sim - 0.01)
        
        fair_v = DCF(last_year_fcf=last_year_fcf, S_outs = S_outs, debt = debt, 
                     net_debt = net_debt, proj = proj, 
                     fcf_growth = growth_sim, r = r_sim, g = g_sim, comp = comp)
        
        results.append(fair_v)
    return results
        





def main():
    print("-----DCF TOOL-----")
    
    comp = int(input('Enter 1 for High-Growth/Early-Stage Companies, 2 for Stable/Mid-Stage/Tech Companies, or 3 for Large/Stable Companies: '))

    S_outs = float(input("Enter shares outstanding: "))
    last_year_fcf = float(input("Enter last year's FCF (most recent full year): "))
    debt = float(input("Enter total debt: "))
    net_debt = float(input("Enter net debt: "))
    

    print("\nEnter historical FCFs (comma separated):")
    hist_fcf_str = input("Historical FCFs (skipping most recent full year): ")
    hist_fcf = [float(x.strip()) for x in hist_fcf_str.split(",")]
    

    proj = int(input("Projection years: "))

    fcf_growth = choose_scenario(hist_fcf, comp)   
    
    print('-----WACC Inputs-----')
    beta = float(input("Beta (use industry avg. if the company is very stable (Mag 7)): "))
    rf = float(input("Risk-free rate - use 10y US Treasury Yield (decimal): "))    
    rm = float(input("Market return (0.1 for S&P500) (decimal): "))      
    interest_expense = float(input("Interest expense: "))
    tax_rate = float(input("Tax rate (decimal): "))     
    price = float(input("Current share price: "))

    r = CAPM(beta, rf, rm, price, S_outs, debt, interest_expense, tax_rate) 

    g = float(input("Terminal growth (0.025 - US avg.) (decimal): "))     

    fair = DCF(
        last_year_fcf=last_year_fcf,
        S_outs=S_outs,
        debt=debt,
        net_debt=net_debt,
        proj=proj,
        fcf_growth=fcf_growth,
        r=r,                     
        g=min(g, r-0.005),
        comp = comp                     
    )

    print(f"\nFair Value per Share: {fair:.2f}")
    
    run_mc = input('Run MC? y/n: ')
    
    if run_mc.lower() == 'y':
        sims = mc_dcf(n =  100000, last_year_fcf = last_year_fcf, 
                      S_outs = S_outs, debt = debt, 
                      net_debt = net_debt, proj = proj, 
                      fcf_growth = fcf_growth, r = r, g = g, comp = comp)
        
        ci_l = np.percentile(sims, 10)
        ci_h = np.percentile(sims, 90)
    
        print(f"\n90% Confidence Interval for the Fair Price is: ({ci_l}, {ci_h})")
    
    return fair

ASML = main()


