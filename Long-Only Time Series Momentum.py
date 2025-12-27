#This is a sample Python script.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

df = yf.download('ASML', start = '2018-01-01', end = '2025-12-26', progress = False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.sort_index(inplace=True)


df['LogReturn'] = np.log(df['Close']).diff()
df['LogReturn'] = df['LogReturn'].shift(-1)

train_df = df.iloc[:-500].copy()

returns_list = []
w_list = []

for w in range(20, 151):
    train_df['MA'] = train_df['Close'].rolling(w).mean()
    train_df['MASlope'] = train_df['MA'] - train_df['MA'].shift(10)

    train_df['PosTrend'] = (train_df['MASlope'] > 0).fillna(False).astype(bool)
    train_df['PrevTrend'] = train_df['PosTrend'].shift(1, fill_value=False)

    train_df['Buy'] = (train_df['PrevTrend'] == False) & (train_df['PosTrend'] == True)
    train_df['Sell'] = (train_df['PrevTrend'] == True) & (train_df['PosTrend'] == False)

    is_invested = [False]

    def check_is_inv(row):
        if is_invested[0] and row['Sell']:
            is_invested[0] = False
        if (not is_invested[0]) and row['Buy']:
            is_invested[0] = True
        return is_invested[0]


    train_df['IsInvested'] = train_df.apply(check_is_inv, axis=1)
    train_df['AlgoReturn'] = train_df['IsInvested'] * train_df['LogReturn']

    returns_list.append(train_df['AlgoReturn'].fillna(0).sum())
    w_list.append(w)

is_invested = False

def trend_cont(df, w):
    global is_invested
    df = df.copy()

    df['MA'] = df['Close'].rolling(w).mean()
    df['MASlope'] = df['MA'] - df['MA'].shift(10)

    df['PosTrend'] = (df['MASlope'] > 0).fillna(False).astype(bool)
    df['PrevTrend'] = df['PosTrend'].shift(1, fill_value=False)

    df['Buy']  = (df['PrevTrend'] == False) & (df['PosTrend'] == True)
    df['Sell'] = (df['PrevTrend'] == True)  & (df['PosTrend'] == False)

    train = df.iloc[:-500].copy()
    test  = df.iloc[-500:].copy()

    def assign_is_invested(row):
        global is_invested
        if is_invested and row['Sell']:
            is_invested = False
        if (not is_invested) and row['Buy']:
            is_invested = True
        return is_invested

    is_invested = False
    train['IsInvested'] = train.apply(assign_is_invested, axis=1)
    train['AlgoReturn'] = train['IsInvested'] * train['LogReturn']

    is_invested = False
    test['IsInvested'] = test.apply(assign_is_invested, axis=1)
    test['AlgoReturn'] = test['IsInvested'] * test['LogReturn']

    return train['AlgoReturn'].fillna(0).sum(), test['AlgoReturn'].fillna(0).sum()

best_w = w_list[np.argmax(returns_list)]

print('Max Return: ', max(returns_list),
      'Buy&Hold Return (last 500 trading days): ',
      df['LogReturn'].iloc[-500:].sum(),
      'Best MA: ', best_w)


train_ret, test_ret = trend_cont(df, best_w)
print('Train Return: ', train_ret, 'Test Return: ', test_ret)





















