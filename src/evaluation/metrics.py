import numpy as np
import pandas as pd

def sharpe(r, rf=0):
    r=pd.Series(r).dropna()
    return np.nan if r.std(ddof=1)==0 else np.sqrt(365)*(r.mean()-rf)/r.std(ddof=1)

def sortino(r, rf=0):
    r=pd.Series(r).dropna()
    d=np.minimum(r-rf,0)
    dd=np.sqrt(np.mean(d**2))
    return np.nan if dd==0 else np.sqrt(365)*(r.mean()-rf)/dd

def max_drawdown(r):
    e=(1+pd.Series(r).fillna(0)).cumprod()
    return (e/e.cummax()-1).min()

def turnover(old,new):
    return np.abs(np.asarray(new)-np.asarray(old)).sum()

def net_returns(gross, turnover_series, cost_rate=0.001):
    return pd.Series(gross).reset_index(drop=True)-pd.Series(turnover_series).reset_index(drop=True)*cost_rate
