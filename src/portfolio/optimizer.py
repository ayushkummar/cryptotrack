import numpy as np
from scipy.optimize import minimize

def minimum_variance(cov):
    n = len(cov)
    x0 = np.ones(n)/n
    fun = lambda w: float(w @ cov @ w)
    res = minimize(fun,x0,method="SLSQP",
                   bounds=[(0,1)]*n,
                   constraints={"type":"eq","fun":lambda w:w.sum()-1})
    if not res.success: raise RuntimeError(res.message)
    return res.x

def probability_aware_weights(cov, probabilities,
                               risk_aversion=10.0,max_weight=0.60):
    p=np.asarray(probabilities,float); cov=np.asarray(cov,float); n=len(p)
    fun=lambda w: -(w@p-risk_aversion*(w@cov@w))
    res=minimize(fun,np.ones(n)/n,method="SLSQP",
                 bounds=[(0,max_weight)]*n,
                 constraints={"type":"eq","fun":lambda w:w.sum()-1})
    if not res.success: raise RuntimeError(res.message)
    return res.x
