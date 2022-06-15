import numpy as np
import cupy as cp
from cupyx.scipy.special import erfinv


def rmse(predictions, data):
    return ((predictions - data) ** 2).mean() ** 0.5

def apply_gauss_rank(dataset, col, epsilon = 1e-6):
    data = dataset[col].values
    r_gpu = data.argsort().argsort()
    r_gpu = (r_gpu/r_gpu.max()-0.5)*2 # scale to (-1,1)
    r_gpu = cp.clip(a=r_gpu,a_min=(-1+epsilon),a_max=(1-epsilon))
    r_gpu = erfinv(r_gpu)
    dataset[col] = r_gpu

def log_norm(dataset, col):
    tmp = cp.log(dataset[col])
    mean = cp.mean(tmp)
    std = cp.std(tmp)
    dataset[col] = (cp.log(dataset[col]) - mean) / std
