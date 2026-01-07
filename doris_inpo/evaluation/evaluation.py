import numpy as np

def rms(series):
    return np.sqrt((series ** 2).mean())

def std(series, ddof=0):
    return series.std(ddof=ddof)