import xarray as xr
import numpy as np
import pandas as pd

def precip_sum(data, period):
    '''Computes precipitation totals at various timescales.'''
    ds_f = data.where(data >= 0)
    ds_sum = ds_f.resample(time=period, label='right').sum()
    return ds_sum