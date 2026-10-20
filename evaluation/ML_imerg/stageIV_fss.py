import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import matplotlib.colors as mcolors

def precip_sum12(ds_f, period):
    '''Computes precipitation totals at various timescales with a 12Z to 12Z accumulation period.'''
    ds_f = ds_f.assign_coords(time=ds_f.time - np.timedelta64(12, 'h'))  # Shift time backward by 12 hours
    ds_sum = ds_f.resample(time=period, label='right').sum()
    return ds_sum

def get_FSS_wtime(data: np.ndarray, truth: np.ndarray, q: float, n: int) -> np.ndarray:
    """
    Compute Fraction Skill Score (FSS) at each time step for a single threshold.

    Parameters
    ----------
    data : model output (time, lat, lon)
    truth : verification data (time, lat, lon)
    q : precipitation threshold (mm)
    n : box size for fraction computation (use 5 for ~1-deg data)

    Returns
    -------
    FSS : array of length Nt
    """
    Nt = len(data)
    FSS = np.empty(Nt)
    frac_above = np.empty(Nt)
    kernel = np.ones((n, n)) / n ** 2

    for t in range(Nt):
        fc_bin = (data[t]  >= q).astype(float)
        gt_bin = (truth[t] >= q).astype(float)

        M   = convolve2d(fc_bin, kernel, mode="valid")
        O   = convolve2d(gt_bin, kernel, mode="valid")
        MSE     = np.mean((O - M) ** 2)
        MSE_ref = np.mean(O ** 2) + np.mean(M ** 2)

        FSS[t] = 1 if MSE_ref == 0 else 1 - MSE / MSE_ref
        frac_above[t] = gt_bin.mean()
        
    return FSS, frac_above

def plot_fss_w_time(mli_fss, img_fss, era_fss):
    
    mfss = pd.Series((mli_fss)).rolling(window=30, center=True, min_periods=1).mean(skipna=True)
    efss = pd.Series((era_fss)).rolling(window=30, center=True, min_periods=1).mean(skipna=True)
    ifss = pd.Series((img_fss)).rolling(window=30, center=True, min_periods=1).mean(skipna=True)

    plt.figure(figsize=(10,3), dpi=200)
    plt.plot(dia_m.time, mfss, color='cornflowerblue', label=r"ML$_{\rm IMERG}$")
    plt.plot(era_m.time, efss, color='crimson', label='ERA5')
    plt.plot(img_m.time, ifss, color='grey', label='IMERG')
    plt.ylabel('FSS q=10mm')
    plt.grid()
    plt.legend()
    plt.savefig('fig15a.png')
    plt.close()
    
def plot_stage4_precip(fa):
    
    fa_smooth = pd.Series((fa)).rolling(window=30, center=True, min_periods=1).mean(skipna=True)

    plt.figure(figsize=(10,3), dpi=200)
    plt.plot(dia_m.time, fa*100, color='grey',alpha=0.5, label='precipitation > 10mm')
    plt.plot(dia_m.time, fa_smooth*100, color='black', label='smoothed')

    plt.ylabel('%')
    plt.grid()
    plt.legend()
    plt.savefig('fig15b.png')
    plt.close()

def main(mli_path, img_path, era_path, stg4_path, lsm_path, szn):
    
    mli = xr.open_dataset(mli_path)['tp3'][:,0,40:66,250:291]*1000
    img = xr.open_dataset(img_path)['predictors'][:-7,0,40:66,250:291]*1000
    era = xr.open_dataset(era_path)['predictors'][:-7,0,40:66,250:291]*1000
    
    diag_day = precip_sum12(mli, '1D')
    imerg_day = precip_sum12(img, '1D')
    era5_day = precip_sum12(era, '1D')

    stage4 = xr.open_dataset(stg4_path)['predictors'][:,0,40:66,250:291]
    
    lsm = xr.open_dataset(lsm_path)['lsm']
    lsm = lsm.rename({'longitude':'lon'})
    lsm = lsm.rename({'latitude':'lat'})

    # mask non-land regions
    img = imerg_day[:].where(lsm[40:66,250:291])
    era = era5_day[:].where(lsm[40:66,250:291])
    dia = diag_day[:].where(lsm[40:66,250:291])
    sg4 = stage4[:].where(lsm[40:66,250:291])
    
    valid = sg4.notnull()

    img_m = img.where(valid)
    sg4_m = sg4.where(valid)
    mcorr_img = xr.corr(img_m, sg4_m, dim="time")

    thresh = 0.3

    dia_m = dia.where(mcorr_img >= thresh)
    img_m = img.where(mcorr_img >= thresh)
    era_m = era.where(mcorr_img >= thresh)
    sg4_m = sg4.where(mcorr_img >= thresh)
    
    mli_fss, fa = get_FSS_wtime(dia_m, sg4_m, 10, 5)
    era_fss, fa = get_FSS_wtime(era_m, sg4_m, 10, 5)
    img_fss, fa = get_FSS_wtime(img_m, sg4_m, 10, 5)
    
    plot_fss_w_time(mli_fss, img_fss, era_fss)
    plot_stage4_precip(fa)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    parser.add_argument("--stg4", required=True)
    parser.add_argument("--lsm", required=True)
    parser.add_argument("--szn", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era, args.stg4, args.lsm)
    