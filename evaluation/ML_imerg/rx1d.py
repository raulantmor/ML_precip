import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import matplotlib.colors as mcolors
from evaluation.utils.data import precip_sum

def monthly_szn(data, season):
    ds_szn = data.sel(time = data['time.season'] == season)
    ds_szn_m = ds_szn.max(dim='time')
    return ds_szn_m

def plot_rx1d_rmse(mli_rx1d_m, imerg_rx1d_m, era5_rx1d_m):
    
    times = imerg_rx1d_m.time

    era5_rmse = np.sqrt(((era5_rx1d_m - imerg_rx1d_m)**2).mean(axis=(1,2)))
    mli_rmse = np.sqrt(((mli_rx1d_m - imerg_rx1d_m)**2).mean(axis=(1,2)))

    plt.figure(figsize=(12, 3), dpi=150)
    plt.plot(times, era5_rmse, color='red', label='ERA5')
    plt.plot(times, mli_rmse, color='cornflowerblue', label=r"ML$_{\rm IMERG}$")

    plt.legend()
    plt.grid()
    plt.ylabel('Rx1d RMSE (mm)')

    plt.savefig('fig10a.png')
    plt.close()
    
def plot_rx1d_maps(mli_rx1d_m, imerg_rx1d_m, era5_rx1d_m):
    seasons = ['JJA', 'SON']

    img_rx1dm = {szn: monthly_szn(imerg_rx1d_m, szn) for szn in seasons}
    era_rx1dm = {szn: monthly_szn(era5_rx1d_m, szn) for szn in seasons}
    dia_rx1dm = {szn: monthly_szn(mli_rx1d_m, szn) for szn in seasons}

    lats = imerg_rx1d_m[12].coords['lat']
    lons = imerg_rx1d_m[12].coords['lon']

    vm = 180
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    fig, axes = plt.subplots(2, 3, figsize=(12, 4.5), dpi=250, subplot_kw={'projection': ccrs.PlateCarree()})

    for i, szn in enumerate(seasons):
        im1 = dia_rx1dm[szn].plot(
            ax=axes[i, 0], transform=ccrs.PlateCarree(), cmap='Spectral_r',
            vmin=0, vmax=vm, add_colorbar=False
        )
        axes[i, 0].set_extent(extent, crs=ccrs.PlateCarree())
        axes[i, 0].coastlines()
        axes[i, 0].add_feature(cfeature.BORDERS)
        axes[i, 0].set_title(f'ML Output ({szn})', fontsize=10)
        axes[i, 0].set_title(r"ML$_{\rm IMERG}$" f" ({szn})", fontsize=10)

        im2 = img_rx1dm[szn].plot(
            ax=axes[i, 1], transform=ccrs.PlateCarree(), cmap='Spectral_r',
            vmin=0, vmax=vm, add_colorbar=False
        )
        axes[i, 1].set_extent(extent, crs=ccrs.PlateCarree())
        axes[i, 1].coastlines()
        axes[i, 1].add_feature(cfeature.BORDERS)
        axes[i, 1].set_title(f'IMERG ({szn})', fontsize=10)

        im3 = era_rx1dm[szn].plot(
            ax=axes[i, 2], transform=ccrs.PlateCarree(), cmap='Spectral_r',
            vmin=0, vmax=vm, add_colorbar=False
        )
        axes[i, 2].set_extent(extent, crs=ccrs.PlateCarree())
        axes[i, 2].coastlines()
        axes[i, 2].add_feature(cfeature.BORDERS)
        axes[i, 2].set_title(f'ERA5 ({szn})', fontsize=10)

    plt.subplots_adjust(wspace=0.05)
    cbar = fig.colorbar(im1, ax=axes, orientation='vertical', pad=0.02)
    cbar.set_label('mm')

    plt.savefig('fig10b-g.png')
    plt.close()
    
def main():
    
    mli = xr.open_dataset(mle_path)['tp3'][:,0,60:111,70:151]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,60:111,70:151]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,60:111,70:151]*1000
    
    mli_day = precip_sum(mli,'1D')
    ERA5_day = precip_sum(era,'1D')
    IMERG_day = precip_sum(img,'1D')

    mli_rx1d_m = mli_day.resample(time='1M', label='left').max()
    era5_rx1d_m = ERA5_day.resample(time='1M', label='left').max()
    imerg_rx1d_m = IMERG_day.resample(time='1M', label='left').max()
    
    plot_rx1d_rmse(mli_rx1d_m, imerg_rx1d_m, era5_rx1d_m)
    plot_rx1d_maps(mli_rx1d_m, imerg_rx1d_m, era5_rx1d_m)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era)
