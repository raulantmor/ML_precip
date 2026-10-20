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

def monthly_szn(data, season):
    ds_szn = data.sel(time = data['time.season'] == season)
    ds_1m_szn = ds_szn.resample(time='1M').sum()
    ds_szn_m = ds_1m_szn.mean(axis=0)
    return ds_szn_m

def format_axis(ax):
    ax.coastlines(color=(0.5, 0.5, 0.5), linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.2)
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.4, color='gray',
        alpha=0.5, linestyle='--', zorder=1
    )
    gl.xlocator = ticker.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = ticker.FixedLocator(np.arange(-90, 91, 10))

def plot_field(ax, lon, lat, field):
    field = np.maximum(field, 0)
    fill = ax.contourf(
        lon, lat, field,
        levels=lvls,
        locator=ticker.LogLocator(),
        cmap='BrBG',
        extend='both',
        transform=transform,
        zorder=-1
    )
    return fill

def make_avg_monthly_fig(era_m, img_m, dia_m, sg4_m):
    proj = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    seasons = ["DJF", "MAM", "JJA", "SON"]
    lvls = [10,15,25,50,75,100,150,200,250,300]

    models = [era_m, img_m, dia_m, sg4_m]
    model_names = ["ERA5", "IMERG", r"ML$_{\rm IMERG}$", "StageIV"]
    
    seasonal_fields = {}   # dict: seasonal_fields[model_name][season] = DataArray

    for model_name, model_data in zip(model_names, models):
        seasonal_fields[model_name] = {}
        for szn in seasons:
            seasonal_fields[model_name][szn] = monthly_szn(model_data, szn)

    fig, axes = plt.subplots(
        4, 4,
        figsize=(15, 10),
        dpi=200,
        subplot_kw={"projection": proj}
    )

    fill_for_cbar = None

    for i, szn in enumerate(seasons):         # ROWS = seasons
        for j, model_name in enumerate(model_names):   # COLS = models

            ax = axes[i, j]
            format_axis(ax)

            data_szn = seasonal_fields[model_name][szn]

            lon = data_szn["lon"]
            lat = data_szn["lat"]

            fill = plot_field(ax, lon, lat, data_szn)
            if fill_for_cbar is None:
                fill_for_cbar = fill

            # Titles across columns
            if i == 0:
                ax.set_title(model_name, fontsize=15)

            if j == 0:
                ax.text(
                    -0.1, 0.5, szn,
                    va="center", ha="right", rotation=90,
                    transform=ax.transAxes, fontsize=15
                )
                
    cbar = fig.colorbar(
        fill_for_cbar,
        ax=axes,
        orientation="vertical",
        pad=0.015,
        fraction=0.025,
        format=ticker.LogFormatterSciNotation(), shrink=0.8
    )

    log_scale_values = lvls
    cbar.set_label("mm")
    cbar.set_ticks(log_scale_values)
    cbar.set_ticklabels(log_scale_values)

    fig.subplots_adjust(right=0.85, hspace=0.1, wspace=0.1)

    plt.savefig('fig12.png')
    plt.close()

def main(mli_path, img_path, era_path, stg4_path, lsm_path):

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
    dia_m = dia.where(valid)
    era_m = era.where(valid)
    sg4_m = sg4.where(valid)
    
    make_avg_monthly_fig(era_m, img_m, dia_m, sg4_m)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    parser.add_argument("--stg4", required=True)
    parser.add_argument("--lsm", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era, args.stg4, args.lsm)

    