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

from evaluation.utils.plotting import blue_green_cmap
from evaluation.utils.data import precip_sum

##### Example precip field w/ verification #####

def get_axis(figure, panel):
    axis = plt.subplot(1, 2, panel, projection=proj)
    axis.coastlines(linewidth=0.1)
    axis.gridlines(linewidth=0.15, zorder=1)
    return axis

def get_figure():
    figure = plt.figure(figsize=(10, 4), dpi=200)
    return figure, [get_axis(figure, 1), get_axis(figure, 2)]

def plot_field(ax, lon, lat, field_f):

    fills = ax.contourf(lon, lat, field_f, levels=np.geomspace(0.1, 10, 10),
        locator=ticker.LogLocator(), cmap=bg_precip, extend='both', transform=transform, zorder=-1)
    return ax, fills

def make_precip_figure(mle, era):
    bg_precip = blue_green_cmap()

    proj = ccrs.PlateCarree(central_longitude=180)
    transform = ccrs.PlateCarree()
    
    fc_plot = mle.sel(time='2022-04-30T00')
    era_plot = era.sel(time='2022-04-30T00')
    
    fig, ax = get_figure()

    ax[0], fills = plot_field(ax[0], fc_plot['lon'], fc_plot['lat'], fc_plot)
    ax[1], _ = plot_field(ax[1], era_plot['lon'], era_plot['lat'], era_plot)

    ax[0].text(0.95, 0.95, r"ML$_{\rm IMERG}$", transform=ax[0].transAxes, fontsize=8, va='top', ha='right',
              backgroundcolor='white')
    ax[1].text(0.95, 0.95, 'ERA5', transform=ax[1].transAxes, fontsize=8, va='top', ha='right',
              backgroundcolor='white')

    log_scale_values = np.geomspace(0.1, 10, 10).round(decimals=1)


    cbar = plt.colorbar(_, ax=ax, orientation='vertical', pad=0.02, format=ticker.LogFormatterSciNotation(), 
                        shrink=0.5)
    cbar.set_label('mm')

    cbar.update_ticks()
    cbar.ax.set_yticklabels(log_scale_values)
    plt.subplots_adjust(wspace=0.05, right=0.77)

    plt.savefig('fig3ab.png')
    plt.close()
    

##### Correlation of 24h totals #####

def plot_field_corr(ax, lon, lat, field_f):
    cfield, clon = add_cyclic_point(field_f, lon)
    fills = ax.contourf(
        clon, lat, cfield,
        levels=np.arange(0.0, 1.01, 0.05),
        cmap='jet',
        extend='both',
        transform=transform,
        zorder=-1
    )
    return fills

def make_corr_figure(mle_d, era_d):
    
    proj = ccrs.PlateCarree(central_longitude=180)
    transform = ccrs.PlateCarree()
    
    corr = xr.corr(mle_d, era_d, dim="time")

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = plt.subplot(1, 1, 1, projection=proj)

    ax.coastlines(linewidth=0.1)
    
    fills = plot_field_corr(ax, corr['lon'], corr['lat'], corr)

    cbar = plt.colorbar(fills, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
    cbar.set_label('Correlation')

    plt.savefig('fig3d.png')
    plt.close()

def main(mle_path, era_path):

    mle = xr.open_dataset(mle_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mle_d = precip_sum(mle, '1d')
    era_d = precip_sum(era, '1d')

    make_precip_figure(mle, era)
    make_corr_figure(mle_d, era_d)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mle", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mle, args.era)