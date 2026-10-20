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

def get_axis(fig, nrows, ncols, index):
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)
    ax.coastlines(color=(0.5, 0.5, 0.5), linewidth=0.4)
    return ax

def get_2x3_figure():
    fig = plt.figure(figsize=(12, 8), dpi=200)
    axs = [get_axis(fig, 3, 2, 1),
           get_axis(fig, 3, 2, 2),
           get_axis(fig, 3, 2, 3),
           get_axis(fig, 3, 2, 5),
           get_axis(fig, 3, 2, 6)]
    return fig, axs

def plot_1field(ax, lon, lat, field, vmin, vmax):
    cfield, clon = add_cyclic_point(field, lon)
    cfield = np.maximum(cfield, 0) #remove negatives (from remapping step)

    fills = ax.contourf(
        clon, lat, cfield,
        levels=log_scale_values,
        locator=ticker.LogLocator(),
        cmap=bg_precip,
        extend='both',
        transform=transform,
        zorder=-1,
    )
    return fills

def plot_corr(ax, lon, lat, field):
    corr, clon = add_cyclic_point(field, lon)
    
    corr = np.maximum(corr, 0)

    fills = ax.contourf(
        clon, lat, corr,
        levels=np.arange(0,1.01,0.05),
        cmap='jet',#'plasma',
        #extend='both',
        transform=transform,
        zorder=-1,
    )
    return fills

def make_imerg_corr_fig(mli, img, era):
    
    bg_precip = blue_green_cmap()

    mli_d_clean = mli.where(mli >= 0.2, 0)
    img_d_clean = img.where(img >= 0.2, 0)
    era_d_clean = era.where(era >= 0.2, 0)

    mask = (mli_d_clean != 0) | (img_d_clean != 0)
    mask2 = (era_d_clean != 0) | (img_d_clean != 0)

    mli_masked = mli_day.where(mask)
    img_masked = img_day.where(mask)
    img_masked2 = img_day.where(mask2)
    era_masked = era_day.where(mask2)
    
    corr1 = xr.corr(mli_masked, img_masked, dim="time")
    corr2 = xr.corr(era_masked, img_masked2, dim="time")

    proj = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()

    log_scale_values = np.geomspace(0.2, 50, 10).round(decimals=1)

    time = '2023-07-30'
    fc_plot   = mli.sel(time=time)
    gt_plot   = img.sel(time=time)
    era5_plot = era.sel(time=time)

    fig, axs = get_2x3_figure()

    vmin = 0
    vmax = img.max()


    fills_fc = plot_1field(axs[0], fc_plot.lon, fc_plot.lat, fc_plot, vmin, vmax)
    axs[0].text(0.97, 0.95, r"ML$_{\rm IMERG}$", transform=axs[0].transAxes,
                fontsize=9, ha='right', va='top', backgroundcolor='white')

    fills_gt = plot_1field(axs[2], gt_plot.lon, gt_plot.lat, gt_plot, vmin, vmax)
    axs[2].text(0.97, 0.95, 'IMERG', transform=axs[2].transAxes,
                fontsize=9, ha='right', va='top', backgroundcolor='white')

    fills_era5 = plot_1field(axs[3], era5_plot.lon, era5_plot.lat, era5_plot, vmin, vmax)
    axs[3].text(0.97, 0.95, 'ERA5', transform=axs[3].transAxes,
                fontsize=9, ha='right', va='top', backgroundcolor='white')


    diff1 = plot_corr(axs[1], corr1.lon, corr1.lat, corr1)
    axs[1].text(0.97, 0.95, r'ML$_{\rm IMERG}$', transform=axs[1].transAxes,
                fontsize=8, ha='right', va='top', backgroundcolor='white')


    diff2 = plot_corr(axs[4], corr2.lon, corr2.lat, corr2)
    axs[4].text(0.97, 0.95, 'ERA5', transform=axs[4].transAxes,
                fontsize=8, ha='right', va='top', backgroundcolor='white')
    fig.subplots_adjust(wspace=0.05, hspace=0.02)

    left_col_pos  = axs[0].get_position()   # ML column (left)
    right_col_pos = axs[1].get_position()   # IMERG/ERA5 column (right)

    # Create colorbar axes manually, perfectly centered with each column
    cax1 = fig.add_axes([
        left_col_pos.x0,
        0.1,                               # vertical placement
        left_col_pos.width,
        0.02                                # height
    ])
    cax2 = fig.add_axes([
        right_col_pos.x0,
        0.1,
        right_col_pos.width,
        0.02
    ])

    # Create the colorbars
    cbar1 = fig.colorbar(fills_fc, cax=cax1, orientation='horizontal')
    cbar1.set_label("mm")
    cbar1.ax.set_xticklabels(log_scale_values)

    cbar2 = fig.colorbar(diff1, cax=cax2, orientation='horizontal')
    cbar2.set_label("fig6.png")
    plt.close()
    
def main(mli_path, img_path, era_path):

    mli = xr.open_dataset(mli_path)['tp3'][:,0,:,:]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mli_d = precip_sum(mli[:,20:-20,:], '1d')
    img_d = precip_sum(img[:,20:-20,:], '1d')
    era_d = precip_sum(era[:,20:-20,:], '1d')

    make_imerg_corr_fig(mli_d, img_d, era_d)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era)