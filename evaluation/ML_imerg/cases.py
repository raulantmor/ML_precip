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
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d

from evaluation.utils.plotting import blue_green_cmap
from evaluation.utils.data import precip_sum

def get_FSS(data, truth, q, n):

    Nt = 1
    FSS = np.empty(Nt)
    
    kernel = np.ones((n, n)) / (n ** 2)
    
    fc_grid = data
    gt_grid = truth

    fc_bin = np.where(fc_grid >= q, 1, 0)
    gt_bin = np.where(gt_grid >= q, 1, 0)

    M = convolve2d(fc_bin, kernel, mode='valid')
    O = convolve2d(gt_bin, kernel, mode='valid')

    MSE = np.mean((O - M) ** 2)
    MSE_ref = (np.mean(O ** 2) + np.mean(M ** 2))

    if MSE_ref == 0:
        FSS = 1
    else:
        FSS = 1 - (MSE / MSE_ref)
    
    return FSS

def get_3panels_axis(figure, panel):
    axis = plt.subplot(1, 3, panel, projection=proj)
    axis.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.6)
    axis.coastlines(color=(0.5, 0.5, 0.5))
    axis.gridlines(linewidth=0.4, zorder=1)
    return axis

def get_3panel_figure():
    figure = plt.figure(figsize=(18, 8), dpi=200)
    return figure, [get_3panels_axis(figure, 1),
                    get_3panels_axis(figure, 2),
                    get_3panels_axis(figure, 3)]

def plot_1field(ax, lon, lat, field_f, vmin, vmax):
    field_f = np.maximum(field_f, 0.00001)
    fills = ax.contourf(lon, lat, field_f,
                        levels=np.geomspace(1, 80, 10), locator=ticker.LogLocator(), cmap=bg_precip, 
                        extend='both', transform=transform, zorder=-1)
    return ax, fills

def plot_case(mli_day, img_day, era_day):
    
    proj = ccrs.PlateCarree()
    transform = ccrs.PlateCarree()
    
    log_scale_values = np.geomspace(1, 80, 10).round(decimals=1)

    time = '2023-09-05' # default: Rio Grande do Sul floods. Typhoon Chaba: '2022-07-01'
    fc_plot = mli_day[:,105:136,-65:-34].sel(time=time) # default: Rio Grande do Sul floods. Typhoon Chaba: [:,60:91,98:129]
    gt_plot = img_day[:,105:136,-65:-34].sel(time=time)
    third_plot = era_day[:,105:136,-65:-34].sel(time=time)
    time_str = str(gt_plot.time.values)[:10]

    ml_fss = f'FSS 50mm: {get_FSS(fc_plot, gt_plot, 50, 5).round(decimals=2)}'
    era_fss = f'FSS 50mm: {get_FSS(third_plot, gt_plot, 50, 5).round(decimals=2)}'

    fig, axs = get_3panel_figure()

    vmin = 0
    vmax = img_day.max()

    axs[0], _ = plot_1field(axs[0], fc_plot['lon'], fc_plot['lat'], fc_plot, vmin, vmax)
    axs[0].text(0.95, 0.95, r"ML$_{\rm IMERG}$", transform=axs[0].transAxes, fontsize=15, va='top', ha='right',
              backgroundcolor='white')
    axs[0].text(0.05, 0.05, ml_fss, transform=axs[0].transAxes, fontsize=12, va='bottom', ha='left',
              backgroundcolor='white')

    axs[1], _ = plot_1field(axs[1], gt_plot['lon'], gt_plot['lat'], gt_plot, vmin, vmax)
    axs[1].text(0.95, 0.95, 'IMERG', transform=axs[1].transAxes, fontsize=15, va='top', ha='right',
              backgroundcolor='white')

    axs[2], _ = plot_1field(axs[2], third_plot['lon'], third_plot['lat'], third_plot, vmin, vmax)
    axs[2].text(0.95, 0.95, 'ERA5', transform=axs[2].transAxes, fontsize=15, va='top', ha='right',
              backgroundcolor='white')
    axs[2].text(0.05, 0.05, era_fss, transform=axs[2].transAxes, fontsize=12, va='bottom', ha='left',
              backgroundcolor='white')

    # Adjust spacing between panels
    plt.subplots_adjust(wspace=0.1)

    plt.suptitle(f'24 Hour Total Precipitation\n2023 Rio Grande do Sul floods - {time_str}', y=0.83, fontsize=16, x=0.125, 
                 horizontalalignment='left')
    cbar = plt.colorbar(_, ax=axs, orientation='vertical', pad=0.02, format=ticker.LogFormatterSciNotation(), 
                        shrink=0.7)
    cbar.set_label('mm')

    cbar.update_ticks()
    cbar.ax.set_yticklabels(log_scale_values)

    plt.savefig('fig11.png')
    plt.close()
    
def main(mli_path, img_path, era_path):

    mli = xr.open_dataset(mli_path)['tp3'][:,0,:,:]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mli_d = precip_sum(mli, '1d')
    img_d = precip_sum(img, '1d')
    era_d = precip_sum(era, '1d')

    plot_case(mli_d, img_d, era_d)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era)
    
