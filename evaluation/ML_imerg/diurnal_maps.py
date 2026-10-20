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

# make colormap
base = plt.cm.RdBu(np.linspace(0, 1, 256))

base[:5] = mcolors.to_rgba("#f2c0ef")
base[5:10] = mcolors.to_rgba("#e869e1")
base[10:15] = mcolors.to_rgba("#e616db")

base[-15:-10] = mcolors.to_rgba("#12B50D")
base[-10:-5] = mcolors.to_rgba("#68d165")
base[-5:] = mcolors.to_rgba("#baedb9") 

custom_cmap = mcolors.ListedColormap(base)

def add_local_hour(ds):
    # ds dims: (time, lat, lon)
    
    ####
    ds = ds / 3
    ds = ds - ds.mean(axis=0)

    # local time offset in hours
    offset = ds.lon / 15.0

    # broadcast time + lon
    local_time = ds.time + offset * np.timedelta64(1, "h")

    # assign local hour as a coordinate
    return ds.assign_coords(local_hour=local_time.dt.hour)

def get_maps(ds_lt):
    one = ds_lt.where(
        (ds_lt.local_hour >= 8) & (ds_lt.local_hour < 11),
        drop=True
    )

    two = ds_lt.where(
        (ds_lt.local_hour >= 12) & (ds_lt.local_hour < 15),
        drop=True
    )

    three = ds_lt.where(
        (ds_lt.local_hour >= 18) & (ds_lt.local_hour < 21),
        drop=True
    )

    one_mean = one.mean("time", skipna=True)
    two_mean = two.mean("time", skipna=True)
    three_mean = three.mean("time", skipna=True)
    return one_mean, two_mean, three_mean

def plot_diurnal_maps(mli_lt, img_lt, era_lt):
    
    mli_1, mli_2, mli_3  = get_maps(mli_lt)
    era_1, era_2, era_3  = get_maps(era_lt)
    img_1, img_2, img_3  = get_maps(img_lt)
    
    vmin = -15
    vmax =  15

    fig, axs = plt.subplots(
        3, 3,
        figsize=(15, 12), dpi=200,
        subplot_kw=dict(projection=ccrs.PlateCarree()),

    )

    datasets = [
        (r"ML$_{\rm IMERG}$", mli_1 * 24, mli_2 * 24, mli_3 * 24),
        ("IMERG", img_1 * 24, img_2 * 24, img_3 * 24),
        ("ERA5",  era_1 * 24, era_2 * 24, era_3 * 24),
    ]

    cmap = custom_cmap
    levels = np.arange(-10, 10.5, 1)

    # ---- plotting ----
    for col, (_, one, two, three) in enumerate(datasets):
        for row, data in enumerate([one, two, three]):

            ax = axs[row, col]

            pcm = ax.contourf(
                data.lon,
                data.lat,
                data,
                levels=levels,
                extend="both",
                cmap=cmap,
                transform=ccrs.PlateCarree(),
            )

            ax.coastlines()

    for col, (name, _, _, _) in enumerate(datasets):
        axs[0, col].set_title(name, fontsize=15, weight='bold')

    axs[0, 1].text(-20,0.5,"08–11 LT", va="center", ha="right", rotation=90, fontsize=15)
    axs[1, 1].text(-20,0.5,"12–15 LT", va="center", ha="right", rotation=90, fontsize=15)
    axs[2, 1].text(-20,0.5,"18–21 LT", va="center", ha="right", rotation=90, fontsize=15)

    cbar = fig.colorbar(
        pcm,
        ax=axs,
        orientation="horizontal",
        pad=0.08,
        aspect=40,
    )
    cbar.set_label("Precipitation anomaly (mm/day)", fontsize=15)

    fig.subplots_adjust(bottom=0.28, hspace=0.1, wspace=0.1)
    plt.savefig('fig9.png')
    plt.close()

def main(mli_path, img_path, era_path):

    mli = xr.open_dataset(mle_path)['tp3'][:,0,60:111,70:151]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,60:111,70:151]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,60:111,70:151]*1000

    new_time = pd.date_range(mli.time[0].values, mli.time[-1].values, freq='1h')
    mli_int = mli.interp(time=new_time)
    img_int = img.interp(time=new_time)
    era_int = era.interp(time=new_time)
    
    mli_lt  = add_local_hour(mli_int)
    era_lt  = add_local_hour(era_int)
    img_lt  = add_local_hour(img_int)

    plot_diurnal_maps(mli_lt, img_lt, era_lt)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era)