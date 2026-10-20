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

def diurnal_land_localtime(ds, lsm):
    ds_1hr = ds / 3
    anom_1hr = ds_1hr - ds_1hr.mean(axis=0)
    ds_1hr = anom_1hr.where(lsm==1)

    # compute local-time offset map thing (this is in hours)
    lon = ds_1hr.lon
    offset_hours = lon / 15.0

    offset = xr.DataArray(
        offset_hours,
        dims="lon",
        coords={"lon": lon}
    )

    # shift time to local solar time
    local_time = ds_1hr["time"] + offset * np.timedelta64(1, "h")
    ds_local = ds_1hr.assign_coords(local_time=local_time)

    # group diurnal cycle in local time
    diurnal = ds_local.groupby("local_time.hour").mean(skipna=True)

    return diurnal

def diurnal_ocean_localtime(ds, lsm):

    ds_1hr = ds / 3
    anom_1hr = ds_1hr - ds_1hr.mean(axis=0)
    ds_1hr = anom_1hr.where(lsm==0)

    lon = ds_1hr.lon
    offset_hours = lon / 15.0

    offset = xr.DataArray(
        offset_hours,
        dims="lon",
        coords={"lon": lon}
    )

    local_time = ds_1hr["time"] + offset * np.timedelta64(1, "h")
    ds_local = ds_1hr.assign_coords(local_time=local_time)

    diurnal = ds_local.groupby("local_time.hour").mean(skipna=True)

    return diurnal

def plot_diurnal_cycle(mli, img, era, lsm):

    dmc_mli = diurnal_land_localtime(mli, lsm)
    dmc_era = diurnal_land_localtime(era, lsm)
    dmc_img = diurnal_land_localtime(img, lsm)

    odmc_mli = diurnal_ocean_localtime(mli, lsm)
    odmc_era = diurnal_ocean_localtime(era, lsm)
    odmc_img = diurnal_ocean_localtime(img, lsm)
    
    plt.figure(figsize=(10,3), dpi=200)
    
    # convert to daily rates
    m = pd.Series(dmc_mli.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)
    e = pd.Series(dmc_era.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)
    i = pd.Series(dmc_img.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)

    om = pd.Series(odmc_mli.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)
    oe = pd.Series(odmc_era.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)
    oi = pd.Series(odmc_img.mean(axis=0)*24).rolling(window=3, center=True).mean(skipna=True)

    plt.plot(dmc_img.hour, m, linestyle='-',
             color='cornflowerblue', label=r"ML$_{\rm IMERG}$ (land)")
    plt.plot(dmc_img.hour, e, linestyle='-',
             color='crimson', label="ERA5 (land)")
    plt.plot(dmc_img.hour, i, linestyle='-',
             color='black', label="IMERG (land)")

    plt.plot(dmc_img.hour, om, linestyle=':',
             color='cornflowerblue', label=r"ML$_{\rm IMERG}$ (ocean)")
    plt.plot(dmc_img.hour, oe, linestyle=':',
             color='crimson', label="ERA5 (ocean)")
    plt.plot(dmc_img.hour, oi, linestyle=':',
             color='black', label="IMERG (ocean)")

    plt.xlabel("local time")
    plt.ylabel("Diurnal Precipitation Rate Anomaly\n (mm/day)")
    plt.xticks([0,3,6,9,12,15,18,21])
    plt.title("Diurnal Cycle of Precipitation", horizontalalignment='left', x=0.01)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.savefig('fig8_diurnal_cycle.png')
    plt.close()
    
def main(mli_path, img_path, era_path, lsm_path):
    
    lsm = xr.open_dataset(lsm_path)['lsm'][60:111,70:151] # Maritime Continent Ex.
    lsm = lsm.rename({'longitude':'lon'})
    lsm = lsm.rename({'latitude':'lat'})

    mli = xr.open_dataset(mle_path)['tp3'][:,0,60:111,70:151]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,60:111,70:151]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,60:111,70:151]*1000

    new_time = pd.date_range(mli.time[0].values, mli.time[-1].values, freq='1h')
    mli_int = mli.interp(time=new_time)
    img_int = img.interp(time=new_time)
    era_int = era.interp(time=new_time)

    plot_diurnal_cycle(mli_int, img_int, era_int, lsm)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    parser.add_argument("--lsm", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era, args.lsm)