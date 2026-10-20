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

def monthly_szn(data,season):
    ds_szn = data.sel(time = data['time.season']==season)
    return ds_szn

def calc_precip_stats(model, valid, threshold):
    '''
    Calculates POD, FAR, CSI stats

    model: precip prediction (time,lat,lon)
    valid: validation (time,lat,lon)
    threshold: precip threshold in mm/day
    '''
    hits = ((model > threshold) & (valid > threshold))
    n_hits = hits.sum().item()

    misses = ((valid > threshold) & (model <= threshold))
    n_misses = misses.sum().item()

    fa = ((model > threshold) & (valid <= threshold))
    n_fa = fa.sum().item()
    
    return [n_hits, n_misses, n_fa]

def plot_performance_diagram_pretty(hits, misses, false_alarms,
                                     model_labels=None,
                                     colors=None,
                                     markers=None,
                                     figsize=(8, 6)):
    hits = np.array(hits)
    misses = np.array(misses)
    false_alarms = np.array(false_alarms)

    pod = hits / (hits + misses)
    far = false_alarms / (hits + false_alarms)
    sr = 1 - far
    csi = hits / (hits + misses + false_alarms)
    bias = (hits + false_alarms) / (hits + misses)

    fig, ax = plt.subplots(figsize=(10,6), dpi=200)

    # CSI contours
    csi_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    sr_grid, pod_grid = np.meshgrid(np.linspace(0.01, 1, 300), np.linspace(0.01, 1, 300))
    
    csi_grid = (sr_grid * pod_grid) / (sr_grid + pod_grid - sr_grid * pod_grid)
    cs = ax.contour(sr_grid, pod_grid, csi_grid, levels=csi_levels, colors='black',
                    linestyles='--', linewidths=2)
    ax.clabel(cs, fmt='%0.1f', fontsize=12)

    # Frequency Bias lines + labels
    bias_vals = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    sr_line = np.linspace(0.01, 1.1, 300)
    
    for b in bias_vals:
        pod_line = b * sr_line
        pod_line[pod_line > 1.1] = np.nan

        ax.plot(sr_line, pod_line, linestyle=':', color='gray', linewidth=1, alpha=0.6)

        # label position
        if b >= 1:
            x_label = .4 / b
        else:
            x_label = .4
        y_label = b * x_label

        if y_label <= 1.0:
            ax.text(
                x_label, y_label,
                f'{b}',
                fontsize=10, color='gray',
                ha='left', va='bottom',
                #rotation=np.degrees(np.arctan(b))
            )

    # Plot points
    n_points = len(hits)
    if model_labels is None:
        model_labels = [f'Model {i+1}' for i in range(n_points)]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_points))
    if markers is None:
        markers = ['o'] * n_points

    for i in range(n_points):
        ax.scatter(sr[i], pod[i],
                   color=colors[i],
                   marker=markers[i],
                   s=120,
                   label=model_labels[i])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('SR (1 - FAR)', fontsize=14)
    ax.set_ylabel('POD', fontsize=14)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_aspect('equal')
    ax.grid(False)
    ax.tick_params(labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='lower left', fontsize=10).set_visible(False)
    plt.tight_layout()
    #plt.title('Eastern US: 2022-2024', weight='bold', fontsize=15, y=1.05)
    plt.show()

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
    
    thresholds = [5, 10, 20]  # Thresholds to evaluate
    models = [('ERA5', monthly_szn(era_m, szn)), ('IMERG', monthly_szn(img_m, szn)), 
              (r'ML$_{\rm IMERG}$', monthly_szn(dia_m, szn))]
    hits, misses, false_alarms = [], [], []
    model_labels = []
    colors = []
    markers = []

    color_map = {'ERA5': 'crimson', 'IMERG': 'grey', r'ML$_{\rm IMERG}$': 'cornflowerblue'}
    marker_map = {5: 'o', 10: 's', 20: 'D'}

    for name, model_data in models:
        for t in thresholds:
            stats = calc_precip_stats(model_data, monthly_szn(sg4_m, szn), threshold=t)
            hits.append(stats[0])
            misses.append(stats[1])
            false_alarms.append(stats[2])
            model_labels.append(f'{name} ({t} mm)')
            colors.append(color_map[name])
            markers.append(marker_map[t])

    #correct_negatives = [1] * len(hits)  # Placeholder, not used
    plot_performance_diagram_pretty(hits, misses, false_alarms,
                                    model_labels=model_labels,
                                    colors=colors,
                                    markers=markers)
    
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