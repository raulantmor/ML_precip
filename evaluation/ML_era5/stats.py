import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import ticker, cm
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from evaluation.utils.data import precip_sum
from evaluation.utils.metrics import get_FSS, compute_ets, compute_seeps_score_with_contributions

def plot_fss(mle, era, mle_d, era_d):
    
    period = '1D'
    bins = [1, 2, 5, 10, 15, 30, np.inf]
    bin_labels = ['1-2', '2-5', '5-10', '10-15', '15-30', '>30']
    units = 'mm/24h'

    bins3 = [0.1, 0.25, 0.5, 1, 2, 5, np.inf]
    bin_labels3 = ['0.1-0.25','0.25-0.5', '0.5-1', '1-2', '2-5','>5']
    units3 = 'mm/3h'

    diago_FSS, FSS_random, FSS_uniform = get_FSS(mle_d, era_d, bins, 5)
    diag_FSS3, FSS_random3, FSS_uniform3 = get_FSS(mle, era, bins3, 5)

    x_values = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    ax.plot(x_values, diago_FSS, color='cornflowerblue', marker='s',
            label=r"ML$_{\rm ERA5}$ (24h)")
    ax.plot(x_values, FSS_uniform, color='grey', linestyle='--',
            label='Skillful forecast (24h)')

    ax.plot(x_values, diag_FSS3, color='navy', marker='o',
            label=r"ML$_{\rm ERA5}$ (3h)")
    ax.plot(x_values, FSS_uniform3, color='black', linestyle='--',
            label='Skillful forecast (3h)')

    ax.set_ylabel('Mean Fraction Skill Score')
    ax.set_ylim(0, 1)

    ax.set_xticks(x_values)
    ax.set_xticklabels(bin_labels3)
    ax.set_xlabel(f"Precipitation Intensity {units3}", color='navy')

    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks(x_values)
    ax_top.set_xticklabels(bin_labels)
    ax_top.set_xlabel(f"Precipitation Intensity {units}", color='cornflowerblue')

    ax.legend(loc=(0.7, 0.21), frameon=True)
    ax.grid()

    plt.savefig('fig4a.png')
    
def plot_ets(mle, era, mle_d, era_d):
    bins = [0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10]
    bins_ = [2.5, 5, 10, 20, 30, 40, 50]
    scores_paper = [0.825, 0.74, 0.7, 0.59, 0.51, 0.45, 0.38]

    ets_scores = {thr: compute_ets(mle, era, thr) for thr in bins}
    ets_scores_ = {thr: compute_ets(mle_d, era_d, thr) for thr in bins_}

    scores = list(ets_scores.values())
    scores_ = list(ets_scores_.values())

    thresh = list(ets_scores.keys())
    thresh_ = list(ets_scores_.keys())

    fig, ax1 = plt.subplots(figsize=(8,5), dpi=150)

    ax1.plot(thresh, scores, marker='o', color='navy', label=r"3h ML$_{\rm ERA5}$")
    ax1.set_xlabel("3-hour Precip. Threshold (mm/3hr)", color='navy')
    ax1.set_ylabel("Equitable Threat Score (ETS)")
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twiny()
    ax2.plot(thresh_, scores_paper, marker='s', linestyle='--', color='grey', label='Göber et al. (2008)\n"Perfect Model"')
    ax2.plot(thresh_, scores_, marker='s', color='cornflowerblue', label=r"Daily ML$_{\rm ERA5}$")

    ax2.set_xlabel("Daily Precip. Threshold (mm/24hr)", color='cornflowerblue')
    ax2.tick_params(axis='x', colors='grey')
    for spine in ax2.spines.values():
        spine.set_edgecolor('grey')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('fig4b.png')
    plt.close()

def plot_seeps(mle_d, era_d, P, threshold):
    score, counts, values = compute_seeps_score_with_contributions(mle_day, era_day, threshold, P)

    days = pd.date_range('2022-01-01', '2024-12-31', freq='1D')

    plt.figure(figsize=(15,3), dpi=200)

    diag_skill = pd.Series((1-score)).rolling(window=1, center=True).mean(skipna=True)

    plt.plot(days, diag_skill, label=r"ML$_{\rm ERA5}$", color='cornflowerblue')
    plt.plot(days, np.full(len(days),0.72), color='black',linestyle='--', label='Rodwell et al. (2008)\n"Perfect Model"')

    plt.ylim(0.65,1)
    plt.xlabel('Time')
    plt.ylabel('1-SEEPS')
    plt.grid()
    plt.legend()
    plt.savefig('fig4c.png')
    plt.close()
    
def main(mle_path, era_path, P_path, thresh_path):

    mle = xr.open_dataset(mle_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mle_d = precip_sum(mle, '1d')
    era_d = precip_sum(era, '1d')
    
    P = xr.open_dataset(P_path)['predictors'][0,0,20:-20,:]
    threshold = xr.open_dataset(thresh_path)['predictors'][0,0,20:-20,:]

    plot_fss(mle, era, mle_d, era_d)
    plot_ets(mle, era, mle_d, era_d)
    plot_seeps(mle_d[:,20:-20,:], era_d[:,20:-20,:], P, threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mle", required=True)
    parser.add_argument("--era", required=True)
    parser.add_argument("--P", required=True)
    parser.add_argument("--thresh", required=True)
    args = parser.parse_args()

    main(args.mle, args.era, args.P, args.thresh)