import xarray as xr
import numpy as np
import pandas as pd
from matplotlib import ticker, cm
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from evaluation.utils.data import precip_sum
from evaluation.utils.metrics import compute_ets, compute_seeps_score_with_contributions

def get_FSS(data, truth, thresholds, n):
    """
    Computes Fraction Skill Score (FSS) for data in format (time,lat,lon).
    Returns array of FSS scores with length=Nt.
    
    Parameters:
    - data: model output to be evaluated (time, lat, lon)
    - truth: verification data (time, lat, lon)
    - thresholds: array of threshold values for binning
    - n: size of box for fractions
    """
    Nt = len(data)
    FSS = np.empty((Nt, len(thresholds) - 1))
    FSS_uniform = np.empty(len(thresholds) - 1)
    
    kernel = np.ones((n, n)) / (n ** 2)
    
    for t in range(Nt):
        fc_grid = data[t, :, :]
        gt_grid = truth[t, :, :]
        
        for thr_idx in range(len(thresholds) - 1):
            lower_bound = thresholds[thr_idx]
            upper_bound = thresholds[thr_idx + 1]

            fc_bin = np.where((fc_grid >= lower_bound) & (fc_grid < upper_bound), 1, 0)
            gt_bin = np.where((gt_grid >= lower_bound) & (gt_grid < upper_bound), 1, 0)

            # Compute fractions using 2D convolution (moving average over n x n boxes)
            M = convolve2d(fc_bin, kernel, mode='valid')
            O = convolve2d(gt_bin, kernel, mode='valid')

            # Calculate Mean Squared Errors
            MSE = np.mean((O - M) ** 2)
            MSE_ref = (np.mean(O ** 2) + np.mean(M ** 2))

            # FSS calculation
            if MSE_ref == 0:
                FSS[t, thr_idx] = np.nan
            else:
                FSS[t, thr_idx] = 1 - (MSE / MSE_ref)
                
            if t == 0:  # initialize accumulator
                F_mean = np.zeros(len(thresholds) - 1)
                O_mean = np.zeros(len(thresholds) - 1)
            F_mean[thr_idx] += np.mean(M)
            O_mean[thr_idx] += np.mean(O)
            
    # Average over time
    F_mean /= Nt
    O_mean /= Nt

    # Compute random baseline FSS
    for thr_idx in range(len(thresholds) - 1):
        num = O_mean[thr_idx]
        FSS_uniform[thr_idx] = 0.5 + (num / 2)
    
    return np.nanmean(FSS,axis=0), FSS_uniform

def plot_fss(mli, img, era):
    
    period = '1D'
    bins = [1, 2, 5, 10, 15, 30, np.inf]
    bin_labels = ['1-2', '2-5', '5-10', '10-15', '15-30', '>30']
    units = 'mm/day'

    era5_FSS, uniform_FSSe = get_FSS(era, img, bins, 5)
    diag_FSS, uniform_FSSd = get_FSS(mli, img, bins, 5)

    x_values = np.arange(len(bin_labels))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    ax.plot(x_values, era5_FSS, color='crimson', marker='d', label='ERA5')
    ax.plot(x_values, diag_FSS, color='cornflowerblue', marker='s', label=r"ML$_{\rm IMERG}$")
    ax.plot(x_values, uniform_FSSe, color='black', linestyle='--', label='Skillful forecast')

    ax.set_ylabel('Mean Fraction Skill Score')
    ax.set_ylim(0, 1)
    
    ax.set_xticks(x_values)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(f"Precipitation Intensity {units}")

    ax.legend(loc='lower right', frameon=True)
    ax.grid()
    plt.savefig('fig7a.png')
    plt.close()
    
def plot_ets(mli, img, era):
    
    bins = [2.5, 5, 10, 20, 30, 40, 50]
    scores_paper = [0.825, 0.74, 0.7, 0.59, 0.51, 0.45, 0.38]

    ets_scores = {thr: compute_ets(mli, img, thr) for thr in bins}
    ets_scores2 = {thr: compute_ets(era, img, thr) for thr in bins}

    scores = list(ets_scores.values())
    scores2 = list(ets_scores2.values())
    thresh = list(ets_scores.keys())

    plt.figure(figsize=(8,5), dpi=150)
    plt.plot(thresh, scores2, marker='d',color='crimson', label='ERA5')
    plt.plot(thresh, scores, marker='s',color='cornflowerblue', label=r"ML$_{\rm IMERG}$")
    plt.plot(thresh, scores_paper, marker='o', linestyle='--', color='black', label='Göber et al. (2008)\n"Perfect Model"')

    plt.xlabel("24h Precip. Threshold (mm)")
    plt.ylabel('Equitable Threat Score (ETS)')
    plt.legend()
    plt.grid()
    plt.ylim(0,1)
    plt.savefig('fig7b.png')
    plt.close()
    
def plot_seeps(mli, img, era, P, threshold):
    
    seeps_era5, counts1, values1 = compute_seeps_score_with_contributions(era, 
                                                                  img, threshold, P)
    seeps_diag, counts2, values2 = compute_seeps_score_with_contributions(mli, 
                                                                      img, threshold, P)

    days = pd.date_range('2022-01-01', '2024-12-31', freq='1D')

    plt.figure(figsize=(15,3), dpi=200)

    era_skill = pd.Series((1-seeps_era5)).rolling(window=5, center=True).mean(skipna=True)
    diag_skill = pd.Series((1-seeps_diag)).rolling(window=5, center=True).mean(skipna=True)

    plt.plot(days, era_skill, label='ERA5', color='crimson')
    plt.plot(days, diag_skill, label=r"ML$_{\rm IMERG}$", color='cornflowerblue')
    plt.plot(days, np.full(len(days),0.72), color='black',linestyle='--', label='Rodwell et al. (2008) "Perfect Model"')

    plt.ylim(0.4,1)
    plt.xlabel('Time')
    plt.ylabel('1-SEEPS')
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('fig7c.png')
    plt.close()
    
def main(mli_path, img_path, era_path, P_path, thresh_path):

    mli = xr.open_dataset(mle_path)['tp3'][:,0,:,:]*1000
    img = xr.open_dataset(img_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mli_d = precip_sum(mli, '1d')
    img_d = precip_sum(img, '1d')
    era_d = precip_sum(era, '1d')
    
    P = xr.open_dataset(P_path)['predictors'][0,0,20:-20,:]
    threshold = xr.open_dataset(thresh_path)['predictors'][0,0,20:-20,:]

    plot_fss(mli_d, img_d, era_d)
    plot_ets(mli, img, era)
    plot_seeps(mli_d[:,20:-20,:], img_d[:,20:-20,:], era_d[:,20:-20,:], P, threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mli", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--era", required=True)
    parser.add_argument("--P", required=True)
    parser.add_argument("--thresh", required=True)
    args = parser.parse_args()

    main(args.mli, args.img, args.era, args.P, args.thresh)