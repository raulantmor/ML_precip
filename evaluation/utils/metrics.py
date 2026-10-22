import xarray as xr
import numpy as np
import pandas as pd
from scipy.signal import convolve2d

def get_FSS(data, truth, thresholds, n):
    """
    Computes Fraction Skill Score (FSS) for data in format (time,lat,lon).
    Returns average FSS by threshold as well as the baselines FSS_random
    and FSS_uniform.
    
    Parameters:
    - data: model output to be evaluated (time, lat, lon)
    - truth: verification data (time, lat, lon)
    - thresholds: array of threshold values for binning
    - n: size of box for neighborhood fractions
    """
    Nt = len(data)
    FSS = np.empty((Nt, len(thresholds) - 1))
    FSS_random = np.empty(len(thresholds) - 1)
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
        num = 2 * F_mean[thr_idx] * O_mean[thr_idx]
        den = F_mean[thr_idx] ** 2 + O_mean[thr_idx] ** 2
        FSS_random[thr_idx] = np.nan if den == 0 else num / den
    for thr_idx in range(len(thresholds) - 1):
        num = O_mean[thr_idx]
        FSS_uniform[thr_idx] = 0.5 + (num / 2)
    
    return np.nanmean(FSS,axis=0), FSS_random, FSS_uniform

def compute_ets(model_precip, truth_precip, threshold):
    """
    Compute Equitable Threat Score (ETS) for a given precipitation threshold.
    
    Parameters:
    - model_precip: model output to be evaluated (time, lat, lon)
    - truth_precip: verification data (time, lat, lon)
    - threshold: float - Precipitation threshold to evaluate
    
    Returns:
    - ets: float - Equitable Threat Score for the given threshold
    """
    # Convert to binary occurrence based on threshold
    model_binary = model_precip >= threshold
    truth_binary = truth_precip >= threshold

    # Compute contingency table
    H = np.sum(model_binary & truth_binary)  # Hits
    F = np.sum(model_binary & ~truth_binary) # False Alarms
    M = np.sum(~model_binary & truth_binary) # Misses
    C = np.sum(~model_binary & ~truth_binary) # Correct Negatives

    # Compute expected hits by chance
    H_r = ((H + M) * (H + F)) / (H + M + F + C)

    # Compute ETS (avoid division by zero)
    if (H + M + F - H_r) == 0:
        return np.nan
    else:
        ets = (H - H_r) / (H + M + F - H_r)
        return ets
    
def assign_seeps_category(precip, threshold_LH):
    """
    
    Categorize precipitation based on SEEPS definitions.
    Data must be daily accumulations.
    """
    category = np.zeros_like(precip, dtype=int)
    category = np.where(precip > 0.20, 1, 0)
    category = np.where(precip > threshold_LH, 2, category)
    return category

def compute_seeps_score_with_contributions(forecast, observation, threshold_LH, P):
    fcst_cat = assign_seeps_category(forecast, threshold_LH)
    obs_cat = assign_seeps_category(observation, threshold_LH)

    time, lat, lon = observation.shape

    # SEEPS penalty matrix
    se_matrix = 0.5 * np.stack([
        [0 * P,           1 / (1 - P),         4 / (1 - P)],
        [1 / P,           0 * P,               3 / (1 - P)],
        [(1 / P) + (3 / (2 + P)), 3 / (2 + P),   0 * P]
    ], axis=0)

    seeps_score = np.empty((time, lat, lon))
    contrib_counts = np.zeros((time, 3, 3), dtype=int)
    contrib_values = np.zeros((time, 3, 3))
    for t in range(time):
        o = obs_cat[t]
        f = fcst_cat[t]

        for i in range(3):
            for j in range(3):
                contrib_counts[t, i, j] += np.sum((o == i) & (f == j))

        seeps_score[t] = se_matrix[o, f, np.arange(lat)[:, None], np.arange(lon)]
        seeps_score[t] = np.minimum(seeps_score[t], 1)

        # compute total contribution
        for i in range(3):
            for j in range(3):
                contrib_values[t, i, j] = contrib_counts[t, i, j] * np.mean(se_matrix[i, j])
    mean_score = np.mean(seeps_score, axis=(1, 2))

    return mean_score, np.sum(contrib_counts, axis=0), np.mean(contrib_values, axis=0)