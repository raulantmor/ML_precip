import numpy as np
import matplotlib.pyplot as plt
from evaluation.utils.data import precip_sum

def calc_precip_stats(model, valid, threshold):
    '''
    Calculates POD, FAR, CSI stats

    model: precip prediction (time,lat,lon)
    valid: validation (time,lat,lon)
    threshold: precip threshold in mm/day
    '''
    hits = ((model > threshold) & (valid > threshold))
    n_hits = hits.sum().item()
    
    nhits = ((model < threshold) & (valid < threshold))
    n_nhits = nhits.sum().item()

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
    sr_grid, pod_grid = np.meshgrid(np.linspace(0.01, 1.1, 300), np.linspace(0.01, 1.1, 300))
    csi_grid = (sr_grid * pod_grid) / (sr_grid + pod_grid - sr_grid * pod_grid)
    cs = ax.contour(sr_grid, pod_grid, csi_grid, levels=csi_levels, colors='black',
                    linestyles='--', linewidths=2)
    ax.clabel(cs, fmt='%0.1f', fontsize=12)

    # Frequency Bias lines + labels
    bias_vals = [0.1, 0.25, 0.5, 1, 2, 4, 10]
    for b in bias_vals:
        sr_line = np.linspace(0.01, 1.1, 300)
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

    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    #plt.title()
    plt.savefig('fig4c.png')
    plt.close()

def main(mle_path, era_path):

    mle = xr.open_dataset(mle_path)['tp3'][:,0,:,:]*1000
    era = xr.open_dataset(era_path)['tp3'][:,0,:,:]*1000

    mle_d = precip_sum(mle, '1d')
    era_d = precip_sum(era, '1d')

    thresholds = [0.5, 1, 2]  # for 3-hour data
    thresholds2 = [5, 10, 20]  # for daily data

    models = [
        (r"ML$_{\rm ERA5}$, 3-h", mle),
        (r"ML$_{\rm ERA5}$, Daily", mle_d)
    ]

    color_map = {
        r"ML$_{\rm ERA5}$, 3-h": 'navy',
        r"ML$_{\rm ERA5}$, Daily": 'cornflowerblue'
    }

    marker_map = {0.5: 'o', 1: 's', 2: 'D'}
    marker_map2 = {5: 'o', 10: 's', 20: 'D'}

    hits, misses, false_alarms = [], [], []
    model_labels, colors, markers = [], [], []

    for name, model_data in models:
        if "3-h" in name:
            thresh_list, marker_dict = thresholds, marker_map
            data_ref = era
        else:
            thresh_list, marker_dict = thresholds2, marker_map2
            data_ref = era_d

        for t in thresh_list:
            stats = calc_precip_stats(model_data, data_ref, threshold=t)
            hits.append(stats[0])
            misses.append(stats[1])
            false_alarms.append(stats[2])
            model_labels.append(f'{name} ({t} mm)')
            colors.append(color_map[name])
            markers.append(marker_dict[t])

    plot_performance_diagram_pretty(
        hits, misses, false_alarms,
        model_labels=model_labels,
        colors=colors,
        markers=markers
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mle", required=True)
    parser.add_argument("--era", required=True)
    args = parser.parse_args()

    main(args.mle, args.era)