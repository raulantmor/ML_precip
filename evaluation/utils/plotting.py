import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def blue_green_cmap(num_colors=20, brightness_factor=1.5, alpha_value=0.4):
    ''' makes custom cmap for plotting precip cases.'''
    
    original_cmap = plt.get_cmap('PuBuGn')
    colors = original_cmap(np.linspace(0, 1, num_colors - 2))
    green_colors = [mcolors.to_rgba("green")[:-1], mcolors.to_rgba("green", alpha=0.6)[:-1]]
    modified_green_colors = [[min(1.0, c * brightness_factor) for c in color] for color in green_colors]
    colors = np.vstack([colors, [*modified_green_colors[0], 1.0]]) 
    colors = np.vstack([colors, [*modified_green_colors[1], alpha_value]])
    
    return mcolors.ListedColormap(colors)