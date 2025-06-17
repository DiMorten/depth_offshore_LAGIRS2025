
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import tqdm
import cv2
import glob, os
import sys
from icecream import ic
import tqdm
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse
import psutil
import pickle
import sys
parser = argparse.ArgumentParser()


def plot_overlaid_dual_histogram(histogram_data_before, histogram_data_after, lims,
    output_path='overlaid_dual_histogram.png'):
    # Unpack
    (centers1, means1, std1_lows1, std1_highs1,
     std2_lows1, std2_highs1, _) = histogram_data_before

    (centers2, means2, std1_lows2, std1_highs2,
     std2_lows2, std2_highs2, _) = histogram_data_after

    # plt.figure(figsize=(12, 6))
    plt.figure(figsize=(6, 4))  # Width: 6 inches, Height: 4 inches

    # Plot BEFORE
    color_before = 'orange'
    color_after = 'blue'
    plt.plot(centers1, means1, marker='o', linestyle='-', label='Mean (Before)', color=color_before)
    plt.fill_between(centers1, std2_lows1, std2_highs1, color=color_before, alpha=0.15, label='±2×Std (Before)')
    plt.fill_between(centers1, std1_lows1, std1_highs1, color=color_before, alpha=0.3, label='±1×Std (Before)')
    plt.tight_layout()
    # plt.plot(centers1, std1_lows1, color=color_before, linestyle='--', linewidth=0.8)
    # plt.plot(centers1, std1_highs1, color=color_before, linestyle='--', linewidth=0.8)

    # Plot AFTER
    plt.plot(centers2, means2, marker='s', linestyle='-', label='Mean (After)', color=color_after)
    plt.fill_between(centers2, std2_lows2, std2_highs2, color=color_after, alpha=0.15, label='±2×Std (After)')
    plt.fill_between(centers2, std1_lows2, std1_highs2, color=color_after, alpha=0.3, label='±1×Std (After)')
    # plt.plot(centers2, std1_lows2, color=color_after, linestyle='--', linewidth=0.8)
    # plt.plot(centers2, std1_highs2, color=color_after, linestyle='--', linewidth=0.8)

    plt.xlabel("Predicted Value Bin Center")
    plt.ylabel("Reference Value")
    plt.title("Mean and Std Bands (Before vs After Regression)")
    # plt.grid(True, axis='x')
    plt.legend(loc='upper left', fontsize=8)
    '''
    lims = { # DAv2 indoor
        "xlim": [0, 20],
        "ylim": [-10, 40],
        "yticks": np.arange(-10, 40, 2.5)
    }
    '''
    
    
    plt.xlim(lims["xlim"])
    plt.ylim(lims["ylim"])
    plt.yticks(lims["yticks"])  # From -10 to 40 in steps of 5
    plt.grid(True, axis='x')  # Vertical grid lines at x-ticks
    plt.grid(True, axis='y')  # Optional: Horizontal grid lines too
    plt.tight_layout()

    # plt.xlim([0, 35])
    # plt.ylim([-1, 25])

    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Overlaid histogram plot saved to {output_path}")

import matplotlib.pyplot as plt
import numpy as np

def plot_multi_histograms(histogram_data_list, model_labels, lims,
                          output_path='overlaid_multi_histogram.png'):
    """
    Plot overlaid mean and std bands for multiple models.
    
    Args:
        histogram_data_list: List of tuples. Each tuple contains 7 elements:
            (centers, means, std1_lows, std1_highs, std2_lows, std2_highs, _) for each model.
        model_labels: List of model names (e.g., ['DAv2', 'ZoeDepth', ...])
        lims: Dict with keys: 'xlim', 'ylim', 'yticks'
        output_path: Where to save the figure
    """
    assert len(histogram_data_list) == len(model_labels), "Mismatch between data and labels"

    colors = ['orange', 'blue', 'green', 'purple']
    # plt.figure(figsize=(6, 4))
    plt.figure(figsize=(6, 6))

    for idx, (data, label) in enumerate(zip(histogram_data_list, model_labels)):
        centers, means, std1_lows, std1_highs, std2_lows, std2_highs, _ = data
        color = colors[idx % len(colors)]
        
        # Plot mean line and shaded std bands
        plt.plot(centers, means, marker='o', linestyle='-', color=color, label=f'Mean ({label})')
        plt.fill_between(centers, std2_lows, std2_highs, color=color, alpha=0.15, label=f'±2×Std ({label})')  # ±2 Std
        plt.fill_between(centers, std1_lows, std1_highs, color=color, alpha=0.3, label=f'±1×Std ({label})')

    # Diagonal y = x line (gray, dashed)
    min_val = max(lims["xlim"][0], lims["ylim"][0])
    max_val = min(lims["xlim"][1], lims["ylim"][1])
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1)#, label='y = x')

    # Axes and labels
    plt.xlabel("Predicted Value Bin Center (meters)")
    plt.ylabel("Reference Value (meters)")
    plt.title("Mean and Std Bands by Model")
    plt.xlim(lims["xlim"])
    plt.ylim(lims["ylim"])
    # plt.yticks(lims["yticks"])
    plt.xticks(lims["xticks"])

    plt.grid(True, axis='both')
    # plt.legend(loc='upper left', fontsize=8)
    plt.legend(loc='upper left', fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Multi-model histogram plot saved to {output_path}")

platform = "PA"


if platform == "PA":
    lims = { # Metric3D
            # "xlim": [0, 50],
            # "xticks": np.arange(0, 51, 5),

            "xlim": [0, 40],
            "xticks": np.arange(0, 40, 5),

            "ylim": [-10, 65],
            "yticks": np.arange(-10, 65, 5),
        }
else:
    lims = { # Metric3D
            "xlim": [0, 40],
            "xticks": np.arange(0, 40, 5),

            "ylim": [-1, 23],
            "yticks": np.arange(-5, 23, 5),
        }
    
# Load from file
method = "DAv2"
    
with open(f"histogram_data_before_{method}_{platform}.pkl", "rb") as f:
    histogram_data_dav2 = pickle.load(f)

method = "PatchfusionDA"

with open(f"histogram_data_before_{method}_{platform}.pkl", "rb") as f:
    histogram_data_patchfusion = pickle.load(f)

method = "ZoeDepthNYU"

with open(f"histogram_data_before_{method}_{platform}.pkl", "rb") as f:
    histogram_data_zoe = pickle.load(f)

method = "Metric3D"

with open(f"histogram_data_before_{method}_{platform}.pkl", "rb") as f:
    histogram_data_metric3d = pickle.load(f)
# sys.exit()
print("Here")

model_labels = ['DAv2', 'ZoeDepth', 'Metric3Dv2', 'Patchfusion']
histogram_data_list = [
    histogram_data_dav2,
    histogram_data_zoe,
    histogram_data_metric3d,
    histogram_data_patchfusion
]

plot_multi_histograms(histogram_data_list, model_labels, lims,
    output_path=f'overlaid_multi_histogram_{platform}.png')