import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, r2_score
import tqdm
import cv2
import glob, os
import sys
from icecream import ic
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

parser.add_argument('--method', type=str, default="DAv2", 
    choices=["DAv2", "DAv2_Outdoor", "ZoeDepthNYU", "ZoeDepthKitti", "ZoeDepthNYUKitti", 
        "Metric3D", "PatchfusionZoeDepth", "PatchfusionDA"])
parser.add_argument('--platform', type=str, default="PA", 
    choices=["PA", "PB"])
parser.add_argument('--reference_path', type=str, 
    default="/mnt/storage/jorge/depth_paper/dataset/images/PA/Depth_faces_decoded")
parser.add_argument('--predicted_path', type=str, 
    default="/mnt/storage/jorge/depth_paper/dataset/images/PA/DAv2_faces/results_DAv2/faces_depth")
parser.add_argument('--csv_path', type=str, 
    default="/mnt/storage/jorge/depth_paper/dataset/selected_photospheres.csv")
parser.add_argument('--load_histogram_data', 
    default=False, action='store_true')

args = parser.parse_args()




print(args)


def get_metrics2(reference, predicted):
    # Compute metrics
    mse = mean_squared_error(reference, predicted)
    mae = mean_absolute_error(reference, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(reference, predicted)

    print("mse, mae, rmse, r2", round(mse, 2), round(mae, 2), round(rmse, 2), round(r2, 4))
def get_ram_usage():
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Percent Used: {mem.percent}%")
        

def get_dataset(reference_path="/mnt/storage/jorge/uff_depth_downloaded/depth_decoded",
                predicted_path="/mnt/storage/jorge/uff_depth_downloaded/results_da2",
                method="DAv2",
                flatten=True,
                reference_threshold=0.1,
                csv_path="/mnt/storage/jorge/depth_paper/dataset/selected_photospheres.csv"):
    print("Getting data...")
    # Step 1: Load selected IDs from CSV
    df = pd.read_csv(csv_path)
    selected_ids = set([os.path.splitext(f)[0] for f in df['filename']])  # strip ".jpg"

    reference_files = glob.glob(os.path.join(reference_path, "*.npy"))

    # Step 2: Filter reference files by matching ID
    def matches_selected_id(file_path):
        base = os.path.basename(file_path)
        return any(id_ in base for id_ in selected_ids)

    selected_files = [f for f in reference_files if matches_selected_id(f)]

    if len(selected_files) == 0:
        raise RuntimeError("No matching reference files found based on the CSV.")

    print(f"Selected {len(selected_files)} reference files based on CSV match.")

    predicted_values = np.zeros((len(selected_files), 1344, 1344), np.float32)
    reference_values = np.zeros((len(selected_files), 1344, 1344), np.float32)

    # filenames = []
    for i, filename in enumerate(tqdm.tqdm(selected_files)):
        # print("i", i)
        if i % 150 == 0:
            print("i", i)
            get_ram_usage()

        # filenames.append(filename)
        reference_values[i] = np.load(filename).astype(np.float32)

        filename_predicted = os.path.basename(filename).replace('.npy', '.npz')
        predicted_file = os.path.join(predicted_path, filename_predicted)

        try:
            predicted_values[i] = np.load(predicted_file)['arr_0'].astype(np.float32)
        except Exception as e:
            print(f"Error loading prediction {predicted_file}: {e}")
            raise


    print("Finished loading files")
    get_ram_usage()

    predicted_values = predicted_values[reference_values > reference_threshold]
    reference_values = reference_values[reference_values > reference_threshold]

    print("Finished filtering invalid reference samples")
    get_ram_usage()

    if flatten:
        print("Started flattening files...")
        reference_values = reference_values.flatten()
        predicted_values = predicted_values.flatten()
        ic(reference_values.shape)
        ic(predicted_values.shape)

    print("Finished flattening files")

    return reference_values, predicted_values

def plot_reference_mean_std_dual_band(centers, means, std1_lows, std1_highs,
        std2_lows, std2_highs, output_path):
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(centers, means, marker='o', label='Mean Reference', color='black')
    plt.fill_between(centers, std2_lows, std2_highs, color='orange', alpha=0.25, label='±2×Std')
    plt.fill_between(centers, std1_lows, std1_highs, color='blue', alpha=0.4, label='±1×Std')

    plt.xlabel("Predicted Value Bin Center")
    plt.ylabel("Reference Value")
    plt.title("Reference Mean with ±1×Std and ±2×Std Bands by Predicted Value")
    plt.grid(True, axis='y')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.xlim(bins[0] - 1, bins[-1] + 1)
    # plt.xlim([0,20])
    # plt.ylim([-4,20])
    plt.xlim([0,35])
    plt.ylim([-1,25])

    plt.savefig(output_path)
    plt.close()
    print(f"Dual std band plot saved to {output_path}")

def get_reference_mean_std_dual_band(reference, predicted, max_bin_value=35, output_path="reference_std_bands.png"):
    """
    Plots the mean of reference values per predicted value bin, with two spread bands:
    one for ±1×std and another for ±2×std.
    
    Parameters:
    - reference: array-like ground truth values
    - predicted: array-like predicted values
    - max_bin_value: max predicted value to bin
    - output_path: path to save the plot
    """
    # reference = np.array(reference)
    # predicted = np.array(predicted)
    assert reference.shape == predicted.shape, "reference and predicted must have the same shape"

    bins = np.arange(0, max_bin_value + 1, 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(predicted, bins) - 1

    means = []
    std1_lows = []
    std1_highs = []
    std2_lows = []
    std2_highs = []
    centers = []

    for i in range(len(bins) - 1):
        idx = np.where(bin_indices == i)[0]
        if len(idx) == 0:
            continue

        refs = reference[idx]
        mean = np.mean(refs)
        # print("Getting errors...")
        # errors = np.abs(reference[idx] - predicted[idx])
        # mean = np.mean(errors)

        std = np.std(refs, ddof=1)
        # std = np.std(errors, ddof=1)

        means.append(mean)
        std1_lows.append(mean - std)
        std1_highs.append(mean + std)
        std2_lows.append(mean - 2 * std)
        std2_highs.append(mean + 2 * std)
        centers.append(bin_centers[i])

    plot_reference_mean_std_dual_band(centers, means, std1_lows, std1_highs,
        std2_lows, std2_highs, output_path)

    return centers, means, std1_lows, std1_highs, std2_lows, std2_highs, output_path

class MonocularDepthMetrics:
    def __init__(self, predictions, ground_truth):
        print("Init MonocularDepthMetrics")
        self.predictions = predictions
        self.ground_truth = ground_truth

    def _validate_inputs(self):
        if not (self.predictions.shape == self.ground_truth.shape):
            raise ValueError("Predictions, ground truth, and mask arrays must have the same shape")

    def _apply_mask(self, array):
        return array# [self.mask]

    # def r2(self):
    #     return r2_score(self.ground_truth, self.predictions)

    def mae(self):
        return np.mean(np.abs(self.predictions - self.ground_truth))

    def rmse(self):
        return np.sqrt(np.mean((self.predictions - self.ground_truth) ** 2))

    def scale_invariant_rmse(self):
        log_diff = np.log(self.predictions + 1e-8) - np.log(self.ground_truth + 1e-8)
        return np.sqrt(np.mean(log_diff ** 2))

    def threshold_accuracy(self, threshold=1.25):
        ratio = np.maximum(self.predictions / self.ground_truth, 
                           self.ground_truth / self.predictions)
        return np.mean(ratio < threshold)

    def log_rmse(self):
        return np.sqrt(np.mean((np.log(self.predictions + 1e-8) - np.log(self.ground_truth + 1e-8)) ** 2))

    def mean_relative_error(self):
        return np.mean(np.abs(self.predictions - self.ground_truth) / self.ground_truth)

    def mean_squared_log_error(self):
        return np.mean((np.log1p(self.predictions) - np.log1p(self.ground_truth)) ** 2)

    # def structural_similarity(self):
    #     return ssim(self.ground_truth, self.predictions, 
    #                 data_range=self.predictions.max() - self.predictions.min())

    def edge_aware_loss(self):
        grad_pred = np.gradient(self.predictions)
        grad_truth = np.gradient(self.ground_truth)
        return np.mean(np.abs(self._apply_mask(grad_pred[0]) - self._apply_mask(grad_truth[0])) + 
                       np.abs(self._apply_mask(grad_pred[1]) - self._apply_mask(grad_truth[1])))

    def compute_all_metrics(self):
        result =  {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "Scale-Invariant RMSE": self.scale_invariant_rmse(),
            "Threshold Accuracy (δ=1.25)": self.threshold_accuracy(1.25),
            "Threshold Accuracy (δ=1.25^2)": self.threshold_accuracy(1.25 ** 2),
            "Threshold Accuracy (δ=1.25^3)": self.threshold_accuracy(1.25 ** 3),
            "Log RMSE": self.log_rmse(),
            "Mean Relative Error": self.mean_relative_error(),
            "Mean Squared Log Error": self.mean_squared_log_error(),
            # "Structural Similarity (SSIM)": self.structural_similarity(),
            "Edge-Aware Loss": self.edge_aware_loss(),
        }

        for k, v in result.items():
            result[k] = round(v, 2)
        print("result", result)
        return result

method = args.method

print("================================== Loading dataset...")

reference_values, predicted_values = get_dataset(predicted_path=args.predicted_path,  # predicted_path[method]
    method=method, flatten=False, reference_path=args.reference_path,
    csv_path=args.csv_path)

reference_threshold = 0.1

print("reference_values.shape", reference_values.shape)
print("predicted_values.shape", predicted_values.shape)

ic(np.min(predicted_values), np.mean(predicted_values), np.max(predicted_values))
ic(np.min(reference_values), np.mean(reference_values), np.max(reference_values))

predicted_values = predicted_values[reference_values>reference_threshold]
reference_values = reference_values[reference_values>reference_threshold]

print("================================== Getting metrics")
if args.method == "DAv2":
    max_depth = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35]
    default_max_depth = 20
    multipliers = [i / default_max_depth for i in max_depth] # 0.5 for 10
    print("multipliers", multipliers)
    for multiplier in multipliers:
        print("multiplier", multiplier)
        get_metrics2(reference_values, predicted_values * multiplier)

        metrics = MonocularDepthMetrics(predicted_values, reference_values)
        metrics.compute_all_metrics()

elif args.method == "DAv2_Outdoor":
    max_depth = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 32.5, 35]
    default_max_depth = 80
    multipliers = [i / default_max_depth for i in max_depth] # 0.5 for 10
    print("multipliers", multipliers)
    for multiplier in multipliers:
        print("multiplier", multiplier)
        get_metrics2(reference_values, predicted_values * multiplier)

        metrics = MonocularDepthMetrics(predicted_values, reference_values)
        metrics.compute_all_metrics()
else:
    get_metrics2(reference_values, predicted_values)
    metrics = MonocularDepthMetrics(predicted_values, reference_values)
    metrics.compute_all_metrics()


print(f"================== Get histogram data and save to histogram_data_{method}_{args.platform}.pkl")

histogram_data = get_reference_mean_std_dual_band(reference_values, predicted_values*0.5, 
    max_bin_value=50, output_path=f"reference_dual_std_band_{method}_{args.platform}_before.png")
# Save to file
with open(f"histogram_data_{method}_{args.platform}.pkl", "wb") as f:
    pickle.dump(histogram_data, f)
sys.exit()

    