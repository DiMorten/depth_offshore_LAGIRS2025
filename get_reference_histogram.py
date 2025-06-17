import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, glob, tqdm
import psutil

def get_ram_usage():
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / 1e9:.2f} GB")
    print(f"Used: {mem.used / 1e9:.2f} GB")
    print(f"Available: {mem.available / 1e9:.2f} GB")
    print(f"Percent Used: {mem.percent}%")

def plot_dual_sample_counts_by_bin(reference1, reference2, ymax=None, max_bin_value=35,
    output_path="sample_distribution_by_bin.png", label1="PA", label2="PB"):
    """
    Plots a grouped bar chart of number of samples per reference value bin
    for two datasets, formatted for single-column width in research papers.

    Parameters:
    - reference1: array-like of ground truth values for dataset 1
    - reference2: array-like of ground truth values for dataset 2
    - ymax: maximum y-axis value (optional)
    - max_bin_value: maximum reference value to bin
    - output_path: file path to save the figure
    - label1, label2: labels for the two datasets
    """
    reference1 = np.array(reference1)
    reference2 = np.array(reference2)

    bins = np.arange(0, max_bin_value + 1, 1)
    bin_indices1 = np.digitize(reference1, bins) - 1
    bin_indices2 = np.digitize(reference2, bins) - 1

    counts1 = [np.sum(bin_indices1 == i) for i in range(len(bins) - 1)]
    counts2 = [np.sum(bin_indices2 == i) for i in range(len(bins) - 1)]
    bin_labels = [f"[{bins[i]},{bins[i+1]})" for i in range(len(bins) - 1)]

    # Filter bins with at least one non-zero count
    filtered_indices = [i for i in range(len(bin_labels)) if counts1[i] > 0 or counts2[i] > 0]
    counts1 = [counts1[i] for i in filtered_indices]
    counts2 = [counts2[i] for i in filtered_indices]
    bin_labels = [bin_labels[i] for i in filtered_indices]

    x = np.arange(len(bin_labels))
    width = 0.35

    # Plot
    plt.figure(figsize=(3.4, 2.4))  # Single-column size (width x height in inches)
    plt.bar(x - width/2, counts1, width=width, label=label1)
    plt.bar(x + width/2, counts2, width=width, label=label2)

    plt.xticks(x, bin_labels, rotation=45, fontsize=6)
    plt.yticks(fontsize=7)
    plt.xlabel("Reference (meters)", fontsize=8)
    plt.ylabel("Pixel Count", fontsize=8)
    # plt.title("Sample Distribution", fontsize=9)
    plt.grid(axis='y', linewidth=0.3)

    if ymax:
        plt.ylim(0, ymax)
    plt.legend(fontsize=7, loc='upper right', frameon=False)
    plt.tight_layout(pad=0.5)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Dual sample count bar plot saved to {output_path}")


def get_dataset(reference_path="/mnt/storage/jorge/depth_paper/dataset/images/PA/Depth_faces_decoded",
                # reference_path2="/mnt/storage/jorge/depth_paper/dataset/images/PB/Depth_faces_decoded",
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
        raise RuntimeError("No matching reference files found based on the CSV 1.")

    print(f"Selected {len(selected_files)} reference files based on CSV match.")

    reference_values = np.zeros((len(selected_files), 1344, 1344), np.float32)

    # filenames = []
    for i, filename in enumerate(tqdm.tqdm(selected_files)): # [:20]
        # print("i", i)
        if i % 150 == 0:
            print("i", i)
            get_ram_usage()

        # filenames.append(filename)
        reference_values[i] = np.load(filename).astype(np.float32)


    print("Finished loading files")
    get_ram_usage()

    reference_values = reference_values[reference_values > reference_threshold]

    print("Finished filtering invalid reference samples")
    get_ram_usage()



    return reference_values

reference_values1 = get_dataset(
    "/mnt/storage/jorge/depth_paper/dataset/images/PA/Depth_faces_decoded",
    csv_path="/mnt/storage/jorge/depth_paper/dataset/selected_photospheres_PA.csv")
reference_values2 = get_dataset(
    "/mnt/storage/jorge/depth_paper/dataset/images/PB/Depth_faces_decoded",
    csv_path="/mnt/storage/jorge/depth_paper/dataset/selected_photospheres_PB.csv")

plot_dual_sample_counts_by_bin(reference_values1, reference_values2, ymax=None, max_bin_value=25,
    output_path="sample_distribution_by_bin.png", label1="PA", label2="PB")