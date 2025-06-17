import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from icecream import ic

def load_data(platform, image_name):
    base_path = f"/mnt/storage/jorge/depth_paper/dataset/images/{platform}"
    reference_path = os.path.join(base_path, "Depth_faces_decoded", image_name)
    rgb_path = os.path.join(base_path, "RGB_faces", image_name.replace('.npy', '.png'))
    image_name_npz = image_name.replace('.npy', '.npz')

    reference = np.load(reference_path)
    rgb = imread(rgb_path)

    ic("Reference")
    ic(np.min(reference), np.mean(reference), np.max(reference))

    predicted_path = {
        "DAv2": os.path.join(base_path, "DAv2_faces/results_DAv2/faces_depth"),
        "ZoeDepth": os.path.join(base_path, "ZoeDepthNYU_faces/results_ZoeDepthHuggingface/faces_depth"),
        "Metric3Dv2": os.path.join(base_path, "Metric3D_faces/results_Metric3D/faces_depth"),
        "Patchfusion": os.path.join(base_path, "Patchfusion_DA_faces/results_Patchfusion/faces_depth")
    }

    predicted = {}
    for name, path in predicted_path.items():
        arr = np.load(os.path.join(path, image_name_npz))['arr_0']
        if name == "DAv2":
            arr *= 0.5
        arr[arr > 20] = 20
        arr[reference < 0.1] = 0
        predicted[name] = arr
        ic(name)
        ic(np.min(arr), np.mean(arr), np.max(arr))

    return rgb, reference, predicted

# Define two samples (platform, image name)
samples = [
    ("PA", "depth_filename0.npy"),
    ("PB", "depth_filename1.npy"),
    ("PB", "depth_filename2.npy"),
]

# cubemap_36a731575aedb260ae0a8f1464114645_posz

titles = ["RGB", "Reference", "DAv2", "ZoeDepth", "Metric3Dv2", "Patchfusion"]
cmap = 'magma'
vmin, vmax = 0, 20  # depth range in meters

# Create figure: 2 rows, 6 columns (RGB + reference + 4 models)
# fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(8.2, 2.6))  # wider to fit one more column
# fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(8.2, 5.5))  # wider to fit one more column
fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(8.2, 3.9))  # wider to fit one more column

# axs = axs.reshape(4, 6)
axs = axs.reshape(3, 6)
# axs = axs.reshape(2, 6)

# Plot each sample
for row_idx, (platform, image_name) in enumerate(samples):
    rgb, reference, predicted = load_data(platform, image_name)

    # Plot RGB image
    axs[row_idx, 0].imshow(rgb)
    axs[row_idx, 0].set_title("RGB" if row_idx == 0 else "")
    axs[row_idx, 0].axis('off')

    # Plot reference
    img = axs[row_idx, 1].imshow(reference, cmap=cmap, vmin=vmin, vmax=vmax)
    if row_idx == 0:
        mappable = img  # capture for colorbar

    axs[row_idx, 1].set_title("Reference" if row_idx == 0 else "")
    axs[row_idx, 1].axis('off')

    # Plot predicted depth maps
    for col_idx, key in enumerate(["DAv2", "ZoeDepth", "Metric3Dv2", "Patchfusion"], start=2):
        axs[row_idx, col_idx].imshow(predicted[key], cmap=cmap, vmin=vmin, vmax=vmax)
        axs[row_idx, col_idx].set_title(key if row_idx == 0 else "")
        axs[row_idx, col_idx].axis('off')

# Add vertical colorbar on the right
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])  # adjust to the new width
fig.colorbar(mappable, cax=cbar_ax, orientation='vertical', label='Depth (meters)')

# Adjust spacing
plt.subplots_adjust(left=0.03, right=0.93, top=0.95, bottom=0.05, wspace=0.05, hspace=0.1)

# Save and show
plt.savefig("depth_predictions_with_rgb_threesamples.png", dpi=300, bbox_inches='tight')
plt.show()