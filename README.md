
# Depth Estimation on Offshore Oil Platforms – LAGIRS 2025

This repository contains code for the evaluation and visualization of monocular depth estimation methods applied to 360-degree images of offshore oil platforms. The study is pending approval at ISPRS LAGIRS 2025.


## 🔍 Overview

We evaluate four state-of-the-art monocular metric depth estimation models in a novel industrial setting:

- **Depth Anything V2**
- **ZoeDepth**
- **Metric3Dv2**
- **Patchfusion**

The target application is structural inspection of offshore oil platforms using 360-degree panoramic images converted to cubemaps.

---

## 1. 📂 Dataset

⚠️ **Note:** The dataset is private and not publicly available.

It consists of 500 photospheres (360-degree images), each transformed into 6 cubemap faces, resulting in 3,000 RGB images and corresponding LiDAR-based depth maps.

---

## 2. 📸 Depth Estimation

Run depth estimation using:

```bash
python calculate_depth_all_models.py \
    --path_input /mnt/storage/jorge/depth_paper/dataset/images/PA/RGB_faces \
    --path_output /mnt/storage/jorge/depth_paper/dataset/images/PA/DAv2_faces \
    --model_id DAv2
```

### Arguments:
- `--path_input`: Folder with RGB input images.
- `--path_output`: Folder to save predicted depth maps.
- `--model_id`: One of `["DAv2", "Metric3D", "ZoeDepth", "Patchfusion", "ZoeDepthHuggingface"]`
- `--get_depth_prediction_thresholded`: Whether to threshold outputs (bool).
- `--save_npy_flag`: Save output as `.npy` (bool).
- `--pretrain_mode`: One of `["indoor", "outdoor", "combined"]`.
- `--patchfusion_mode`: `"DA"` or `"ZoeDepth"` for Patchfusion variants.

---

## 3. 📏 Evaluate Metrics

Evaluate model predictions and save histogram data:

```bash
python get_metrics.py \
    --reference_path /mnt/storage/jorge/depth_paper/dataset/images/PA/Depth_faces_decoded \
    --predicted_path /mnt/storage/jorge/depth_paper/dataset/images/PA/Metric3D_faces/results_Metric3D/faces_depth \
    --csv_path /mnt/storage/jorge/depth_paper/dataset/selected_photospheres_PA.csv \
    --method Metric3D --platform PA
```

### Arguments:
- `--method`: One of `["DAv2", "DAv2_Outdoor", "ZoeDepthNYU", "ZoeDepthKitti", "ZoeDepthNYUKitti", "Metric3D", "PatchfusionZoeDepth", "PatchfusionDA"]`
- `--platform`: Platform ID, `"PA"` or `"PB"`
- `--reference_path`: Path to LiDAR-based depth maps.
- `--predicted_path`: Path to predicted depth maps.
- `--csv_path`: CSV file listing selected photospheres.
- `--load_histogram_data`: Load precomputed histograms (optional flag).

---

## 4. 📊 Plot Histogram Figures (Fig. 2 & 3)

Visualize per-bin accuracy for all models:

```bash
python plot_histogram_multiple_models.py
```

This loads files in the format `histogram_data_before_{method}_{platform}.pkl` and produces overlaid plots of predicted vs. reference depth per bin.

---

## 5. 🎨 Visualize Qualitative Results (Fig. 4)

Generate qualitative comparisons between models:

```bash
python get_visual_results.py
```

This script visualizes a fixed set of depth prediction samples (`depth_filename0.npy`, `depth_filename1.npy`, `depth_filename3.npy`) for each method.


---

## 🛠 Dependencies

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- tqdm
- scikit-learn
- transformers

(Use the provided `requirements.txt` or `environment.yml`)
