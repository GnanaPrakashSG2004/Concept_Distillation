#!/usr/bin/env python3
"""
Compute the Layerwise Mean-Max Concept Similarity (MMCS) from pickle files.

Folder structure:
  BASE_FOLDER/
    class0/
       <multiple .pkl files>  # each filename: "model1layer-model2layer.pkl"
    class1/
       ...
    ...
    class99/

Each .pkl file is expected to be a dictionary with keys:
  - "metadata": { "num_concepts_i": k, "num_concepts_j": k, ... }
  - "1to2": { (i, j): PearsonRResult(statistic=..., pvalue=...) , ... }

For each file, we compute:
  mcs1_file = (1/k) * sum_{i=1}^{k} ( max_{j} R[i,j] )
  mcs2_file = (1/k) * sum_{j=1}^{k} ( max_{i} R[i,j] )
Then, for each layer pair across classes (assume c classes for that pair):
  MMCS1 = (sum over classes of (mcs1_file)) / (c)
  MMCS2 = (sum over classes of (mcs2_file)) / (c)
And final MMCS = (MMCS1 + MMCS2) / 2

This script also checks that the number of concepts (k) is consistent across all files.
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_layer_names(filename):
    """
    Parse a filename of the form "layerX-layerY.pkl" and return (layer1, layer2).
    """
    base = filename.replace(".pkl", "")
    parts = base.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Filename '{filename}' does not match the expected pattern.")
    return parts[0], parts[1]


def load_correlation_matrix(pkl_path):
    """
    Load the pickle file and return the correlation matrix R (a numpy array)
    and the number of concepts (k). It checks that num_concepts_i equals num_concepts_j.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    meta = data.get("metadata", {})
    k_i = meta.get("num_concepts_i")
    k_j = meta.get("num_concepts_j")
    if k_i is None or k_j is None:
        raise ValueError(f"Metadata missing in file {pkl_path}")
    if k_i != k_j:
        raise ValueError(
            f"Number of concepts mismatch in file {pkl_path}: {k_i} vs {k_j}"
        )

    k = k_i
    assert k == k_j, f"Number of concepts mismatch in file {pkl_path}: {k_i} vs {k_j}"

    R = np.zeros((k, k), dtype=np.float32)
    # Fill in the matrix using the (i,j) keys from "1to2"
    for (i, j), result in data["1to2"].items():
        if i < k and j < k:
            R[i, j] = result.statistic
            if np.isnan(R[i, j]):
                raise ValueError(
                    f"NaN value found in correlation matrix for indices ({i}, {j}) in file {pkl_path}"
                )
        else:
            raise IndexError(f"Invalid indices ({i}, {j}) in file {pkl_path}")
    return R, k


def compute_mcs_from_matrix(R):
    """
    Given a k x k correlation matrix R, compute:
      mcs1 = average_{i} (max_j R[i,j])
      mcs2 = average_{j} (max_i R[i,j])
    Return (mcs1, mcs2).
    """
    row_max = np.max(R, axis=1)  # max over columns for each row
    col_max = np.max(R, axis=0)  # max over rows for each column
    mcs1 = np.mean(row_max)
    mcs2 = np.mean(col_max)
    return mcs1, mcs2


def aggregate_mmcs(base_folder, class_list):
    """
    Walk through each class folder under base_folder based on the class_list,
    load all pickle files, compute per-file mcs1 and mcs2,
    and aggregate by layer pair across classes.

    Returns:
      - mmcs_dict: dictionary mapping (layer1, layer2) -> final MMCS value (float)
      - unique_model1_layers: sorted list of unique layer names from model1
      - unique_model2_layers: sorted list of unique layer names from model2
    Also checks that the number of concepts is consistent across all files.
    """

    layer_pair_to_mcs1 = defaultdict(list)
    layer_pair_to_mcs2 = defaultdict(list)

    global_k = None

    class_folders = []
    for cls_idx in class_list:
        cls_folder = os.path.join(base_folder, str(cls_idx))
        if os.path.isdir(cls_folder):
            class_folders.append(cls_folder)
        else:
            print(f"Warning: Class folder {cls_folder} does not exist. Skipping.")
    class_folders.sort(key=lambda x: int(x) if x.isdigit() else x)

    for cls_idx in class_folders:
        cls_path = os.path.join(base_folder, cls_idx)
        # Get all pickle files in this class folder
        pkl_files = glob.glob(os.path.join(cls_path, "*.pkl"))
        for pkl_file in pkl_files:
            filename = os.path.basename(pkl_file)
            try:
                layer1, layer2 = parse_layer_names(filename)
            except ValueError as ve:
                print(f"Skipping file {filename}: {ve}")
                continue

            R, k = load_correlation_matrix(pkl_file)

            if global_k is None:
                global_k = k

            elif global_k != k:
                raise ValueError(
                    f"Inconsistent number of concepts in file {pkl_file}. "
                    f"Expected {global_k} but got {k}."
                )

            # Computing mcs values for this file
            mcs1, mcs2 = compute_mcs_from_matrix(R)
            layer_pair_to_mcs1[(layer1, layer2)].append(mcs1)
            layer_pair_to_mcs2[(layer1, layer2)].append(mcs2)

    # Now aggregating over classes for each layer pair:
    mmcs_dict = {}
    for key in layer_pair_to_mcs1:
        mcs1_vals = layer_pair_to_mcs1[key]
        mcs2_vals = layer_pair_to_mcs2[key]
        # Averaging over classes:
        avg_mcs1 = np.mean(mcs1_vals)
        avg_mcs2 = np.mean(mcs2_vals)
        mmcs = (avg_mcs1 + avg_mcs2) / 2.0
        mmcs_dict[key] = mmcs

    unique_model1_layers = sorted({key[0] for key in mmcs_dict.keys()})
    unique_model2_layers = sorted({key[1] for key in mmcs_dict.keys()})

    return mmcs_dict, unique_model1_layers, unique_model2_layers, global_k


def build_mmcs_matrix(mmcs_dict, model1_layers, model2_layers):
    """
    Given the dictionary mapping (layer1, layer2) -> MMCS and the sorted lists
    of layer names for model1 (rows) and model2 (columns), build a 2D numpy array.
    """
    M = np.full((len(model1_layers), len(model2_layers)), np.nan, dtype=np.float32)
    for (l1, l2), mmcs_val in mmcs_dict.items():
        try:
            i = model1_layers.index(l1)
            j = model2_layers.index(l2)
            M[i, j] = mmcs_val
        except ValueError:
            # Should not happen if our layer lists are correct.
            raise ValueError(f"Layer not found in model1 or model2: {l1} or {l2}")

    return M


def plot_mmcs_heatmap(M, model1_layers, model2_layers, plot_file_path, title="Layerwise MMCS Heatmap"):
    """
    Plot the 2D MMCS heatmap with model1 layers on the y-axis and model2 layers on the x-axis.
    """
    plt.figure(figsize=(20, 8))
    im = plt.imshow(M, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, label="MMCS")
    plt.xticks(ticks=np.arange(len(model2_layers)), labels=model2_layers, rotation=90)
    plt.yticks(ticks=np.arange(len(model1_layers)), labels=model1_layers)
    plt.xlabel("ResNet-50 Layers (Model 2)")
    plt.ylabel("ResNet-18 Layers (Model 1)")
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_file_path, dpi=300)


def main(base_folder, class_list, plot_file_path):
    """
    Main driver:
      - Aggregates MMCS values across all class subfolders.
      - Checks consistency of concept counts.
      - Builds a 2D matrix and plots the heatmap.
    """
    print("Aggregating MMCS values from pickle files...")
    mmcs_dict, model1_layers, model2_layers, k = aggregate_mmcs(base_folder, class_list)
    print(f"Consistent number of concepts (k): {k}")
    print(
        f"Found {len(model1_layers)} unique model1 layers and {len(model2_layers)} unique model2 layers."
    )

    M = build_mmcs_matrix(mmcs_dict, model1_layers, model2_layers)
    plot_mmcs_heatmap(M, model1_layers, model2_layers, plot_file_path)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a heatmap of Layerwise Mean-Max Concept Similarity (MMCS) from pickle files."
    )
    parser.add_argument(
        "base_folder",
        type=str,
        help="Base folder containing class subfolders with .pkl files.",
    )
    parser.add_argument(
        "--start_class_idx",
        type=int,
        default=0,
        help="Starting index for class folders (default: 0).",
    )
    parser.add_argument(
        "--end_class_idx",
        type=int,
        default=1000,
        help="Ending index for class folders (default: 1000).",
    )
    parser.add_argument(
        "--plot_file_path",
        type=str,
        default="mmcs_heatmap.png",
        help="Output file path for saving the heatmap (default: mmcs_heatmap.png).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_folder = args.base_folder
    start_class_idx = args.start_class_idx
    end_class_idx = args.end_class_idx

    # Ensure the base folder exists
    if not os.path.exists(base_folder):
        print(f"Base folder '{base_folder}' does not exist.")
        sys.exit(1)

    class_list = range(start_class_idx, end_class_idx)

    main(base_folder, class_list, args.plot_file_path)
