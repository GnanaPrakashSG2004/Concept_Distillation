import os
import pickle as pkl
from math import ceil

import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings

from scipy.stats import pearsonr, spearmanr
from scipy.stats._warnings_errors import ConstantInputWarning, NearConstantInputWarning

from src.dictionary_learning import DictionaryLearner


def load_concepts(concepts_folder, layer, class_idx):
    try:
        # print(os.path.join(concepts_folder, layer, f'{class_idx}.pkl'), os.path.exists(os.path.join(concepts_folder, layer, f'{class_idx}.pkl')))
        with open(os.path.join(concepts_folder, layer, f"{class_idx}.pkl"), "rb") as f:
            concepts = pkl.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Concepts for class {class_idx} not found")
        concepts = None
    return concepts


def compute_concept_coefficients(
    activations, concepts_w, method, device="cpu", params=None
):
    if params is None:
        params = {}
    params["device"] = device
    out = DictionaryLearner.static_transform(
        method, activations, concepts_w, params=params
    )
    # print(out['err'])
    return out["U"]


@ignore_warnings(category=UserWarning)
def _batch_inference(model, dataset, batch_size=128, resize=None, device="cuda"):
    """
    Code from CRAFT repository
    """
    nb_batchs = ceil(len(dataset) / batch_size)
    start_ids = [i * batch_size for i in range(nb_batchs)]

    results = []

    with torch.inference_mode():
        for i in start_ids:
            x = torch.tensor(dataset[i : i + batch_size])
            x = x.to(device)

            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode="bilinear", align_corners=False
                )

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results


class BatchDataset(torch.utils.data.Dataset):
    """
    Dataset class for batch inference
    """
    def __init__(self, images: np.ndarray):
        """
        Initialize the dataset with a list of images

        Args:
            images: List or numpy array of images
        """
        self.images = images


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx: int):
        return self.images[idx]


def _faster_batch_inference(
        model: torch.nn.Module,
        images: np.ndarray,
        batch_size: int = 128,
        num_workers: int = 8,
        resize: tuple = None,
        device: str = "cuda",
        pbar: tqdm = None,
    ) -> torch.Tensor:
    """
    Faster batch inference using DataLoader

    Args:
        model: The model to use for activation extraction
        images: List or numpy array of images
        batch_size: Batch size for inference
        num_workers: Number of workers for DataLoader
        resize: Tuple (h, w) for resizing images, or None
        device: Device to run inference on
        pbar: Optional tqdm progress bar

    Returns:
        torch.Tensor: Model outputs for all images in the same order
    """
    dataset = BatchDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    results = []
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if pbar is not None:
                pbar.set_postfix_str(f"Activations for batch {i+1}/{len(dataloader)}")

            batch = batch.to(device, non_blocking=True)

            if resize:
                batch = torch.nn.functional.interpolate(
                    batch, size=resize, mode="bilinear", align_corners=False
                )

            batch_feats = model(batch).cpu()
            if torch.isnan(batch_feats).any():
                raise ValueError("NaN detected in model activations")

            results.append(batch_feats)

    results = torch.cat(results)
    return results


@ignore_warnings(category=ConstantInputWarning)
def correlation_comparison(method, Ui, Uj, self_compare=False):
    """
    Compare two coefficient matrices Ui and Uj
    :param Ui: torch.Tensor
    :param Uj: torch.Tensor
    :return: List of dictionaries with each comparison method
    """
    if Ui is None or Uj is None:
        return None
    num_concepts_i = Ui.shape[1]
    num_concepts_j = Uj.shape[1]
    statistic_dict = {}
    statistic_dict1to2 = {}

    for i in range(num_concepts_i):
        for j in range(num_concepts_j):
            if method == "pearson":
                out1to2 = pearsonr(Ui[:, i], Uj[:, j])
            elif method == "spearman":
                out1to2 = spearmanr(Ui[:, i], Uj[:, j])
            else:
                raise ValueError(f"Unknown method: {method}")
            statistic_dict1to2[(i, j)] = out1to2

    statistic_dict["metadata"] = {
        "method": method,
        "num_concepts_i": num_concepts_i,
        "num_concepts_j": num_concepts_j,
    }
    statistic_dict["1to2"] = statistic_dict1to2

    if self_compare:
        statistic_dict1to1 = {}
        statistic_dict2to2 = {}
        for i in range(num_concepts_i):
            for j in range(num_concepts_j):
                if method == "pearson":
                    out1to1 = pearsonr(Ui[:, i], Ui[:, j])
                elif method == "spearman":
                    out1to1 = spearmanr(Ui[:, i], Ui[:, j])
                else:
                    raise ValueError(f"Unknown method: {method}")
                statistic_dict1to1[(i, j)] = out1to1

        for i in range(num_concepts_i):
            for j in range(num_concepts_j):
                if method == "pearson":
                    out2to2 = pearsonr(Uj[:, i], Uj[:, j])
                elif method == "spearman":
                    out2to2 = spearmanr(Uj[:, i], Uj[:, j])
                else:
                    raise ValueError(f"Unknown method: {method}")
                statistic_dict2to2[(i, j)] = out2to2

        statistic_dict["1to1"] = statistic_dict1to1
        statistic_dict["2to2"] = statistic_dict2to2

    return statistic_dict


def convert_to_correlation_comparison_to_array(statistic_dict, metadata):

    arr = np.zeros((metadata["num_concepts_i"], metadata["num_concepts_j"]))
    for i in range(metadata["num_concepts_i"]):
        for j in range(metadata["num_concepts_j"]):
            key = (i, j)
            value = statistic_dict[key]
            arr[key[0], key[1]] = value.statistic

    return arr
