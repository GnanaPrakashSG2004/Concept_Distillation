import os
from pprint import pprint

import torch
import numpy as np
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from src.dictionary_learning import DictionaryLearner
from extract_model_activations import build_param_dicts
from src.utils.parser_helper import concept_extraction_parser
from src.utils import model_loader, concept_extraction_helper as ceh


def load_activations(activations_root_folder, layer, class_idx):
    try:
        activations = torch.load(
            os.path.join(activations_root_folder, layer, f"{class_idx}.pth"),
            weights_only=True,
        )
        return activations
    except FileNotFoundError:
        return None


@ignore_warnings(category=ConvergenceWarning)
def main():
    parser = concept_extraction_parser()
    args = parser.parse_args()
    print("Arguments:")
    pprint(vars(args))
    print()

    param_dicts, save_names = build_param_dicts(args, force_run=True)
    print("=" * 50)
    print()

    # Load model
    model_name, ckpt_path = param_dicts["model"]
    model_out = model_loader.load_timm_model(
        model_name, ckpt_path, args.cache_dir,
        device=param_dicts["device"], is_eval=True
    )
    model = model_out["model"]

    # Insert hooks to track activations
    fe_out = ceh.load_feature_extraction_layers(
        model, param_dicts["feature_extraction_params"]
    )

    class_list = param_dicts["class_list"]
    activations_folder = os.path.join(save_names["activations_dir"], "activations")
    concepts_folder = os.path.join(save_names["concepts_dir"], "concepts")
    os.makedirs(concepts_folder, exist_ok=True)
    dataset_name = param_dicts["dataset_params"]["dataset_name"]

    num_layers = len(fe_out["layer_names"])
    pbar = tqdm(fe_out["layer_names"][::-1])
    for li, layer in enumerate(pbar):  # reverse order to start from the last layer
        pbar.set_description_str(f"Extracting concepts for layer {layer}: {li+1}/{num_layers}")
        layer_folder = os.path.join(concepts_folder, layer)
        os.makedirs(layer_folder, exist_ok=True)

        for class_idx in class_list:
            if not args.overwrite_concepts and os.path.exists(os.path.join(layer_folder, f"{class_idx}.pkl")):
                print("Hi")
                continue
            pbar.set_postfix_str(f"Class {class_idx}")

            activations = load_activations(activations_folder, layer, class_idx)
            if activations is None:
                print(f"Activations for class {class_idx} not found. Skipping...")
                continue

            # Generate Concepts
            dl_params = param_dicts["dl_params"]
            dl = DictionaryLearner(dl_params["decomp_method"], params=dl_params)
            out = dl.fit_transform(activations)

            for val in out.values():
                if isinstance(val, torch.Tensor):
                    if torch.isnan(val).any():
                        raise ValueError(
                            f"NaN values found in the output for class {class_idx} in layer {layer}"
                        )

                elif isinstance(val, np.ndarray):
                    if np.isnan(val).any():
                        raise ValueError(
                            f"NaN values found in the output for class {class_idx} in layer {layer}"
                        )

            # Save concepts
            dl.save(out, path=os.path.join(layer_folder, f"{class_idx}.pkl"))

        if args.only_last_layer:
            break


if __name__ == "__main__":
    main()
