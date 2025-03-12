import torch
from math import ceil
from src.utils.parser_helper import concept_extraction_parser, build_param_dicts
from src.utils import saving, model_loader, concept_extraction_helper as ceh
from src.utils.hooks import ActivationHook
from src import eval_model
import json
import os
from tqdm import tqdm
from src.utils.funcs import _batch_inference, get_username
import numpy as np


def join_image_groups(strategy, image_group1, image_group2):
    """
    Joins two image groups based on a given strategy.

    Args:
        strategy (str): Strategy to join image groups ('union', 'intersection', 'model1', 'model2').
        image_group1 (dict): First image group dictionary.
        image_group2 (dict): Second image group dictionary.

    Returns:
        dict: Joined image group dictionary.
    """
    image_group = {}
    keys = image_group1.keys()
    for key in keys:
        if strategy == "union":
            # Union strategy: combine images from both groups, removing duplicates
            image_group[key] = list(
                set(image_group1.get(key, []) + image_group2.get(key, []))
            )
        elif strategy == "intersection":
            # Intersection strategy: keep only images present in both groups
            image_group[key] = list(
                set(image_group1[key]).intersection(image_group2[key])
            )
        elif strategy == "model1":
            # Model1 strategy: use image group from the first model
            image_group[key] = image_group1[key]
        elif strategy == "model2":
            # Model2 strategy: use image group from the second model
            image_group[key] = image_group2[key]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return image_group


def create_image_group(strategy, param_dicts, return_eval_dict=False):
    """
    Creates an image group based on a given strategy.

    Args:
        strategy (str): Strategy to create image group ('craft', 'union-craft').
        param_dicts (dict or list): Parameter dictionaries, can be a single dict or a list of two for 'union-craft'.
        return_eval_dict (bool, optional): Whether to return evaluation dictionary. Defaults to False.

    Returns:
        dict: Image group dictionary.
    """
    if strategy == "craft":
        # Load/compute model predictions for 'craft' strategy
        dataset_params = param_dicts["dataset_params"]
        dataset = dataset_params["dataset_name"]
        dataset_seed = dataset_params["seed"]
        split = dataset_params["split"]
        model_name, ckpt_path = param_dicts["model"]
        model_eval_path = f"/scratch/{get_username()}/rsvc-exps/model_evaluation/{dataset}/{model_name}_{split}.json"
        try:
            # Try to load pre-computed evaluation results
            with open(model_eval_path, "r") as f:
                eval_dict = json.load(f)
                predictions = eval_dict["predictions"]
        except FileNotFoundError:
            # If evaluation results not found, run evaluation and get predictions
            print("Evals not found; Evaluating model again...")
            predictions = eval_model.main(
                model_name,
                dataset,
                split,
                ckpt_path,
                data_root=f"/scratch/{get_username()}/imagenet_data",
                save_root=f"/scratch/{get_username()}/rsvc-exps/model_evaluation",
            )["predictions"]

        # load images for 'craft' strategy
        print("Loading images")
        image_group = ceh.group_images(
            method="craft",
            params={
                "predictions": predictions,
                "num_images": dataset_params["num_images"],
                "seed": dataset_seed,
            },
        )
    elif strategy == "union-craft":  # requires two models for 'union-craft' strategy
        image_groups = []
        eval_dict = []
        for pd in param_dicts:
            # Load/compute model predictions for each model in 'union-craft'
            dataset_params = pd["dataset_params"]
            dataset = dataset_params["dataset_name"]
            dataset_seed = dataset_params["seed"]
            split = dataset_params["split"]
            model_name, ckpt_path = pd["model"]
            model_eval_path = f"/scratch/{get_username()}/rsvc-exps/model_evaluation/{dataset}/{model_name}_{split}.json"
            _eval_dict = eval_model.load_or_run_evaluation(
                model_eval_path,
                dataset,
                split,
                model_name,
                ckpt_path,
                data_root="/scratch/{get_username()}/imagenet_data",
            )
            eval_dict.append(_eval_dict)
            predictions = _eval_dict["predictions"]

            # load images for each model in 'union-craft'
            print("Loading images")
            image_group = ceh.group_images(
                method="craft",
                params={
                    "predictions": predictions,
                    "num_images": None,
                    "seed": dataset_seed,
                },
            )
            image_groups.append(image_group)

        # Join image groups using union strategy for 'union-craft'
        image_group = join_image_groups("union", image_groups[0], image_groups[1])

        # subsample N images per class for 'union-craft'
        assert (
            param_dicts[0]["dataset_params"]["num_images"]
            == param_dicts[1]["dataset_params"]["num_images"]
        )
        assert (
            param_dicts[0]["dataset_params"]["seed"]
            == param_dicts[1]["dataset_params"]["seed"]
        )
        target_num_images = param_dicts[0]["dataset_params"]["num_images"]
        seed = param_dicts[0]["dataset_params"]["seed"]
        if target_num_images is not None:
            rng = np.random.default_rng(seed)
            subsampled_label_groups = {}
            for i in image_group.keys():
                path_list = sorted(image_group[i])
                if target_num_images > len(path_list):
                    print(
                        f"Warning: class {i} has only {len(path_list)} / {target_num_images} images"
                    )
                    num_images = len(path_list)
                else:
                    num_images = target_num_images

                subsampled_label_groups[i] = list(
                    rng.choice(path_list, size=num_images, replace=False)
                )

            image_group = subsampled_label_groups

    else:
        raise ValueError(f"Unknown image_group_strategy: {strategy}")

    if return_eval_dict:
        return image_group, eval_dict
    return image_group


def transform_images(
    class_idx,
    image_path_list,
    dataset_name,
    model_out,
    param_dicts,
    transform_type="patchify",
):
    """
    Transforms images based on the specified transform type.

    Args:
        image_path_list (list): List of image paths.
        dataset_name (str): Name of the dataset.
        model_out (dict): Model output dictionary containing transforms.
        param_dicts (dict): Parameter dictionary.
        transform_type (str, optional): Type of transformation ('patchify', 'test', 'train'). Defaults to 'patchify'.

    Returns:
        dict: Dictionary containing transformed images and image size.
    """
    if dataset_name == "nabirds_modified" or dataset_name == "nabirds_stanford_cars":
        # Use 'nabirds' data root for modified and stanford cars datasets
        dataset_name = "nabirds"
    if transform_type == "patchify":
        # Patchify transformation
        out = ceh.select_class_and_load_images(
            image_path_list=image_path_list,
            data_root=f"/scratch/{get_username()}/imagenet_data/{dataset_name}/",
            transform=model_out["test_transform"],
        )
        if out is None:
            print(f"Class {class_idx} not found in image group")
            return None

        image_size = out["image_size"]
        patch_size = param_dicts["feature_extraction_params"]["patch_size"]
        images = ceh.patchify_images(
            out["images_preprocessed"], patch_size, strides=None
        )
    elif transform_type == "test":
        # Test transformation (no patchify)
        num_image_repeats = param_dicts["feature_extraction_params"][
            "num_image_repeats"
        ]
        out = ceh.select_class_and_load_images(
            image_path_list=image_path_list,
            data_root=f"/scratch/{get_username()}/imagenet_data/{dataset_name}/",
            transform=model_out["test_transform"],
            num_repeats=num_image_repeats,
        )

        if out is None:
            print(f"Class {class_idx} not found in image group")
            return None

        image_size = out["image_size"]
        images = out["images_preprocessed"]
    elif transform_type == "train":
        # Train transformation (no patchify, uses train transform)
        num_image_repeats = param_dicts["feature_extraction_params"][
            "num_image_repeats"
        ]
        out = ceh.select_class_and_load_images(
            image_path_list=image_path_list,
            data_root=f"/scratch/{get_username()}/imagenet_data/{dataset_name}/",
            transform=model_out["transform"],
            num_repeats=num_image_repeats,
        )
        if out is None:
            print(f"Class {class_idx} not found in image group")
            return None

        image_size = out["image_size"]
        images = out["images_preprocessed"]
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return dict(images=images, image_size=image_size)


if __name__ == "__main__":
    # Initialize argument parser
    parser = concept_extraction_parser()
    # Parse command line arguments
    args = parser.parse_args()
    # Build parameter dictionaries and save names based on arguments
    param_dicts, save_names = build_param_dicts(args)
    force_run = True  # Safety to prevent accidental overwriting

    # Get directories for activations and visualizations
    activations_dir = save_names["activations_dir"]
    visualization_dir = save_names["visualization_dir"]
    # Create activations directory if it doesn't exist or force_run is True
    if not os.path.exists(activations_dir) or force_run:
        os.makedirs(activations_dir, exist_ok=True)
    else:
        raise ValueError("Output directory already exists")

    # Create visualization directory if it doesn't exist or force_run is True
    if not os.path.exists(visualization_dir) or force_run:
        os.makedirs(visualization_dir, exist_ok=True)
    else:
        raise ValueError("Visualization directory already exists")
    print("Created directories...")

    # Load the model
    model_name, ckpt_path = param_dicts["model"]
    model_out = model_loader.load_model(
        model_name, ckpt_path, device=param_dicts["device"], eval=True
    )
    model = model_out["model"]
    print("Loaded model...")

    # Load feature extraction layers and create activation hook
    fe_out = ceh.load_feature_extraction_layers(
        model, param_dicts["feature_extraction_params"]
    )
    act_hook = ActivationHook(
        move_to_cpu_in_hook=args.move_to_cpu_in_hook,
        move_to_cpu_every=args.move_to_cpu_every,
    )
    # Register hooks to extract activations from specified layers
    act_hook.register_hooks(
        fe_out["layer_names"], fe_out["layers"], fe_out["post_activation_func"]
    )
    print("Registered hooks...")

    # Group images based on the specified strategy
    igs = param_dicts["feature_extraction_params"][
        "image_group_strategy"
    ]  # Set as CRAFT by default
    image_group = create_image_group(strategy=igs, param_dicts=param_dicts)
    print("Created image groups...")

    # Get class list and create folder for activations
    class_list = param_dicts["class_list"]
    activations_folder = os.path.join(activations_dir, "activations")
    # Create layer-wise folders to store activations
    for layer in fe_out["layer_names"]:
        os.makedirs(os.path.join(activations_folder, layer), exist_ok=True)
    dataset_name = param_dicts["dataset_params"]["dataset_name"]

    print("Extracting activations...")
    # Iterate over classes and extract activations
    pbar = tqdm(class_list)
    for class_idx in pbar:
        # Skip if class index is not in the image group
        if class_idx not in image_group.keys():
            print(f"Class {class_idx} not found in image group")
            continue
        # Transform images for the current class
        transform_out = transform_images(
            class_idx,
            image_group[class_idx],
            dataset_name,
            model_out,
            param_dicts,
            transform_type=args.transform,
        )
        images = transform_out["images"]
        image_size = transform_out["image_size"]

        # Extract model activations using batch inference
        preds = _batch_inference(
            model,
            images,
            batch_size=args.batch_size,
            resize=image_size,
            device=param_dicts["device"],
        )
        # Concatenate activations collected by hooks
        act_hook.concatenate_layer_activations()

        # Save activations for each layer and class index
        for layer in act_hook.layer_activations.keys():
            torch.save(
                act_hook.layer_activations[layer],
                os.path.join(activations_folder, layer, f"{class_idx}.pth"),
            )

        # Reset activation hook for the next class
        act_hook.reset_activation_dict()
