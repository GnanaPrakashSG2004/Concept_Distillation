from tqdm import tqdm
import os
import pickle
import numpy as np
import argparse
import logging
from datetime import datetime
import time
import warnings
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)
import timm
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from typing import Union
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

try:
    import wandb

    WANDB_AVAILABLE = True
    print("Loaded wandb successfully.")
except ImportError:
    WANDB_AVAILABLE = False
    print(
        "Warning: wandb not available. Install with 'pip install wandb' to enable wandb logging."
    )


def setup_logging(log_dir, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("concept_distillation")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


class ConceptDistillationTrainer:
    def __init__(
        self,
        pkl_dir,
        mapping_file,
        base_dataset,
        teacher_model,
        student_model,
        teacher_layer_name,
        student_layer_name,
        save_dir,
        val_loader,
        alpha,
        lambda_reg,
        kl_temp,
        nmf_max_iter,
        nmf_tol,
        batch_size,
        num_workers,
        device,
        log_interval,
        use_wandb,
        seed,
        save_best_only,
    ):
        global logger
        self.logger = logger
        self.num_classes = 1000
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.save_dir = save_dir
        self.val_loader = val_loader
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.kl_temperature = kl_temp
        self.nmf_max_iter = nmf_max_iter
        self.nmf_tol = nmf_tol
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_name = student_layer_name
        self.seed = seed
        self.save_best_only = save_best_only

        # Ensure alpha is valid
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self.logger.info(
            f"Using alpha = {self.alpha}, lambda_reg = {self.lambda_reg}, kl_temp = {self.kl_temperature}"
        )
        self.logger.info(
            f"NMF params: max_iter={self.nmf_max_iter}, tol={self.nmf_tol}"
        )

        # Set models
        self.teacher_model = teacher_model.to(device)  # Already in eval mode from main
        self.student_model = student_model.to(device)  # Already in train mode from main
        self.logger.info(
            f"Initialized teacher model: {type(teacher_model).__name__} (Layer: {self.teacher_layer_name})"
        )
        self.logger.info(
            f"Initialized student model: {type(student_model).__name__} (Layer: {self.student_layer_name})"
        )

        # Load teacher concepts
        self.logger.info(f"Loading teacher concepts (W_teacher) from {pkl_dir}")
        self.teacher_concepts = {}
        num_concepts = None
        loaded_classes = set()
        for class_id in range(self.num_classes):
            pkl_path = os.path.join(pkl_dir, f"{class_id}.pkl")
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                if "W" not in data:
                    self.logger.warning(
                        f"Key 'W' not found in concept file for class {class_id}. Skipping."
                    )
                    continue

                W_teacher = data["W"]  # Shape [num_concepts, teacher_feat_dim]
                if not isinstance(W_teacher, np.ndarray):
                    self.logger.warning(
                        f"Concept data 'W' for class {class_id} is not a NumPy array. Skipping."
                    )
                    continue

                current_num_concepts = W_teacher.shape[0]
                if num_concepts is None:
                    num_concepts = current_num_concepts
                    self.teacher_feat_dim = W_teacher.shape[
                        1
                    ]  # Get teacher feature dimension
                elif num_concepts != current_num_concepts:
                    self.logger.error(
                        f"Inconsistent number of concepts found for class {class_id} ({current_num_concepts}) vs expected ({num_concepts}). Stopping."
                    )
                    raise ValueError(
                        "Inconsistent number of concepts found across classes."
                    )
                elif self.teacher_feat_dim != W_teacher.shape[1]:
                    self.logger.error(
                        f"Inconsistent feature dimension found for class {class_id} ({W_teacher.shape[1]}) vs expected ({self.teacher_feat_dim}). Stopping."
                    )
                    raise ValueError(
                        "Inconsistent feature dimension found across classes."
                    )

                self.teacher_concepts[class_id] = W_teacher
                loaded_classes.add(class_id)
            except Exception as e:
                self.logger.error(
                    "pkl_path {pkl_path} could not be loaded Check pkl_dir and file contents.",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"pkl_path {pkl_path} could not be loaded. Check pkl_dir and file contents."
                )

        if not self.teacher_concepts:
            raise ValueError(
                "No teacher concepts were successfully loaded. Check pkl_dir and file contents."
            )
        self.num_concepts = num_concepts
        self.logger.info(
            f"Loaded concepts for {len(self.teacher_concepts)} classes, each with {self.num_concepts} concepts (Teacher Feat Dim: {self.teacher_feat_dim})."
        )

        # Data Loading Setup
        self.logger.info(f"Loading class mappings from {mapping_file}")
        try:
            mapping_arr = np.load(mapping_file, allow_pickle=True).item()
        except Exception as e:
            self.logger.error(
                f"Failed to load mapping file {mapping_file}: {e}", exc_info=True
            )
            raise

        train_indices = []
        mapped_classes = set(mapping_arr.keys())
        missing_concept_classes_in_map = set()

        for class_id in range(self.num_classes):
            if class_id in mapped_classes:
                if class_id in loaded_classes:
                    indices_for_class = mapping_arr[class_id]
                    if isinstance(indices_for_class, list) or isinstance(
                        indices_for_class, np.ndarray
                    ):
                        train_indices.extend(indices_for_class)
                    else:
                        self.logger.warning(
                            f"Invalid format for indices in mapping file for class {class_id}. Expected list or ndarray."
                        )
                else:
                    missing_concept_classes_in_map.add(class_id)
            else:
                self.logger.debug(f"Class {class_id} not found in mapping file.")

        if missing_concept_classes_in_map:
            self.logger.warning(
                f"Classes found in mapping file but missing concept data (excluded): {sorted(list(missing_concept_classes_in_map))}"
            )
            raise RuntimeError(
                f"Classes {sorted(list(missing_concept_classes_in_map))} not found in mapping file."
            )

        if not train_indices:
            raise ValueError(
                "No valid training indices found after filtering based on loaded concepts and mapping file."
            )

        self.logger.info(f"Creating training subset with {len(train_indices)} samples.")
        try:
            max_index = len(base_dataset) - 1
            valid_train_indices = [
                idx for idx in train_indices if 0 <= idx <= max_index
            ]
            if len(valid_train_indices) < len(train_indices):
                self.logger.warning(
                    f"Excluded {len(train_indices) - len(valid_train_indices)} indices from mapping file as they were out of bounds for the dataset (size {len(base_dataset)})."
                )
            if not valid_train_indices:
                raise ValueError(
                    "No valid indices remain after checking dataset bounds."
                )
            train_subset = Subset(base_dataset, valid_train_indices)
        except Exception as e:
            self.logger.error(f"Error creating Subset: {e}", exc_info=True)
            raise

        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,  # Keep last batch even if smaller
            pin_memory=True if device == "cuda" else False,
        )
        self.dataset_size = len(train_subset)  # Correct size after filtering
        self.logger.info(
            f"Created training DataLoader with {len(self.train_loader)} batches."
        )

        # Initialize metrics tracker
        self.metrics = {
            "epoch_losses": [],
            "batch_losses": [],  # List of lists, one per epoch
            "best_val_accuracy": -1.0,
            "best_epoch": -1,
        }

        # Loss functions
        self.classification_criterion = nn.KLDivLoss(reduction="batchmean")

    def save_model(self, epoch, is_best=False, optimizer=None, scheduler=None):
        os.makedirs(self.save_dir, exist_ok=True)
        base_filename = "student_model"
        state = {
            "epoch": epoch,
            "model_state_dict": self.student_model.state_dict(),
            "metrics": self.metrics,
            "best_val_accuracy": self.metrics["best_val_accuracy"],
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()

        latest_path = os.path.join(self.save_dir, f"{base_filename}_latest.pth")
        torch.save(state, latest_path)
        self.logger.info(f"Latest model checkpoint saved at {latest_path}")

        if not self.save_best_only:
            epoch_path = os.path.join(
                self.save_dir, f"{base_filename}_epoch_{epoch}.pth"
            )
            torch.save(state, epoch_path)
            self.logger.info(f"Epoch {epoch} model checkpoint saved at {epoch_path}")

        if is_best:
            best_path = os.path.join(self.save_dir, f"{base_filename}_best.pth")
            torch.save(state, best_path)
            self.logger.info(
                f"*** New best model saved at {best_path} (Epoch: {epoch}, Acc: {self.metrics['best_val_accuracy']:.2f}%) ***"
            )

    def extract_features_from_last_layer(self, model, x, target_layer_name):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                # Detach output within the hook
                activation[name] = output.detach()

            return hook

        hook_handle = None
        target_module = None
        features = None

        # No fall backs
        target_module = model.get_submodule(target_layer_name)
        hook_handle = target_module.register_forward_hook(
            get_activation(target_layer_name)
        )
        self.logger.debug(f"Registered hook on specified layer: {target_layer_name}")

        # Forward pass (will trigger hook if registered)
        _ = model(x)

        if hook_handle:
            hook_handle.remove()
            if target_layer_name in activation:
                features = activation[target_layer_name]
            else:
                self.logger.error(
                    f"Hook for {target_layer_name} was registered but activation not found!"
                )
                # Attempt fallback even if hook was registered but failed
                target_module = None

        if features is None:
            raise RuntimeError(
                f"Failed to extract features for model {type(model).__name__}"
            )

        # Global average pooling if features are 4D (common for CNNs)
        if features.ndim == 4:
            features = torch.mean(features, dim=[2, 3])
            self.logger.debug("Applied Global Average Pooling to 4D features.")

        # Ensure features are detached (hooks usually detach, fallbacks might not)
        return features.detach()

    def find_nmf_fixed_H(
        self,
        features,  # Expects torch tensor [group_size, teacher_feat_dim]
        concept_bank: np.ndarray,  # Expects numpy array [n_concepts, teacher_feat_dim] (W_teacher)
        verbose: bool = False,
    ) -> np.ndarray:
        if not isinstance(concept_bank, np.ndarray):
            self.logger.error(
                "Internal error: concept_bank (W_teacher) must be a NumPy array."
            )
            raise TypeError("concept_bank must be a NumPy array.")

        # 1. Prepare features (target V = A_teacher)
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            self.logger.error(
                "Internal error: 'features' input to find_nmf_fixed_H was not a tensor."
            )
            raise TypeError("features must be a torch.Tensor.")

        # NMF requires non-negative input
        if np.any(features_np < 0):
            if verbose:
                warnings.warn(
                    "Input 'features' contains negative values. Clipping to 0.",
                    UserWarning,
                )
            features_np = np.maximum(features_np, 0)

        # 2. Prepare W_teacher (fixed H)
        if np.any(concept_bank < 0):
            if verbose:
                warnings.warn(
                    "Input 'concept_bank' contains negative values. Clipping to 0.",
                    UserWarning,
                )
            W_teacher_nonneg = np.maximum(concept_bank, 0)
        else:
            W_teacher_nonneg = concept_bank

        # 3. Check shapes
        n_samples, feat_dim_features = features_np.shape
        n_concepts, feat_dim_W = W_teacher_nonneg.shape
        if feat_dim_features != feat_dim_W:
            # Feature dimensions should match between teacher features and teacher concepts
            self.logger.error(
                f"Shape mismatch in find_nmf_fixed_H: teacher features dim ({feat_dim_features}) != concept bank dim ({feat_dim_W})."
            )
            raise ValueError(
                f"Feature dimension mismatch: {feat_dim_features} != {feat_dim_W}"
            )
        if n_samples == 0:
            self.logger.error(
                "find_nmf_fixed_H called with 0 samples. Returning empty array."
            )
            raise ValueError("No samples provided for NMF.")

        # 4. Initialize and run NMF (Find W=U_teacher, given V=A_teacher, fixed H=W_teacher)
        nmf_model = NMF(
            n_components=n_concepts,
            init="random",
            solver="mu",  # 'mu' is common for fixed H
            max_iter=self.nmf_max_iter,
            tol=self.nmf_tol,
            random_state=self.seed,
            beta_loss="frobenius",
            verbose=verbose > 1,
        )
        # Initialize W (the matrix to find, U_teacher) [n_samples, n_concepts]
        # Sklearn's NMF with fixed H requires W init, but fit_transform finds W.
        # We are essentially finding W (U_teacher) given X (A_teacher) and fixed H (W_teacher).
        # sklearn NMF terminology: X ≈ W @ H. Here X=features_np, H=W_teacher_nonneg. We want W.

        try:
            with warnings.catch_warnings():
                # Ignore convergence warnings if max_iter is reached
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # Ignore potential sklearn NMF warnings about non-positivity if clipping happened
                warnings.filterwarnings(
                    "ignore", message=".*positive.*", category=UserWarning
                )
                # The `fit_transform` finds W when H is not given.
                # To fit with fixed H, we need to use the internal `_fit_transform` or manually iterate.
                # Let's try the standard `fit_transform` and see if it respects a fixed H if passed somehow.
                # It seems `fit_transform` *doesn't* support fixed H directly in public API.
                # We might need a custom loop or approximation if NMF doesn't have this feature easily accessible.
                # --- Re-evaluating sklearn NMF for fixed H ---
                # Sklearn's NMF `fit` *can* take `H` and `W` for custom init, but `fit_transform` finds `W`.
                # To find `W` with fixed `H`, we might need to transpose the problem or use a different library?
                # Let's try transposing: Solve A_teacher.T ≈ W_teacher.T @ U_teacher.T
                # Here, X = A_teacher.T, W = W_teacher.T (fixed), H = U_teacher.T (to find)
                X_T = features_np.T  # [feat_dim, n_samples]
                W_fixed_T = W_teacher_nonneg.T  # [feat_dim, n_concepts]
                n_components_H = (
                    n_samples  # H should have shape (n_concepts, n_samples)
                )

                # Initialize H (U_teacher.T)
                rng_T = np.random.RandomState(self.seed)
                dtype_T = X_T.dtype
                W_fixed_T = W_fixed_T.astype(dtype_T, copy=False)
                H_init_T = rng_T.rand(n_concepts, n_samples).astype(dtype_T, copy=False)
                H_init_T = np.maximum(H_init_T, 1e-9)  # Ensure positivity

                nmf_transposed = NMF(
                    n_components=n_concepts,  # Number of columns in FIXED W (W_teacher.T)
                    init="custom",
                    solver="mu",
                    beta_loss="frobenius",
                    max_iter=self.nmf_max_iter,
                    tol=self.nmf_tol,
                    random_state=self.seed,
                    verbose=verbose > 1,
                    # No regularization needed here typically for U_teacher
                )

                # Fit using X=A_teacher.T, fixed W=W_teacher.T, initial H=H_init_T (U_teacher.T)
                # Use `fit` not `fit_transform` because we provide W
                nmf_transposed.fit(X=X_T, W=W_fixed_T, H=H_init_T)
                U_teacher_T = (
                    nmf_transposed.components_
                )  # Learned H is U_teacher.T [n_concepts, n_samples]
                U_teacher = U_teacher_T.T  # Transpose back to [n_samples, n_concepts]

            # Check for NaNs/Infs
            if (
                U_teacher is None
                or np.isnan(U_teacher).any()
                or np.isinf(U_teacher).any()
            ):
                self.logger.error(
                    "NMF (find_nmf_fixed_H - transposed) resulted in NaN/Inf."
                )
                return None  # Indicate failure

        except Exception as e:
            self.logger.error(
                f"NMF (find_nmf_fixed_H - transposed) failed: {e}", exc_info=True
            )
            raise

        if verbose:
            # Rec error is for the transposed problem
            rec_err = nmf_transposed.reconstruction_err_
            print(
                f"NMF fixed H (transposed) complete. Final reconstruction error: {rec_err:.4f}"
            )

        assert np.all(
            U_teacher >= 0
        ), "NMF result U_teacher should be non-negative. Check input data and NMF settings."

        return U_teacher  # Shape: (n_samples, n_concepts)

    def compute_projection_matrix_nmf(
        self,
        student_features: torch.Tensor,  # A_S [m, n] (batch_size, student_feat_dim)
        target_coefficients: torch.Tensor,  # U [m, k] (batch_size, n_concepts)
        verbose: bool = False,
    ) -> Union[
        np.ndarray, None
    ]:  # Returns W* [n, k] (student_feat_dim, n_concepts) or None on failure
        # 1. Input Conversion and Validation
        if isinstance(student_features, torch.Tensor):
            A_S = student_features.detach().cpu().numpy()
        else:
            self.logger.error(
                "Internal error: student_features must be Tensor in compute_projection_matrix_nmf."
            )
            raise TypeError("student_features must be a torch.Tensor.")

        if isinstance(target_coefficients, torch.Tensor):
            U = target_coefficients.detach().cpu().numpy()
        else:
            self.logger.error(
                "Internal error: target_coefficients must be Tensor in compute_projection_matrix_nmf."
            )
            raise TypeError("target_coefficients must be a torch.Tensor.")

        # Clip negatives (NMF requirement)
        if np.any(A_S < 0):
            if verbose:
                warnings.warn(
                    "Clipping negative values in student_features (A_S) to 0 for NMF.",
                    UserWarning,
                )
            A_S = np.maximum(A_S, 0)
        if np.any(U < 0):
            if verbose:
                warnings.warn(
                    "Clipping negative values in target_coefficients (U) to 0 for NMF.",
                    UserWarning,
                )
            U = np.maximum(U, 0)

        if A_S.shape[0] != U.shape[0]:
            self.logger.error(
                f"Shape mismatch in compute_projection_matrix_nmf: student_features samples {A_S.shape[0]} != target_coefficients samples {U.shape[0]}."
            )
            raise ValueError(f"Shape mismatch: {A_S.shape[0]} != {U.shape[0]}")
        if self.lambda_reg < 0:
            self.logger.error(
                f"Invalid lambda_reg: {self.lambda_reg}. Must be non-negative."
            )
            raise ValueError(
                f"Invalid lambda_reg: {self.lambda_reg}. Must be non-negative."
            )

        # Handle empty inputs
        m, n_features = A_S.shape  # m samples, n student features
        _, k_targets = U.shape  # m samples, k target coefficients/concepts
        if m == 0:
            self.logger.warning(
                "compute_projection_matrix_nmf called with 0 samples. Returning zeros."
            )
            raise ValueError("No samples provided for NMF.")

        # 2. Setup NMF: Solve U ≈ A_S @ W* (X ≈ W @ H => X=U, W=A_S(fixed), H=W*(learn))
        # We need to find H (W*) given X (U) and fixed W (A_S).
        nmf_model = NMF(
            n_components=n_features,  # Number of columns in FIXED W (A_S) -> This should be n_concepts for H!
            init="random",  # Or 'nndsvda' for potentially better results
            solver="mu",
            beta_loss="frobenius",
            max_iter=self.nmf_max_iter,
            tol=self.nmf_tol,
            random_state=self.seed,
            l1_ratio=0.0,  # l1_ratio=0 means L2 regularization
            alpha_W=0.0,  # No regularization on fixed W (A_S)
            alpha_H=self.lambda_reg,  # L2 regularization on H (W*): lambda * ||W*||_F^2 / 2 (sklearn uses alpha = lambda)
            # Note: Sklearn NMF alpha_H applies 0.5 * alpha_H * ||H||_F^2. So alpha_H=lambda_reg seems correct if lambda_reg is the factor multiplying the L2 norm squared. Double check convention if results are sensitive.
            verbose=verbose > 1,
        )

        # Correcting n_components for finding H:
        # X [m, k_targets] ≈ W [m, n_features] @ H [n_features, k_targets]
        # We want to find H = W*
        # So X = U, W = A_S (fixed), H = W* (learn)
        # The `n_components` parameter in sklearn NMF defines the inner dimension,
        # which is the number of columns in W and the number of rows in H.
        # Therefore, `n_components` should be `n_features` (student_feat_dim).

        nmf_model_correct = NMF(
            n_components=n_features,  # Number of components = student feature dimension
            init="random",
            solver="mu",
            beta_loss="frobenius",
            max_iter=self.nmf_max_iter,
            tol=self.nmf_tol,
            random_state=self.seed,
            l1_ratio=0.0,
            alpha_W=0.0,
            alpha_H=self.lambda_reg,
            verbose=verbose > 1,
        )

        # Initialize H (W*) [n_features, k_targets]
        rng = np.random.RandomState(self.seed)
        dtype = A_S.dtype
        H_init = rng.rand(n_features, k_targets).astype(dtype, copy=False)
        H_init = np.maximum(H_init, 1e-9)  # Ensure positivity

        # Fit using X=U, fixed W=A_S, initial H=H_init
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings(
                    "ignore", message=".*init.*", category=UserWarning
                )  # Ignore custom init warnings if needed
                warnings.filterwarnings(
                    "ignore", message=".*positive.*", category=UserWarning
                )
                # Use fit, providing W (A_S) and H_init (W*_init)
                nmf_model_correct.fit(X=U, W=A_S, H=H_init)
            W_star = (
                nmf_model_correct.components_
            )  # Learned H is our W* [n_features, k_targets]

            # Check for NaNs in result
            if W_star is None or np.isnan(W_star).any() or np.isinf(W_star).any():
                self.logger.error(
                    "NMF (compute_projection_matrix_nmf) resulted in NaN/Inf."
                )
                raise ValueError("NMF result W_star contains NaN/Inf.")

        except Exception as e:
            self.logger.error(
                f"NMF (compute_projection_matrix_nmf) failed: {e}", exc_info=True
            )
            raise RuntimeError(f"NMF failed: {e}")

        if verbose:
            rec_err = nmf_model_correct.reconstruction_err_
            print(f"NMF projection complete. Final reconstruction error: {rec_err:.4f}")

        assert np.all(
            W_star >= 0
        ), "NMF result W_star should be non-negative. Check input data and NMF settings."

        return W_star  # Shape: (n_features, k_targets)

    def concept_alignment_loss(self, student_features, W_star, teacher_concept_coeffs):
        # Ensure inputs are torch tensors on the correct device
        if isinstance(W_star, np.ndarray):
            # Ensure W_star non-negative before converting
            W_star_nonneg = np.maximum(W_star, 0)
            W_star_tensor = torch.tensor(
                W_star_nonneg, dtype=torch.float32, device=self.device
            )
        elif isinstance(W_star, torch.Tensor):
            W_star_tensor = W_star  # Assume already checked/non-negative if tensor
        else:
            self.logger.error(
                "Internal Error: W_star type unexpected in concept_alignment_loss"
            )
            raise TypeError("W_star must be a numpy array or torch tensor.")

        if isinstance(teacher_concept_coeffs, np.ndarray):
            # Ensure teacher_concept_coeffs non-negative before converting
            teacher_concept_coeffs_nonneg = np.maximum(teacher_concept_coeffs, 0)
            teacher_concept_coeffs_tensor = torch.tensor(
                teacher_concept_coeffs_nonneg, dtype=torch.float32, device=self.device
            )
        elif isinstance(teacher_concept_coeffs, torch.Tensor):
            teacher_concept_coeffs_tensor = teacher_concept_coeffs
        else:
            self.logger.error(
                "Internal Error: teacher_concept_coeffs type unexpected in concept_alignment_loss"
            )
            raise TypeError(
                "teacher_concept_coeffs must be a numpy array or torch tensor."
            )

        # Project student features to concept space: [B, S_feat] @ [S_feat, N_concepts] -> [B, N_concepts]
        # Student features should ideally be non-negative for consistency with NMF framework
        student_features_nonneg = torch.relu(
            student_features
        )  # Apply ReLU if not already non-negative

        # Check for NaNs/Infs before matmul (more robust checks)
        if (
            torch.isnan(student_features_nonneg).any()
            or torch.isinf(student_features_nonneg).any()
        ):
            self.logger.error(
                "NaN/Inf detected in (non-neg) student_features before alignment loss matmul."
            )
            raise ValueError(
                "NaN/Inf detected in student_features before alignment loss matmul."
            )
        if torch.isnan(W_star_tensor).any() or torch.isinf(W_star_tensor).any():
            self.logger.error(
                "NaN/Inf detected in W_star_tensor before alignment loss matmul."
            )
            raise ValueError(
                "NaN/Inf detected in W_star_tensor before alignment loss matmul."
            )

        student_concepts = torch.matmul(student_features_nonneg, W_star_tensor)

        # Check teacher coeffs after conversion
        if (
            torch.isnan(teacher_concept_coeffs_tensor).any()
            or torch.isinf(teacher_concept_coeffs_tensor).any()
        ):
            self.logger.error(
                "NaN/Inf detected in teacher_concept_coeffs_tensor before alignment loss calculation."
            )
            raise ValueError(
                "NaN/Inf detected in teacher_concept_coeffs_tensor before alignment loss calculation."
            )

        # Check shapes: both should be [B, N_concepts]
        if student_concepts.shape != teacher_concept_coeffs_tensor.shape:
            self.logger.error(
                f"Shape mismatch in alignment loss: student_concepts {student_concepts.shape}, teacher_coeffs {teacher_concept_coeffs_tensor.shape}"
            )
            raise ValueError(
                f"Shape mismatch in alignment loss: {student_concepts.shape} != {teacher_concept_coeffs_tensor.shape}"
            )

        # Compute MSE loss: || student_concepts - teacher_concept_coeffs ||^2
        alignment_loss = torch.nn.functional.mse_loss(
            student_concepts, teacher_concept_coeffs_tensor, reduction="mean"
        )

        # Check for NaN loss
        if torch.isnan(alignment_loss):
            self.logger.error("NaN detected in calculated alignment_loss.")
            raise ValueError("NaN detected in calculated alignment_loss.")

        return alignment_loss

    def train(self, epochs, lr):
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(self.train_loader)
        )

        start_time = time.time()
        self.logger.info(f"Starting training for {epochs} epochs.")

        total_steps = 0

        self.metrics["batch_losses"] = [[] for _ in range(epochs)]
        self.metrics["nmf_u_teacher_failures"] = [0] * epochs
        self.metrics["nmf_w_star_failures"] = [0] * epochs

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"=== EPOCH {epoch+1}/{epochs} ===")
            self.student_model.train()

            epoch_loss = 0.0
            epoch_class_loss = 0.0
            epoch_align_loss = 0.0
            batch_count = 0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
            )

            for batch_idx, batch in enumerate(progress_bar):
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # 1. Forward pass for classification (Teacher and Student)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                student_outputs = self.student_model(images)

                # 2. Classification loss (KL Divergence)
                soft_targets = F.softmax(teacher_outputs / self.kl_temperature, dim=-1)
                log_soft_student_outputs = F.log_softmax(
                    student_outputs / self.kl_temperature, dim=-1
                )
                # Temperature scaling for KL loss
                class_loss = (self.kl_temperature**2) * self.classification_criterion(
                    log_soft_student_outputs, soft_targets
                )

                # --- Concept Alignment Calculation (only if alpha > 0) ---
                alignment_loss = torch.tensor(0.0, device=self.device)
                if self.alpha > 0:
                    # Reset batch-specific variables
                    W_star_np = None

                    # 3. Extract features
                    with torch.no_grad():
                        teacher_features = self.extract_features_from_last_layer(
                            self.teacher_model, images, self.teacher_layer_name
                        )
                    student_features = self.extract_features_from_last_layer(
                        self.student_model, images, self.student_layer_name
                    )

                    # Initialize tensor for U_teacher [B, N_concepts] and tracking mask
                    batch_size = images.size(0)
                    batch_U_teacher_tensor = torch.zeros(
                        batch_size,
                        self.num_concepts,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    processed_indices = torch.zeros(
                        batch_size, dtype=torch.bool, device=self.device
                    )  # Track which samples got U_teacher

                    # 4a. Calculate U_teacher for the relevant parts of the batch
                    unique_labels_in_batch = labels.unique()
                    for class_id_tensor in unique_labels_in_batch:
                        class_id = class_id_tensor.item()

                        # Skip if concepts for this class aren't loaded
                        if class_id not in self.teacher_concepts:
                            self.logger.debug(
                                f"Skipping alignment for class {class_id} samples in batch {batch_idx} - concepts not loaded."
                            )
                            continue

                        mask = labels == class_id_tensor
                        indices_in_batch = mask.nonzero(as_tuple=False).squeeze(-1)

                        if len(indices_in_batch) == 0:
                            continue  # Safety check

                        teacher_features_group = teacher_features[mask]
                        W_teacher = self.teacher_concepts[class_id]  # Numpy array

                        # Calculate U_teacher for this group (Assume it returns a valid np.ndarray)
                        U_teacher_group_np = self.find_nmf_fixed_H(
                            teacher_features_group, W_teacher, verbose=False
                        )

                        # Convert result to tensor and place in batch tensor
                        U_teacher_group = torch.tensor(
                            U_teacher_group_np, dtype=torch.float32, device=self.device
                        )
                        batch_U_teacher_tensor[indices_in_batch] = U_teacher_group
                        processed_indices[indices_in_batch] = (
                            True  # Mark these samples as having U_teacher calculated
                        )

                    # Proceed only if at least one U_teacher was computed
                    if processed_indices.any():
                        # Filter features and U_teacher to only include processed samples
                        student_features_filt = student_features[processed_indices]
                        batch_U_teacher_filt = batch_U_teacher_tensor[processed_indices]

                        # 4b. Compute ONE projection matrix W* for the valid part of the batch
                        W_star_np = self.compute_projection_matrix_nmf(
                            student_features_filt,  # Torch tensor [B_filt, S_feat]
                            batch_U_teacher_filt,  # Torch tensor [B_filt, N_concepts]
                            verbose=False,
                        )  # Assume it returns a valid np.ndarray

                        # 5. Compute alignment loss (using the W* computed for the batch subset)
                        # (The concept_alignment_loss function internally handles W* conversion)
                        alignment_loss = self.concept_alignment_loss(
                            student_features_filt,  # Pass filtered student features [B_filt, S_feat]
                            W_star_np,  # Pass computed W* (numpy)
                            batch_U_teacher_filt,  # Pass filtered U_teacher [B_filt, N_concepts]
                        )
                    else:
                        # Case where no samples in the batch had loadable concepts
                        self.logger.warning(
                            f"No processable samples found for alignment in batch {batch_idx}. Skipping alignment loss calculation."
                        )
                # --- End Concept Alignment Calculation ---

                # 6. Compute total loss and backprop
                if torch.isnan(class_loss) or torch.isinf(class_loss):
                    self.logger.error(
                        f"NaN/Inf detected in class_loss at step {total_steps}. Skipping optimizer step."
                    )
                    raise ValueError("NaN/Inf detected in class_loss.")
                if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                    self.logger.warning(
                        f"NaN/Inf detected in alignment_loss at step {total_steps}. Setting alignment contribution to 0."
                    )
                    raise ValueError("NaN/Inf detected in alignment_loss.")

                total_loss = (
                    1.0 - self.alpha
                ) * class_loss + self.alpha * alignment_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    self.logger.error(
                        f"NaN/Inf detected in total_loss at step {total_steps} after combining. Skipping optimizer step."
                    )
                    raise ValueError("NaN/Inf detected in total_loss after combining.")

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(), max_norm=1.0
                )

                optimizer.step()
                scheduler.step()

                epoch_loss += total_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_align_loss += alignment_loss.item()
                batch_count += 1
                total_steps += 1

                # Log batch stats
                current_lr = scheduler.get_last_lr()[0]  # Get LR from scheduler
                if (
                    batch_idx % self.log_interval == 0
                    or batch_idx == len(self.train_loader) - 1
                ):
                    log_msg = (
                        f"Epoch {epoch+1}, Step {total_steps} [{batch_idx+1}/{len(self.train_loader)}], "
                        f"LR: {current_lr:.6f}, Loss: {total_loss.item():.4f} "
                        f"(Class: {class_loss.item():.4f}, Align: {alignment_loss.item():.4f})"
                    )
                    self.logger.info(log_msg)

                progress_bar.set_postfix(loss=total_loss.item(), lr=current_lr)
                self.metrics["batch_losses"][epoch].append(
                    {
                        "total": total_loss.item(),
                        "alignment": alignment_loss.item(),
                        "classification": class_loss.item(),
                    }
                )

                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log(
                        {
                            "train/batch_loss": total_loss.item(),
                            "train/class_loss": class_loss.item(),
                            "train/alignment_loss": alignment_loss.item(),
                            "train/learning_rate": current_lr,
                            "step": total_steps,
                            "epoch_frac": epoch
                            + (batch_idx / len(self.train_loader)),  # Fractional epoch
                        },
                        step=total_steps,
                    )  # Use total_steps as the step counter

            progress_bar.close()

            # Epoch summary
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_epoch_class_loss = (
                epoch_class_loss / batch_count if batch_count > 0 else 0
            )
            avg_epoch_align_loss = (
                epoch_align_loss / batch_count if batch_count > 0 else 0
            )
            self.metrics["epoch_losses"].append(avg_epoch_loss)

            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s.")
            self.logger.info(f"  Avg Loss: {avg_epoch_loss:.4f}")
            self.logger.info(f"  Avg Class Loss: {avg_epoch_class_loss:.4f}")
            self.logger.info(f"  Avg Align Loss: {avg_epoch_align_loss:.4f}")

            # Perform validation
            eval_results = self.evaluate(self.val_loader)
            current_val_accuracy = eval_results["student_accuracy"]

            # Save model checkpoint(s)
            is_best = False
            if current_val_accuracy > self.metrics["best_val_accuracy"]:
                self.metrics["best_val_accuracy"] = current_val_accuracy
                self.metrics["best_epoch"] = epoch + 1
                is_best = True
                self.logger.info(
                    f"*** New best validation accuracy: {current_val_accuracy:.2f}% at epoch {epoch+1} ***"
                )

            self.save_model(
                epoch + 1, is_best=is_best, optimizer=optimizer, scheduler=scheduler
            )

            # Log epoch metrics to wandb
            if self.use_wandb and WANDB_AVAILABLE:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_class_loss": avg_epoch_class_loss,
                    "train/epoch_align_loss": avg_epoch_align_loss,
                    "val/teacher_accuracy": eval_results["teacher_accuracy"],
                    "val/student_accuracy": current_val_accuracy,
                    "val/best_student_accuracy": self.metrics["best_val_accuracy"],
                    "epoch_time": epoch_time,
                }
                wandb.log(wandb_metrics, step=total_steps)  # Log against total steps
                # Also log epoch metrics against epoch number for easier comparison across runs
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        **{
                            k.replace("/", "_epoch/"): v
                            for k, v in wandb_metrics.items()
                            if "/" in k
                        },
                    }
                )

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(
            f"Best validation accuracy: {self.metrics['best_val_accuracy']:.2f}% at epoch {self.metrics['best_epoch']}"
        )
        return self.metrics

    def evaluate(self, loader):
        self.logger.info("Starting evaluation...")
        self.teacher_model.eval()
        self.student_model.eval()
        teacher_correct = 0
        student_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = labels.size(0)
                total += batch_size

                teacher_outputs = self.teacher_model(images)
                _, teacher_preds = torch.max(teacher_outputs, 1)
                teacher_correct += (teacher_preds == labels).sum().item()

                student_outputs = self.student_model(images)
                _, student_preds = torch.max(student_outputs, 1)
                student_correct += (student_preds == labels).sum().item()

        teacher_accuracy = 100.0 * teacher_correct / total if total > 0 else 0
        student_accuracy = 100.0 * student_correct / total if total > 0 else 0

        self.logger.info(f"Evaluation Results:")
        self.logger.info(
            f"  Teacher Accuracy: {teacher_accuracy:.2f}% ({teacher_correct}/{total})"
        )
        self.logger.info(
            f"  Student Accuracy: {student_accuracy:.2f}% ({student_correct}/{total})"
        )

        return {
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concept Distillation Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )  # Show defaults

    # --- Paths and Data ---
    parser.add_argument(
        "--pkl-dir",
        type=str,
        default="/scratch/swayam/rsvc-exps/outputs/data/dn=in_spl=val_ni=100_seed=0/r50_ckpt=None/ps=64_flv=v1_igs=c/dm=nmf_nc=10_seed=0/concepts/layer4.2.act3",
        help="Directory containing teacher concept pickle files",
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        default="restructured_map_val.npy",
        help="Path to class mapping file",
    )
    parser.add_argument(
        "--save-dir-parent",
        type=str,
        default="/scratch/swayam/rsvc-exps/concept_distillation/outputs/",
        help="Path where model is stored during training. Models are stored based on experiment name",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/scratch/swayam/imagenet_data/imagenet",
        help="Root directory for ImageNet dataset",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split to use (train or val)"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to use for generating training data indices ('train' or 'val'). Evaluation always uses 'val'.",
    )

    # --- Models ---
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="resnet50.a2_in1k",
        help="Teacher model architecture name from timm (e.g., 'resnet50.a2_in1k').",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="resnet18.a2_in1k",
        help="Student model architecture name from timm (e.g., 'resnet18.a2_in1k').",
    )
    parser.add_argument(
        "--timm-cache",
        type=str,
        default="/scratch/swayam/timm_cache/",
        help="Cache directory for timm models.",
    )
    parser.add_argument(
        "--teacher-layer",
        type=str,
        default="layer4.2.act3",
        help="Name of the layer in the teacher model for feature extraction (e.g., 'layer4.2.act3', 'global_pool').",
    )
    parser.add_argument(
        "--student-layer",
        type=str,
        default="layer4.1.act2",
        help="Name of the layer in the student model for feature extraction.",
    )

    # --- Training Hyperparameters ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Weight for the concept alignment loss (0.0 to 1.0). 0 means only KL loss.",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.01,
        help="L2 regularization strength (lambda) for NMF W* computation (alpha_H in sklearn).",
    )
    parser.add_argument(
        "--kl-temp",
        type=float,
        default=1.0,
        help="Temperature scaling for KL divergence loss.",
    )
    parser.add_argument(
        "--nmf-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for NMF solver.",
    )
    parser.add_argument(
        "--nmf-tol",
        type=float,
        default=1e-4,
        help="Tolerance for NMF solver convergence.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data loading workers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="If set, only save the best model checkpoint based on validation accuracy.",
    )

    # --- Logging ---
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name (optional, defaults to auto-generated). If None, defaults to exp_name",
    )
    parser.add_argument(
        "--log-dir-parent",
        type=str,
        default="/scratch/swayam/rsvc-exps/concept_distillation/logs/",
        help="Directory to save log files. Experiment name is appended.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=250,
        help="Log training batch metrics every N batches.",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable logging to Weights & Biases."
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="concept-distillation",
        help="WandB project name (used only if --use-wandb).",
    )

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir_parent, args.exp_name)
    args.save_dir = os.path.join(args.save_dir_parent, args.exp_name)
    args.wandb_name = args.wandb_name or args.exp_name

    return args


# Global logger variable
logger = None

if __name__ == "__main__":
    args = parse_args()

    # Setup logging (global logger)
    logger = setup_logging(args.log_dir)
    logger.info("=" * 50)
    logger.info("Starting Concept Distillation Pipeline")
    logger.info("=" * 50)
    logger.info(f"Arguments: {vars(args)}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb if requested
    if args.use_wandb:
        if WANDB_AVAILABLE:
            run_name = (
                args.wandb_name
                or f"distill_{args.teacher_model}_to_{args.student_model}_alpha{args.alpha}_seed{args.seed}"
            )
            logger.info(f"Initializing wandb with run name: {run_name}")
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),  # Log all arguments
            )
        else:
            logger.warning(
                "wandb logging requested but wandb is not available. Proceeding without wandb."
            )
            args.use_wandb = False  # Disable internally if not available

    # Load models
    logger.info(f"Loading teacher model: {args.teacher_model}")
    try:
        teacher_model = (
            timm.create_model(
                args.teacher_model, pretrained=True, cache_dir=args.timm_cache
            )
            .eval()
            .requires_grad_(False)
        )  # Set to eval and freeze
    except Exception as e:
        logger.error(
            f"Failed to load teacher model '{args.teacher_model}': {e}", exc_info=True
        )
        exit(1)

    logger.info(f"Loading student model: {args.student_model}")
    try:
        student_model = timm.create_model(
            args.student_model,
            pretrained=True,  # Start from pretrained weights
            cache_dir=args.timm_cache,
        ).train()  # Set to train mode initially
    except Exception as e:
        logger.error(
            f"Failed to load student model '{args.student_model}': {e}", exc_info=True
        )
        exit(1)

    # Setup data transformation (use teacher's config)
    try:
        config = resolve_data_config({}, model=teacher_model)
        transform = create_transform(**config)
        logger.info(f"Data transform created based on teacher model config.")
    except Exception as e:
        logger.error(f"Failed to create data transform: {e}", exc_info=True)
        exit(1)

    # Load base dataset (for training split)
    logger.info(
        f"Loading base dataset from {args.data_root}, split={args.dataset_split}"
    )
    try:
        base_dataset = datasets.ImageNet(
            root=args.data_root, split=args.dataset_split, transform=transform
        )
        logger.info(
            f"Base dataset ({args.dataset_split} split) loaded with {len(base_dataset)} samples."
        )
    except Exception as e:
        logger.error(
            f"Failed to load dataset from {args.data_root} (split {args.dataset_split}): {e}",
            exc_info=True,
        )
        exit(1)

    # Create validation dataloader (always use 'val' split)
    logger.info(f"Loading validation dataset from {args.data_root}, split=val")
    try:
        val_dataset = datasets.ImageNet(
            root=args.data_root,
            split="val",
            transform=transform,  # Use the same transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.device == "cuda" else False,
        )
        logger.info(f"Validation DataLoader created with {len(val_loader)} batches.")
    except Exception as e:
        logger.error(
            f"Failed to load validation dataset or create loader: {e}", exc_info=True
        )
        exit(1)

    # Checks
    logger.info("Checking if model layer names are valid")
    assert (
        args.student_layer in get_graph_node_names(student_model)[0]
    ), f"Student layer '{args.student_layer}' not found in model."
    assert (
        args.teacher_layer in get_graph_node_names(teacher_model)[0]
    ), f"Teacher layer '{args.teacher_layer}' not found in model."

    # Initialize trainer
    logger.info("Initializing ConceptDistillationTrainer...")
    try:
        trainer = ConceptDistillationTrainer(
            pkl_dir=args.pkl_dir,
            mapping_file=args.mapping_file,
            base_dataset=base_dataset,  # The dataset object for the training split
            teacher_model=teacher_model,
            student_model=student_model,
            teacher_layer_name=args.teacher_layer,
            student_layer_name=args.student_layer,
            save_dir=args.save_dir,
            val_loader=val_loader,  # Pass the val loader
            alpha=args.alpha,
            lambda_reg=args.lambda_reg,
            kl_temp=args.kl_temp,
            nmf_max_iter=args.nmf_max_iter,
            nmf_tol=args.nmf_tol,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            log_interval=args.log_interval,
            use_wandb=args.use_wandb,
            seed=args.seed,
            save_best_only=args.save_best_only,
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}", exc_info=True)
        exit(1)

    # Train
    logger.info("Starting training process...")
    try:
        metrics = trainer.train(epochs=args.epochs, lr=args.lr)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        # Optionally perform final evaluation even if training failed early
        logger.info("Attempting final evaluation despite training error...")
        try:
            eval_results = trainer.evaluate(val_loader)
        except Exception as eval_e:
            logger.error(f"Final evaluation also failed: {eval_e}", exc_info=True)
        exit(1)  # Exit after attempting eval

    # Final Evaluation (after successful training)
    logger.info("Performing final evaluation after training...")
    try:
        eval_results = trainer.evaluate(val_loader)
        logger.info(f"Final Student Accuracy: {eval_results['student_accuracy']:.2f}%")
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}", exc_info=True)

    logger.info("=" * 50)
    logger.info("Pipeline completed.")
    logger.info(
        f"Best validation accuracy achieved: {trainer.metrics['best_val_accuracy']:.2f}% at epoch {trainer.metrics['best_epoch']}"
    )
    logger.info(f"Checkpoints saved in: {args.save_dir}")
    logger.info(f"Logs saved in: {args.log_dir}")
    logger.info("=" * 50)

    # Close wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
