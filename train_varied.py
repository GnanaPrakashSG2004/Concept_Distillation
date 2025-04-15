from tqdm import tqdm
import os
import math
import pickle
import random
import numpy as np
import argparse
import logging
from datetime import datetime
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset # Added Subset
import timm
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from typing import Union
from sklearn.decomposition import NMF
# Optional wandb import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
    print("Loaded wandb successfully.")
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable wandb logging.")

# Configure logging (remains the same)
def setup_logging(log_dir, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('concept_distillation')

# SingleClassDataset is no longer needed for training, but might be kept if used elsewhere
# class SingleClassDataset(Dataset): ...

class ConceptDistillationTrainer:
    def __init__(
        self,
        pkl_dir,
        mapping_file,
        base_dataset, # This should be the dataset for the specified split (e.g., 'train' or 'val')
        teacher_model,
        student_model,
        save_dir,
        val_loader, # Keep the val_loader for evaluation
        alpha,
        kl_temp=1.0,
        batch_size=32,
        num_workers=0,
        device="cuda",
        log_interval=10,
        use_wandb=False
    ):
        self.logger = logging.getLogger('concept_distillation')
        self.num_classes = 1000 # Assuming ImageNet
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.save_dir = save_dir
        self.val_loader = val_loader # Loader for validation split
        self.alpha = alpha # Weight for alignment loss

        # Ensure alpha is valid
        if not (0.0 <= alpha <= 1.0):
             raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        self.logger.info(f"Using alpha = {self.alpha}")

        # Set models
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.logger.info(f"Initialized teacher model: {type(teacher_model).__name__}")
        self.logger.info(f"Initialized student model: {type(student_model).__name__}")

        # Load teacher concepts (remains the same)
        self.logger.info(f"Loading teacher concepts from {pkl_dir}")
        self.teacher_concepts = {}
        num_concepts = None # To store the number of concepts per class (should be consistent)
        for class_id in range(self.num_classes):
            pkl_path = os.path.join(pkl_dir, f"{class_id}.pkl")
            if not os.path.exists(pkl_path):
                self.logger.warning(f"Concept file not found for class {class_id} at {pkl_path}. Skipping.")
                continue # Or handle as error if all classes must have concepts
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.teacher_concepts[class_id] = data["W"] # Assuming shape [num_concepts, teacher_feat_dim]
            if num_concepts is None:
                num_concepts = self.teacher_concepts[class_id].shape[0]
            elif num_concepts != self.teacher_concepts[class_id].shape[0]:
                raise ValueError("Inconsistent number of concepts found across classes.")
        self.num_concepts = num_concepts # Store the number of concepts
        self.logger.info(f"Loaded concepts for {len(self.teacher_concepts)} classes, each with {self.num_concepts} concepts.")

        # --- Modified Data Loading Setup ---
        self.logger.info(f"Loading class mappings from {mapping_file}")
        mapping_arr = np.load(mapping_file, allow_pickle=True).item()

        # Create a list of all indices to be used for training from the base_dataset
        train_indices = []
        for class_id in range(self.num_classes):
            if class_id in mapping_arr: # Check if class exists in mapping
                 # Ensure we only add indices for which we have concepts loaded
                 if class_id in self.teacher_concepts:
                    indices_for_class = mapping_arr[class_id]
                    train_indices.extend(indices_for_class)
                 else:
                     self.logger.warning(f"Class {class_id} found in mapping but concepts are missing. Excluding samples.")
            # else: # Optional: Warn if a class is missing from mapping
            #      self.logger.warning(f"Class {class_id} not found in mapping file.")

        if not train_indices:
             raise ValueError("No valid training indices found. Check mapping file and concept directory.")

        self.logger.info(f"Creating training subset with {len(train_indices)} samples.")
        train_subset = Subset(base_dataset, train_indices)

        # Create a single DataLoader for the training subset
        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True if device == "cuda" else False # Optimization
        )
        self.dataset_size = len(train_indices)
        self.logger.info(f"Created training DataLoader with {len(self.train_loader)} batches.")
        # --- End Modified Data Loading Setup ---

        # Initialize metrics tracker
        self.metrics = {
            "epoch_losses": [],
            "batch_losses": [],
        }

        # Loss functions
        self.kl_temperature = kl_temp
        self.classification_criterion = nn.KLDivLoss(reduction='batchmean') # Use 'batchmean' for averaging like CE loss

        # Regularization parameter for NMF
        self.lambda_reg = 0.01 # Used in compute_projection_matrix_nmf

    # save_model remains the same
    def save_model(self, epoch):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"student_model_epoch_{epoch}.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'metrics': self.metrics
        }, save_path)
        self.logger.info(f"Model checkpoint saved at {save_path}")

    # extract_features_from_last_layer remains the same
    def extract_features_from_last_layer(self, model, x):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Simplified logic - assumes forward_features or fallback
        # Add more specific checks if needed based on exact model types
        hook_registered = False
        if hasattr(model, 'layer4'): target_layer, hook_name = model.layer4, 'layer4'
        elif hasattr(model, 'blocks') and len(model.blocks) > 0: target_layer, hook_name = model.blocks[-1], 'final_block'
        elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential) and len(model.features) > 0: target_layer, hook_name = model.features[-1], 'final_features'
        else: target_layer = None # Fallback handled below

        if target_layer:
            try:
                handle = target_layer.register_forward_hook(get_activation(hook_name))
                hook_registered = True
            except Exception as e:
                 self.logger.warning(f"Failed to register hook on {hook_name}: {e}. Using fallback.")
                 target_layer = None # Force fallback


        # Forward pass
        _ = model(x) # Trigger hook or run standard forward

        if hook_registered:
            features = activation[hook_name]
            handle.remove() # Remove hook
        else: # Fallback if hook failed or no layer identified
            self.logger.debug("Using fallback feature extraction method.")
            if hasattr(model, "forward_features"):
                features = model.forward_features(x)
            else:
                modules = list(model.children())[:-1] # Assume last child is classifier
                if not modules: # Handle simple models with no children list?
                     self.logger.error("Cannot extract features: Model has no children list.")
                     # Fallback to using the direct model output before classifier? Very brittle.
                     return model(x) # Or raise error
                feature_extractor = nn.Sequential(*modules)
                features = feature_extractor(x)

        # Global average pooling if features are 4D
        if features.ndim == 4:
            features = torch.mean(features, dim=[2, 3])

        # Detach features if not already detached (hooks usually detach)
        return features.detach()


    # compute_projection_matrix (unused least squares version) can be removed or kept
    # def compute_projection_matrix(...)

    # find_nmf_fixed_H remains the same
    def find_nmf_fixed_H(
        self,
        features, # Expects torch tensor or numpy array
        concept_bank: np.ndarray, # Expects numpy array [n_concepts, teacher_feat_dim]
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'mu',
        random_state: int = None,
        beta_loss: str = 'frobenius',
        verbose: bool = False
    ) -> np.ndarray: # Returns numpy array [n_samples, n_concepts]
        if not isinstance(concept_bank, np.ndarray):
            raise TypeError("concept_bank (W_teacher) must be a NumPy array.")

        # 1. Prepare features (target V)
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        elif isinstance(features, np.ndarray):
            features_np = features
        else:
            raise TypeError("features must be a NumPy array or PyTorch Tensor.")

        if np.any(features_np < 0):
            if verbose: warnings.warn("Input 'features' contains negative values. Clipping to 0.", UserWarning)
            features_np = np.maximum(features_np, 0)

        # 2. Prepare W_teacher (fixed H)
        if np.any(concept_bank < 0):
            if verbose: warnings.warn("Input 'concept_bank' contains negative values. Clipping to 0.", UserWarning)
            W_teacher_nonneg = np.maximum(concept_bank, 0)
        else:
            W_teacher_nonneg = concept_bank

        # 3. Check shapes
        n_samples, feat_dim_teacher = features_np.shape
        n_concepts, feat_dim_W = W_teacher_nonneg.shape
        if feat_dim_teacher != feat_dim_W:
            raise ValueError(f"Shape mismatch: features columns ({feat_dim_teacher}) != concept_bank columns ({feat_dim_W}).")
        if n_samples == 0:
             self.logger.warning("find_nmf_fixed_H called with 0 samples. Returning empty array.")
             return np.zeros((0, n_concepts), dtype=features_np.dtype)


        # 4. Initialize and run NMF
        nmf_model = NMF(
            n_components=n_concepts, init='custom', solver=solver, max_iter=max_iter,
            tol=tol, random_state=random_state, beta_loss=beta_loss, verbose=verbose > 1
        )
        # Initialize W (the matrix to find, U_teacher)
        rng = np.random.RandomState(random_state)
        # Ensure dtype consistency, float32 is common
        dtype = features_np.dtype
        W_init = rng.rand(n_samples, n_concepts).astype(dtype, copy=False)
        W_teacher_nonneg = W_teacher_nonneg.astype(dtype, copy=False)

        # Fit the model: find W (our U) given X (features) and fixed H (W_teacher)
        try:
            # Need to handle potential warnings or errors if fit fails (e.g., singular matrix)
             with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", message=".*positive.*", category=UserWarning) # Ignore sklearn NMF warnings about non-positivity if clipping happened
                 U = nmf_model.fit_transform(X=features_np, H=W_teacher_nonneg, W=W_init)
        except Exception as e:
             self.logger.error(f"NMF (find_nmf_fixed_H) failed: {e}")
             # Decide how to handle: re-raise, return zeros, etc.
             # Returning zeros might let training continue but silently fail on alignment
             return np.zeros((n_samples, n_concepts), dtype=dtype) # Example fallback


        if verbose:
            rec_err = nmf_model.reconstruction_err_
            print(f"NMF fixed H complete. Final reconstruction error: {rec_err:.4f}")

        return U # Shape: (n_samples, n_concepts)

    # concept_alignment_loss remains the same
    def concept_alignment_loss(self, student_features, W_star, teacher_concept_coeffs):
        # Ensure inputs are torch tensors on the correct device
        if isinstance(W_star, np.ndarray):
            W_star = torch.tensor(W_star, dtype=torch.float32, device=self.device)
        if isinstance(teacher_concept_coeffs, np.ndarray):
            teacher_concept_coeffs = torch.tensor(teacher_concept_coeffs, dtype=torch.float32, device=self.device)

        # Check for NaNs/Infs before matmul
        if torch.isnan(student_features).any() or torch.isinf(student_features).any():
            self.logger.warning("NaN/Inf detected in student_features before alignment loss.")
            return torch.tensor(0.0, device=self.device) # Or handle differently
        if torch.isnan(W_star).any() or torch.isinf(W_star).any():
            self.logger.warning("NaN/Inf detected in W_star before alignment loss.")
            return torch.tensor(0.0, device=self.device)
        if torch.isnan(teacher_concept_coeffs).any() or torch.isinf(teacher_concept_coeffs).any():
            self.logger.warning("NaN/Inf detected in teacher_concept_coeffs before alignment loss.")
            return torch.tensor(0.0, device=self.device)


        # Project student features to concept space: [B, S_feat] @ [S_feat, N_concepts] -> [B, N_concepts]
        student_concepts = torch.matmul(student_features, W_star)

        # Compute MSE loss: || student_concepts - teacher_concept_coeffs ||^2
        # Ensure shapes match: both should be [B, N_concepts]
        if student_concepts.shape != teacher_concept_coeffs.shape:
             self.logger.error(f"Shape mismatch in alignment loss: student_concepts {student_concepts.shape}, teacher_coeffs {teacher_concept_coeffs.shape}")
             # Attempt to recover or return zero loss? Returning zero might hide issues.
             return torch.tensor(0.0, device=self.device) # Be cautious with this

        # Use torch.nn.functional.mse_loss for stability and correctness
        alignment_loss = torch.nn.functional.mse_loss(student_concepts, teacher_concept_coeffs, reduction='mean')

        # Check for NaN loss
        if torch.isnan(alignment_loss):
             self.logger.warning("NaN detected in calculated alignment_loss.")
             return torch.tensor(0.0, device=self.device) # Prevent NaN propagation

        return alignment_loss


    # compute_projection_matrix_nmf remains the same
    def compute_projection_matrix_nmf(
        self,
        student_features: Union[np.ndarray, torch.Tensor], # A_S [m, n]
        target_coefficients: Union[np.ndarray, torch.Tensor], # U [m, k]
        lambda_reg: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'mu',
        random_state: int = None,
        verbose: bool = False
    ) -> np.ndarray: # Returns W* [n, k]
        # 1. Input Conversion and Validation
        if isinstance(student_features, torch.Tensor):
            A_S = student_features.detach().cpu().numpy()
        elif isinstance(student_features, np.ndarray): A_S = student_features
        else: raise TypeError("student_features must be NumPy or Tensor.")

        if isinstance(target_coefficients, torch.Tensor):
            U = target_coefficients.detach().cpu().numpy()
        elif isinstance(target_coefficients, np.ndarray): U = target_coefficients
        else: raise TypeError("target_coefficients must be NumPy or Tensor.")

        # Clip negatives (NMF requirement)
        if np.any(A_S < 0):
            if verbose: warnings.warn("Clipping negative values in student_features (A_S) to 0 for NMF.", UserWarning)
            A_S = np.maximum(A_S, 0)
        if np.any(U < 0):
            if verbose: warnings.warn("Clipping negative values in target_coefficients (U) to 0 for NMF.", UserWarning)
            U = np.maximum(U, 0)

        if A_S.shape[0] != U.shape[0]:
            raise ValueError(f"Shape mismatch: student_features samples {A_S.shape[0]} != target_coefficients samples {U.shape[0]}.")
        if lambda_reg < 0: raise ValueError("lambda_reg must be non-negative.")

        # Handle empty inputs
        if A_S.shape[0] == 0:
            self.logger.warning("compute_projection_matrix_nmf called with 0 samples. Returning zeros.")
            n_features = A_S.shape[1]
            k_targets = U.shape[1]
            return np.zeros((n_features, k_targets), dtype=A_S.dtype)


        # Data types
        dtype = A_S.dtype
        A_S = A_S.astype(dtype, copy=False)
        U = U.astype(dtype, copy=False)

        m, n_features = A_S.shape # m samples, n student features
        _, k_targets = U.shape    # m samples, k target coefficients/concepts

        # 2. Setup NMF: Solve U ≈ A_S @ W* (X ≈ W @ H => X=U, W=A_S(fixed), H=W*(learn))
        nmf_model = NMF(
            n_components=n_features, # Number of columns in FIXED W (A_S)
            init='custom', solver=solver, beta_loss='frobenius', max_iter=max_iter, tol=tol,
            random_state=random_state, l1_ratio=0.0,
            alpha_W=0.0,              # No regularization on fixed A_S
            alpha_H=2 * lambda_reg,   # L2 regularization on W* (H) = lambda * ||W*||^2
            verbose=verbose > 1
        )

        # Initialize H (W*)
        rng = np.random.RandomState(random_state)
        # H shape: (n_components, n_features_X) = (n_features_A_S, n_targets_U) = (n, k)
        H_init = rng.rand(n_features, k_targets).astype(dtype, copy=False)
        H_init = np.maximum(H_init, 1e-9) # Ensure positivity

        # Fit
        try:
            with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", message=".*init='custom'.*", category=FutureWarning)
                 warnings.filterwarnings("ignore", message=".*positive.*", category=UserWarning) # Ignore sklearn NMF warnings
                 # Fit using X=U, fixed W=A_S, initial H=H_init
                 nmf_model.fit(X=U, W=A_S, H=H_init)
            W_star = nmf_model.components_ # Learned H is our W*

            # Check for NaNs in result
            if np.isnan(W_star).any():
                 self.logger.warning("NaN detected in computed W_star. Returning zeros.")
                 return np.zeros((n_features, k_targets), dtype=dtype)


        except Exception as e:
             self.logger.error(f"NMF (compute_projection_matrix_nmf) failed: {e}")
             # Return zeros as fallback
             return np.zeros((n_features, k_targets), dtype=dtype)

        if verbose:
            rec_err = nmf_model.reconstruction_err_
            print(f"NMF projection complete. Final reconstruction error: {rec_err:.4f}")

        # W_star should be non-negative, but clip just in case of numerical issues
        return np.maximum(W_star, 0) # Shape: (n_features, k_targets)


    # --- Modified Training Loop ---
    def train(self, epochs, lr):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        # Consider adding a scheduler? e.g., CosineAnnealingLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(self.train_loader))

        start_time = time.time()
        self.logger.info(f"Starting training for {epochs} epochs using mixed-class batches.")

        total_steps = 0 # Use steps instead of batches if comparing runs

        self.metrics["batch_losses"] = [[] for _ in range(epochs)]

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"=== EPOCH {epoch+1}/{epochs} ===")
            self.student_model.train() # Ensure model is in training mode

            epoch_loss = 0.0
            epoch_class_loss = 0.0
            epoch_align_loss = 0.0
            batch_count = 0

            # Use the unified train_loader
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(progress_bar):
                images, labels = batch # batch contains images and labels from multiple classes
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                # print(np.unique(labels.cpu().numpy())) # Debugging line to check unique labels in batch

                optimizer.zero_grad()

                # 1. Forward pass through student model for classification
                student_outputs = self.student_model(images)

                # 2. Class loss using KL divergence
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)

                soft_targets = F.softmax(teacher_outputs / self.kl_temperature, dim=-1)
                log_soft_student_outputs = F.log_softmax(student_outputs / self.kl_temperature, dim=-1)
                class_loss = (self.kl_temperature ** 2) * self.classification_criterion(
                    log_soft_student_outputs,
                    soft_targets
                )

                # --- Concept Alignment Calculation (only if alpha > 0) ---
                alignment_loss = torch.tensor(0.0, device=self.device) # Default to zero
                if self.alpha > 0:
                    try:
                        # 3. Extract features (only if needed)
                        # Ensure teacher model is eval mode (should be set in __main__)
                        with torch.no_grad(): # Teacher features don't need gradients
                            teacher_features = self.extract_features_from_last_layer(self.teacher_model, images)

                        # Student features needed for both W* and alignment loss calculation
                        student_features = self.extract_features_from_last_layer(self.student_model, images)

                        # Check if feature extraction returned valid tensors
                        if torch.isnan(teacher_features).any() or torch.isnan(student_features).any():
                             self.logger.warning(f"NaN detected in features at step {total_steps}. Skipping alignment loss for this batch.")
                             # class_loss is still valid, proceed with alpha=0 essentially

                        else:
                            # 4a. Calculate U_teacher for each class group in the batch
                            unique_labels_in_batch = labels.unique()
                            batch_size = images.size(0)
                            # Initialize tensor to store U_teacher in correct batch order
                            batch_U_teacher = torch.zeros(batch_size, self.num_concepts, device=self.device, dtype=torch.float32)
                            valid_alignment_calc = True # Flag to track if NMF succeeded for all groups

                            for class_id_tensor in unique_labels_in_batch:
                                class_id = class_id_tensor.item()

                                # Check if we have concepts for this class
                                if class_id not in self.teacher_concepts:
                                    self.logger.warning(f"Skipping alignment for class {class_id} in batch {batch_idx} - concepts not loaded.")
                                    continue # Skip this class

                                mask = (labels == class_id_tensor)
                                indices_in_batch = mask.nonzero(as_tuple=False).squeeze(-1) # Get indices for this class

                                if len(indices_in_batch) == 0: continue # Should not happen with unique_labels, but safe check

                                # Get features for this group
                                teacher_features_group = teacher_features[mask]
                                # Get the corresponding teacher concepts (W_teacher)
                                W_teacher = self.teacher_concepts[class_id] # Numpy array

                                # Calculate U_teacher for this group
                                U_teacher_group_np = self.find_nmf_fixed_H(teacher_features_group, W_teacher, verbose=False) # Returns numpy array

                                # Check if NMF returned valid results
                                if U_teacher_group_np is None or np.isnan(U_teacher_group_np).any():
                                     self.logger.warning(f"NMF for U_teacher failed or returned NaN for class {class_id} in batch {batch_idx}. Skipping alignment loss.")
                                     valid_alignment_calc = False
                                     break # Stop processing this batch for alignment

                                # Convert to tensor and place in the correct batch positions
                                U_teacher_group = torch.tensor(U_teacher_group_np, dtype=torch.float32, device=self.device)
                                batch_U_teacher[indices_in_batch] = U_teacher_group

                            if valid_alignment_calc: # Proceed only if all groups' U_teacher were calculated
                                # 4b. Compute ONE projection matrix W* for the entire batch
                                # Needs student_features [B, S_feat], batch_U_teacher [B, N_concepts]
                                W_star_np = self.compute_projection_matrix_nmf(
                                    student_features,       # Torch tensor [B, S_feat]
                                    batch_U_teacher,        # Torch tensor [B, N_concepts]
                                    lambda_reg=self.lambda_reg,
                                    verbose=False
                                ) # Returns numpy array [S_feat, N_concepts]

                                # Check if NMF for W* returned valid results
                                if W_star_np is None or np.isnan(W_star_np).any():
                                     self.logger.warning(f"NMF for W* failed or returned NaN in batch {batch_idx}. Skipping alignment loss.")
                                     # alignment_loss remains 0
                                else:
                                    # 5. Compute alignment loss for the whole batch
                                    alignment_loss = self.concept_alignment_loss(
                                        student_features,   # [B, S_feat]
                                        W_star_np,          # Numpy [S_feat, N_concepts] -> converted inside function
                                        batch_U_teacher     # [B, N_concepts]
                                    )
                    except Exception as e:
                        self.logger.error(f"Error during alignment loss calculation at step {total_steps}: {e}", exc_info=True)
                        alignment_loss = torch.tensor(0.0, device=self.device) # Skip alignment for this batch on error

                # --- End Concept Alignment Calculation ---

                # 6. Compute total loss and backprop
                # Defensive check for NaN/Inf in losses before combining
                if torch.isnan(class_loss) or torch.isinf(class_loss):
                     self.logger.error(f"NaN/Inf detected in class_loss at step {total_steps}. Skipping batch.")
                     continue # Skip optimizer step if loss is invalid
                if torch.isnan(alignment_loss) or torch.isinf(alignment_loss):
                     self.logger.warning(f"NaN/Inf detected in alignment_loss at step {total_steps}. Setting alignment contribution to 0.")
                     alignment_loss = torch.tensor(0.0, device=self.device)


                total_loss = (1.0 - self.alpha) * class_loss + self.alpha * alignment_loss

                if torch.isnan(total_loss):
                     self.logger.error(f"NaN detected in total_loss at step {total_steps} even after checks. Skipping batch.")
                     continue


                total_loss.backward()

                # Optional: Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                # Update metrics
                epoch_loss += total_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_align_loss += alignment_loss.item() # This will be 0 if alpha=0 or if calculation failed
                batch_count += 1
                total_steps += 1

                # Log batch stats
                if batch_idx % self.log_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"Epoch {epoch+1}, Step {total_steps} [{batch_idx}/{len(self.train_loader)}], "
                        f"LR: {current_lr:.5f}, Loss: {total_loss.item():.4f} "
                        f"(Class: {class_loss.item():.4f}, Align: {alignment_loss.item():.4f})"
                    )
                    self.logger.info(log_msg)

                progress_bar.set_postfix(loss=total_loss.item())
                self.metrics["batch_losses"][epoch].append({
                    "total": total_loss.item(),
                    "alignment": alignment_loss.item(),
                    "classification": class_loss.item(),
                })

                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train/batch_loss": total_loss.item(),
                        "train/class_loss": class_loss.item(),
                        "train/alignment_loss": alignment_loss.item(), # Log the actual computed value
                        "train/learning_rate": current_lr,
                        "step": total_steps,
                        "epoch": epoch + (batch_idx / len(self.train_loader)) # Log fractional epoch
                    })

            progress_bar.close()

            # Epoch summary
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            avg_epoch_class_loss = epoch_class_loss / batch_count if batch_count > 0 else 0
            avg_epoch_align_loss = epoch_align_loss / batch_count if batch_count > 0 else 0
            self.metrics["epoch_losses"].append(avg_epoch_loss)

            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s.")
            self.logger.info(f"  Avg Loss: {avg_epoch_loss:.4f}")
            self.logger.info(f"  Avg Class Loss: {avg_epoch_class_loss:.4f}")
            self.logger.info(f"  Avg Align Loss: {avg_epoch_align_loss:.4f}")

            # Perform validation at the end of each epoch
            eval_results = self.evaluate(self.val_loader)
            # Optionally save model based on validation performance
            self.save_model(epoch + 1) # Save every epoch for now

            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_epoch_loss,
                    "train/epoch_class_loss": avg_epoch_class_loss,
                    "train/epoch_align_loss": avg_epoch_align_loss,
                    "epoch_time": epoch_time,
                    # Log validation metrics with epoch
                    "val/teacher_accuracy": eval_results["teacher_accuracy"],
                    "val/student_accuracy": eval_results["student_accuracy"],
                })

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        return self.metrics

    # evaluate remains the same
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
                total += labels.size(0)

                # Teacher predictions
                teacher_outputs = self.teacher_model(images)
                _, teacher_preds = torch.max(teacher_outputs, 1)
                teacher_correct += (teacher_preds == labels).sum().item()

                # Student predictions
                student_outputs = self.student_model(images)
                _, student_preds = torch.max(student_outputs, 1)
                student_correct += (student_preds == labels).sum().item()

        teacher_accuracy = 100.0 * teacher_correct / total if total > 0 else 0
        student_accuracy = 100.0 * student_correct / total if total > 0 else 0

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Teacher Accuracy: {teacher_accuracy:.2f}%")
        self.logger.info(f"  Student Accuracy: {student_accuracy:.2f}%")

        # No wandb logging here, done in train loop after evaluate call

        return {
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy
        }


# parse_args remains the same (ensure --alpha is added)
def parse_args():
    parser = argparse.ArgumentParser(description="Concept Distillation Training Pipeline")
    
    # Dataset and paths
    parser.add_argument("--pkl-dir", type=str, default="/scratch/swayam/rsvc-exps/outputs/data/dn=in_spl=val_ni=100_seed=0/r50_ckpt=None/ps=64_flv=v1_igs=c/dm=nmf_nc=10_seed=0/concepts/layer4.2.act3", 
                        help="Directory containing teacher concept pickle files")
    parser.add_argument("--mapping-file", type=str, default="restructured_map_val.npy", 
                        help="Path to class mapping file")
    parser.add_argument("--save-dir", type=str, default="/scratch/swayam/rsvc_model_trains", 
                        help="Path where model is stored during training")
    parser.add_argument("--data-root", type=str, default="/scratch/swayam/imagenet_data/imagenet",
                        help="Root directory for ImageNet dataset")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to use (train or val)")
    
    # Models
    parser.add_argument("--teacher-model", type=str, default="resnet50.a2_in1k",
                        help="Teacher model architecture from timm")
    parser.add_argument("--student-model", type=str, default="resnet18.a2_in1k",
                        help="Student model architecture from timm")
    parser.add_argument("--timm-cache", type=str, default="/scratch/swayam/timm_cache/",
                        help="Cache directory for timm models")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--alpha", type=float, required=True,
                        help="Parameter used to weigh alignment loss")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    
    # Logging parameters
    parser.add_argument("--log-dir", type=str, default="/scratch/swayam/rsvc_logs",
                        help="Directory to save logs")
    parser.add_argument("--log-interval", type=int, default=250,
                        help="Interval for logging training metrics")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="concept-distillation",
                        help="wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="wandb run name")
    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting concept distillation pipeline")
    logger.info(f"Arguments: {args}")
    
    # Initialize wandb if requested
    if args.use_wandb:
        if WANDB_AVAILABLE:
            logger.info("Initializing wandb")
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"distill_{args.teacher_model}_to_{args.student_model}",
                config=vars(args)
            )
        else:
            logger.warning("wandb logging requested but wandb is not available")
    
    # Load models
    logger.info(f"Loading teacher model: {args.teacher_model}")
    teacher_model = timm.create_model(
        args.teacher_model,
        pretrained=True,
        cache_dir=args.timm_cache
    ).eval().requires_grad_(False).to(args.device)
    
    logger.info(f"Loading student model: {args.student_model}")
    student_model = timm.create_model(
        args.student_model,
        pretrained=True,
        cache_dir=args.timm_cache
    ).train().to(args.device)
    
    # Setup data transformation
    config = resolve_data_config({}, model=teacher_model)
    transform = create_transform(**config)
    
    # Load dataset
    logger.info(f"Loading ImageNet dataset from {args.data_root}, split={args.split}")
    base_dataset = datasets.ImageNet(
        root=args.data_root, 
        split=args.split, 
        transform=transform
    )
    
    # Create validation dataloader for evaluation
    val_loader = DataLoader(
        base_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize trainer
    trainer = ConceptDistillationTrainer(
        pkl_dir=args.pkl_dir,
        mapping_file=args.mapping_file,
        base_dataset=base_dataset,
        teacher_model=teacher_model,
        student_model=student_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        log_interval=args.log_interval,
        use_wandb=args.use_wandb,
        save_dir=args.save_dir,
        val_loader=val_loader,
        alpha=args.alpha,
    )
    
    # Train
    logger.info("Starting training")
    metrics = trainer.train(epochs=args.epochs, lr=args.lr)
    
    # Evaluate
    logger.info("Starting evaluation")
    eval_results = trainer.evaluate(val_loader)
    
    logger.info("Pipeline completed successfully")
    
    # Close wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()