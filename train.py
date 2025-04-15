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
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from typing import Union
from sklearn.decomposition import NMF


# Optional wandb import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable wandb logging.")

# Configure logging
def setup_logging(log_dir, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('concept_distillation')

class SingleClassDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


class ConceptDistillationTrainer:
    def __init__(
        self,
        pkl_dir,
        mapping_file,
        base_dataset,
        teacher_model,
        student_model,
        save_dir,
        val_loader,
        alpha,
        batch_size=32,
        num_workers=0,
        device="cuda",
        log_interval=10,
        use_wandb=False
    ):
        self.logger = logging.getLogger('concept_distillation')
        self.num_classes = 1000
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.save_dir = save_dir
        self.val_loader = val_loader
        self.alpha = alpha # Weight for alignment loss
        
        # Set models
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        
        self.logger.info(f"Initialized teacher model: {type(teacher_model).__name__}")
        self.logger.info(f"Initialized student model: {type(student_model).__name__}")

        # Load teacher concepts
        self.logger.info(f"Loading teacher concepts from {pkl_dir}")
        self.teacher_concepts = {}
        for class_id in range(self.num_classes):
            pkl_path = os.path.join(pkl_dir, f"{class_id}.pkl")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            self.teacher_concepts[class_id] = data["W"]
        self.logger.info(f"Loaded concepts for {len(self.teacher_concepts)} classes")

        # Load dataset mappings
        self.logger.info(f"Loading class mappings from {mapping_file}")
        mapping_arr = np.load(mapping_file, allow_pickle=True).item()

        # Create per-class datasets and dataloaders
        self.datasets = {}
        self.dataloaders = {}
        for class_id in range(self.num_classes):
            indices_for_class = mapping_arr[class_id]
            self.datasets[class_id] = SingleClassDataset(base_dataset, indices_for_class)

            loader = DataLoader(
                self.datasets[class_id],
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers
            )
            self.dataloaders[class_id] = iter(loader)

        self.OrigRemLen = {
            class_id: len(mapping_arr[class_id]) for class_id in range(self.num_classes)
        }
        self.dataset_size = sum(self.OrigRemLen.values())
        self.logger.info(f"Total dataset size: {self.dataset_size} samples")

        # Initialize metrics tracker
        self.metrics = {
            "epoch_losses": [],
            "batch_losses": []
        }
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # Regularization parameter for NMF
        self.lambda_reg = 0.01
        
    def save_model(self, epoch):
        """
        Save the student model checkpoint
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"student_model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'metrics': self.metrics
        }, save_path)
        self.logger.info(f"Model checkpoint saved at {save_path}")
        
        # if self.use_wandb and WANDB_AVAILABLE:
        #     wandb.save(save_path)
        #     wandb.save(latest_path)

    def extract_features_from_last_layer(self, model, x):
        """
        Extract features from the model's last layer before classification
        using a hook-based approach similar to extract_model_activations.py.
        
        This allows for more flexibility in choosing which layer's activations to use.
        
        Args:
            x: Input tensor
            model: The model to extract features from
        """
        # Use a hook to capture the output of the specified layer
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Identify the layer to extract features from
        # This could be customized based on the model architecture
        if hasattr(model, 'layer4'):  # ResNet-like architecture
            target_layer = model.layer4
            hook_name = 'layer4'
        elif hasattr(model, 'blocks') and len(model.blocks) > 0:  # ViT-like architecture
            target_layer = model.blocks[-1]
            hook_name = 'final_block'
        elif hasattr(model, 'features'):  # Many CNN architectures
            target_layer = model.features[-1]
            hook_name = 'final_features'
        else:
            # Fallback to the original implementation
            self.logger.warning("Could not identify target layer for hooks, using fallback method")
            if hasattr(model, "forward_features"):
                return model.forward_features(x)
            else:
                # This assumes a standard architecture where the final layer is a classifier
                modules = list(model.children())
                feature_extractor = nn.Sequential(*modules[:-1])
                features = feature_extractor(x)
                
                # Handle different output shapes
                if len(features.shape) == 4:  # If features are [B, C, H, W]
                    features = torch.mean(features, dim=[2, 3])  # Global average pooling
                
                return features
            
        # Register the hook
        handle = target_layer.register_forward_hook(get_activation(hook_name))
        
        # Forward pass to trigger the hook
        _ = model(x)
        
        # Get the captured features
        features = activation[hook_name]
        
        # If features are 4D (B, C, H, W), apply global average pooling
        if len(features.shape) == 4:
            features = torch.mean(features, dim=[2, 3])
        
        # Remove the hook to prevent memory leaks
        handle.remove()
        
        return features
        
    
    def compute_projection_matrix(self, student_features, teacher_concepts, lambda_reg):
        """
        Compute the optimal projection matrix W* using Non-negative Matrix Factorization (NMF).
        
        Solves: W* = argmin ||A_S * W* - U||^2 + lambda * ||W*||^2
        
        This is a regularized least squares problem which has the closed-form solution:
        W* = (A_S^T * A_S + lambda * I)^(-1) * A_S^T * U
        
        Args:
            student_features (torch.Tensor): Student model activations [batch_size, feature_dim]
            teacher_concepts (torch.Tensor): Teacher concepts [concept_dim, ...]
            lambda_reg (float): Regularization parameter
            
        Returns:
            torch.Tensor: Projection matrix W*
        """
        # Ensure teacher_concepts is on the correct device
        teacher_concepts = torch.tensor(teacher_concepts, dtype=torch.float32, device=self.device)
        
        # Get dimensions
        batch_size, student_dim = student_features.shape
        concept_dim = teacher_concepts.shape[0]
        
        # Compute the projection matrix using regularized least squares
        # (A^T * A + lambda * I)^(-1) * A^T * b
        AtA = torch.matmul(student_features.t(), student_features)
        reg_term = lambda_reg * torch.eye(student_dim, device=self.device)
        inv_term = torch.inverse(AtA + reg_term)
        AtB = torch.matmul(student_features.t(), teacher_concepts)
        W_star = torch.matmul(inv_term, AtB)
        
        return W_star
    
    def find_nmf_fixed_H(
        self,
        features,
        concept_bank: np.ndarray,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = 'mu', # 'mu' or 'cd'
        random_state: int = None,
        beta_loss: str = 'frobenius', # 'frobenius', 'kullback-leibler', 'itakura-saito'
        verbose: bool = False # Add verbosity control
    ) -> np.ndarray:
        """
        Finds the non-negative matrix U such that teacher_features ≈ U @ W_teacher
        using Non-negative Matrix Factorization (NMF), keeping W_teacher fixed.

        Args:
            teacher_features: The target matrix V (batch_size, teacher_feat_dim).
                            Can be a NumPy array or a PyTorch Tensor.
                            Values should ideally be non-negative.
            W_teacher: The fixed factor matrix H (num_concepts, teacher_feat_dim).
                    Must be a NumPy array and non-negative.
            max_iter: Maximum number of iterations for the NMF solver.
            tol: Tolerance of the stopping condition.
            solver: Numerical solver to use ('mu' or 'cd').
            random_state: Seed for random number generation (affects initialization if needed).
            beta_loss: String ('frobenius', 'kullback-leibler', 'itakura-saito').
                    Beta divergence to be minimized.
            verbose: If True, prints status messages and warnings.

        Returns:
            np.ndarray: The calculated non-negative activation matrix U
                        (batch_size, num_concepts).

        Raises:
            ValueError: If the dimensions of teacher_features and W_teacher are incompatible
                        for matrix multiplication after finding U.
            TypeError: If W_teacher is not a NumPy array.
        """
        if not isinstance(concept_bank, np.ndarray):
            raise TypeError("W_teacher must be a NumPy array.")

        # 1. Prepare teacher_features (convert to NumPy, ensure non-negative)
        if isinstance(features, torch.Tensor):
            if verbose:
                print("Converting teacher_features from PyTorch Tensor to NumPy array.")
            teacher_features_np = features.detach().cpu().numpy()
        elif isinstance(features, np.ndarray):
            teacher_features_np = features
        else:
            raise TypeError("teacher_features must be a NumPy array or PyTorch Tensor.")

        if np.any(teacher_features_np < 0):
            if verbose:
                warnings.warn("Input 'teacher_features' contains negative values. "
                            "Clipping to 0 for NMF.", UserWarning)
            teacher_features_np = np.maximum(teacher_features_np, 0) # Ensure non-negativity

        # 2. Prepare W_teacher (ensure non-negative)
        if np.any(concept_bank < 0):
            if verbose:
                warnings.warn("Input 'W_teacher' contains negative values. "
                            "Clipping to 0 for NMF.", UserWarning)
            W_teacher_nonneg = np.maximum(concept_bank, 0)
        else:
            W_teacher_nonneg = concept_bank

        # Add a small epsilon if using 'mu' solver and W_teacher might have zeros,
        # as division by zero can occur in MU updates if H has zeros.
        # Or ensure the input W_teacher doesn't have problematic zero rows/columns.
        # For simplicity here, we'll rely on sklearn's handling or suggest ensuring
        # W_teacher is well-behaved if issues arise.
        # if solver == 'mu':
        #     W_teacher_nonneg = np.maximum(W_teacher_nonneg, 1e-9) # Optional epsilon
        # 3. Check shapes
        n_samples, feat_dim_teacher = teacher_features_np.shape
        n_concepts, feat_dim_W = W_teacher_nonneg.shape

        if feat_dim_teacher != feat_dim_W:
            raise ValueError(
                f"Shape mismatch: teacher_features columns ({feat_dim_teacher}) "
                f"must match W_teacher columns ({feat_dim_W})."
            )

        # 4. Initialize and run NMF
        nmf_model = NMF(
            n_components=n_concepts,
            init='custom',
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            beta_loss=beta_loss,
            verbose=verbose > 1 # Pass higher verbosity to sklearn NMF
        )

        if verbose:
            print(f"Running NMF (fixed H) to find U ({n_samples}, {n_concepts})...")
            
            
        rng = np.random.RandomState(random_state)
        W_init = rng.rand(n_samples, n_concepts) # Shape (batch_size, num_concepts)
        # Ensure dtype matches others, float32 often used in ML
        W_init = W_init.astype(teacher_features_np.dtype, copy=False)
        W_teacher_nonneg = W_teacher_nonneg.astype(teacher_features_np.dtype, copy=False)

        # Fit the model: finds W (our U) given X (teacher_features) and fixed H (W_teacher)
        # print(teacher_features_np.shape, W_teacher_nonneg.shape) # (32, 2048) (10, 2048)
        U = nmf_model.fit_transform(X=teacher_features_np, H=W_teacher_nonneg, W=W_init)

        if verbose:
            print("NMF complete.")
            # Optional: Report reconstruction error
            reconstruction_error = nmf_model.reconstruction_err_
            print(f"Final reconstruction error: {reconstruction_error:.4f}")

        # nmf_model.components_ should be very close to W_teacher_nonneg
        # U now holds the non-negative activations

        return U # Shape: (n_samples, n_concepts) i.e. (batch_size, num_concepts)
    
    def concept_alignment_loss(self, student_features, W_star, teacher_concept_coeffs):
        """
        Compute the alignment loss between student concepts and teacher concepts
        
        Args:
            student_features (torch.Tensor): Student model activations
            W_star (torch.Tensor): Projection matrix
            teacher_concept_coeffs (torch.Tensor): Teacher concepts
            
        Returns:
            torch.Tensor: Alignment loss
        """
        # Project student features to concept space
        student_concepts = torch.matmul(student_features, W_star)
        
        # Compute MSE loss between student concepts and teacher concepts
        alignment_loss = torch.mean(torch.square(student_concepts - teacher_concept_coeffs))
        
        return alignment_loss
    
    def compute_projection_matrix_nmf(
        self,
        student_features: Union[np.ndarray, torch.Tensor],
        target_coefficients: Union[np.ndarray, torch.Tensor],
        lambda_reg: float = 0.01,
        max_iter: int = 1000, # NMF iterations
        tol: float = 1e-4,    # NMF tolerance
        solver: str = 'mu',     # NMF solver ('mu' or 'cd')
        random_state: int = None,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Computes the optimal non-negative projection matrix W* that maps
        student_features to target_coefficients using NMF with L2 regularization.

        Solves: W* = argmin_{W>=0} ||A_S @ W - U||^2 + lambda * ||W||^2
                using sklearn.decomposition.NMF by fixing A_S and finding W*.

        Where A_S = student_features, U = target_coefficients, W = W*.

        Args:
            student_features (A_S): Input features (m samples, n features).
                                    NumPy array or PyTorch Tensor. MUST BE NON-NEGATIVE for NMF.
            target_coefficients (U): Target coefficients (m samples, k targets).
                                     NumPy array or PyTorch Tensor. MUST BE NON-NEGATIVE.
            lambda_reg: L2 regularization strength (lambda). Must be non-negative.
            max_iter: Maximum number of iterations for the NMF solver.
            tol: Tolerance of the stopping condition for NMF.
            solver: Numerical solver for NMF ('mu', 'cd').
            random_state: Seed for NMF initialization.
            verbose: If True, prints status messages.

        Returns:
            np.ndarray: The computed non-negative projection matrix W* (n features, k targets).

        Raises:
            ValueError: If shapes are incompatible, lambda_reg is negative, or inputs negative.
            TypeError: If inputs are not NumPy arrays or PyTorch Tensors.
        """
        # 1. Input Conversion and Validation
        if isinstance(student_features, torch.Tensor):
            if verbose: print("Converting student_features to NumPy.")
            A_S = student_features.detach().cpu().numpy()
        elif isinstance(student_features, np.ndarray):
            A_S = student_features
        else:
            raise TypeError("student_features must be a NumPy array or PyTorch Tensor.")

        if isinstance(target_coefficients, torch.Tensor):
            if verbose: print("Converting target_coefficients to NumPy.")
            U = target_coefficients.detach().cpu().numpy()
        elif isinstance(target_coefficients, np.ndarray):
            U = target_coefficients
        else:
            raise TypeError("target_coefficients must be a NumPy array or PyTorch Tensor.")

        # --- NMF Requirement: Check for Non-Negativity ---
        if np.any(A_S < 0):
            warnings.warn("Input 'student_features' (A_S) contains negative values. "
                          "Clipping to 0 for NMF. This might affect the result "
                          "if negative values were meaningful.", UserWarning)
            A_S = np.maximum(A_S, 0)

        if np.any(U < 0):
            # U should be non-negative from the previous NMF step, but check anyway
            warnings.warn("Input 'target_coefficients' (U) contains negative values. "
                          "Clipping to 0 for NMF.", UserWarning)
            U = np.maximum(U, 0)
        # --- End Non-Negativity Check ---

        if A_S.shape[0] != U.shape[0]:
            raise ValueError(f"Incompatible shapes: student_features has {A_S.shape[0]} samples, "
                             f"but target_coefficients has {U.shape[0]} samples.")

        if lambda_reg < 0:
            raise ValueError("lambda_reg (regularization strength) must be non-negative.")

        # Data types
        dtype = A_S.dtype # Match input type if possible, NMF handles dtypes
        A_S = A_S.astype(dtype, copy=False)
        U = U.astype(dtype, copy=False)

        m, n_features = A_S.shape # m samples, n student features
        _, k_targets = U.shape    # m samples, k target coefficients/concepts

        if verbose:
            print(f"Setting up NMF to find W* ({n_features}, {k_targets}) solving U ≈ A_S @ W*")
            print(f"Shapes: U=({m}, {k_targets}), A_S=({m}, {n_features})")
            print(f"Regularization: lambda={lambda_reg} -> alpha_H={2 * lambda_reg}")

        # 2. Setup and Run NMF
        # We want U ≈ A_S @ W*
        # Map to NMF: X ≈ W @ H  =>  X=U, W=A_S (fixed), H=W* (learn)
        nmf_model = NMF(
            n_components=n_features,  # Number of columns in FIXED W (A_S)
            init='custom',            # We provide the fixed W (A_S) and initial H (W*)
            solver=solver,
            beta_loss='frobenius',    # Matches least squares objective ||A_S @ W* - U||^2
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            l1_ratio=0.0,             # Use L2 regularization
            alpha_W=0.0,              # No regularization on the fixed A_S
            alpha_H=2 * lambda_reg,   # L2 regularization on W* (H) = lambda * ||W*||^2
            verbose=verbose > 1
        )

        # Create an initial guess for H (our W*)
        # NMF needs H to be initialized when init='custom' and update_H=True
        rng = np.random.RandomState(random_state)
        # Shape of H must be (n_components, n_features_X) = (n_features, k_targets)
        H_init = rng.rand(n_features, k_targets).astype(dtype, copy=False)
        H_init = np.maximum(H_init, 1e-9) # Ensure positivity for some solvers

        if verbose:
            print(f"Running NMF with fixed W (A_S), finding H (W*) shape {H_init.shape}...")

        # Fit the model. Finds H (W*) given X (U) and fixed W (A_S).
        # Use fit() then access components_, as fit_transform behavior might be ambiguous here.
        try:
            with warnings.catch_warnings():
                 # Filter potential FutureWarning about init='custom'
                 warnings.filterwarnings("ignore", message=".*init='custom'.*", category=FutureWarning)
                 nmf_model.fit(X=U, W=A_S, H=H_init) # Pass fixed W=A_S, initial H=H_init

            W_star = nmf_model.components_ # The learned H matrix is our W*
            if verbose:
                print("NMF fitting complete.")
                rec_err = nmf_model.reconstruction_err_
                if rec_err is not None and np.isfinite(rec_err):
                    print(f"Final reconstruction error: {rec_err:.4f}")
                else:
                    print(f"Final reconstruction error: {rec_err}")


        except Exception as e:
             print(f"Error during NMF fitting: {e}")
             # Optionally re-raise or return None/raise specific error
             raise e

        # W_star should already be non-negative due to NMF constraints
        if verbose and np.any(W_star < 0):
             # This should theoretically not happen with standard NMF solvers
             print("Warning: NMF resulted in negative values in W* (unexpected). Clipping.")
             W_star = np.maximum(W_star, 0)


        return W_star # Shape: (n_features, k_targets)

    def train(self, epochs=1, lr=0.001):
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        
        start_time = time.time()
        self.logger.info(f"Starting training for {epochs} epochs")
        
        total_batches = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.logger.info(f"=== EPOCH {epoch+1}/{epochs} ===")

            RemLen = dict(self.OrigRemLen)
            valid_class_ids = list(RemLen.keys())
            
            epoch_loss = 0.0
            batch_count = 0

            num_batches = sum(math.ceil(len(self.datasets[class_id]) / self.batch_size) for class_id in valid_class_ids)
            progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}")

            while valid_class_ids:
                chosen_class = random.choice(valid_class_ids)

                try:
                    batch = next(self.dataloaders[chosen_class])
                except StopIteration:
                    self.logger.error(
                        f"DataLoader for class {chosen_class} exhausted! "
                        "Please check your dataset and mapping logic once again."
                    )
                    raise RuntimeError(
                        f"DataLoader for class {chosen_class} exhausted! "
                        "Please check your dataset and mapping logic once again."
                    )

                # Get images and move to device
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                # Get teacher concept vectors for the chosen class
                W_teacher = self.teacher_concepts[chosen_class]
                
                # Training procedure implementation
                self.student_model.train()
                optimizer.zero_grad()
                
                # 1. Forward pass through student model for classification
                student_outputs = self.student_model(images)
                
                # 2. Get the CE loss between predicted and actual class
                class_loss = self.classification_criterion(student_outputs, labels)
                
                # 3. Get the activations of the student model at the last layer
                student_features = self.extract_features_from_last_layer(self.student_model, images)
                teacher_features = self.extract_features_from_last_layer(self.teacher_model, images)
                
                # 4a. Find teacher coeff matrix
                U_teacher = self.find_nmf_fixed_H(teacher_features, W_teacher, verbose=False)
                
                # 4b. Compute the best projection matrix using regularized least squares
                """
                Here, I wish to extract a non-negative matrix W* such that:
                
                Compute the optimal projection matrix W* using Non-negative Matrix Factorization (NMF).
                
                Solves: W* = argmin ||A_S * W* - U||^2 + lambda * ||W*||^2
                
                """
                W_star = self.compute_projection_matrix_nmf(
                    student_features, 
                    U_teacher,
                    verbose=False
                )
                """
                print(student_outputs.shape) # 32, 1000 -> 32 batch size with 1000 outputs; torch tensor
                print(student_features.shape) # 32, 512 -> 32 batch size with 512 dim features; torch tensor
                print(teacher_features.shape) # 32, 2048; torch tensor
                print(W_teacher.shape) # 10, 2048 -> 10 concepts of size 2048; np array
                print(U_teacher.shape) # 32, 10 -> coefficient matrix; np array
                print(W_star.shape) # 512, 10 -> dimension of student * number of concepts; np array  
                """              
                
                # 5. Compute alignment loss 
                alignment_loss = self.concept_alignment_loss(
                    student_features,
                    torch.tensor(W_star).to(device=self.device),
                    torch.tensor(U_teacher).to(device=self.device),
                )
                
                # 6. Compute total loss and backprop
                total_loss = (1 - self.alpha)*class_loss + self.alpha * alignment_loss
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                batch_count += 1
                total_batches += 1
                
                # Log batch stats
                if batch_count % self.log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}, Batch {batch_count}/{num_batches}, "
                        f"Loss: {total_loss.item():.4f} "
                        f"(Class: {class_loss.item():.4f}, Align: {alignment_loss.item():.4f})"
                    )
                    self.metrics["batch_losses"].append(total_loss.item())
                    
                    if self.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "batch_loss": total_loss.item(),
                            "class_loss": class_loss.item(),
                            "alignment_loss": alignment_loss.item(),
                            "batch": total_batches,
                        })

                # Update remaining samples counter
                RemLen[chosen_class] = max(0, RemLen[chosen_class] - self.batch_size)
                if RemLen[chosen_class] == 0:
                    valid_class_ids.remove(chosen_class)
                
                progress_bar.update(1)

            progress_bar.close()
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            self.metrics["epoch_losses"].append(avg_epoch_loss)
            
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_epoch_loss:.4f}")
            
            self.save_model(epoch + 1)
            self.evaluate(self.val_loader)
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch_loss": avg_epoch_loss,
                    "epoch_time": epoch_time,
                })
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        return self.metrics

    def evaluate(self, val_loader):
        """
        Evaluate both teacher and student models on validation data
        """
        self.logger.info("Starting evaluation...")
        
        # Set models to evaluation mode
        self.teacher_model.eval()
        self.student_model.eval()
        
        teacher_correct = 0
        student_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Teacher predictions
                teacher_outputs = self.teacher_model(images)
                _, teacher_preds = torch.max(teacher_outputs, 1)
                teacher_correct += (teacher_preds == labels).sum().item()
                
                # Student predictions
                student_outputs = self.student_model(images)
                _, student_preds = torch.max(student_outputs, 1)
                student_correct += (student_preds == labels).sum().item()
                
                total += labels.size(0)
        
        teacher_accuracy = 100 * teacher_correct / total
        student_accuracy = 100 * student_correct / total
        
        self.logger.info(f"Evaluation results:")
        self.logger.info(f"Teacher accuracy: {teacher_accuracy:.2f}%")
        self.logger.info(f"Student accuracy: {student_accuracy:.2f}%")
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "teacher_accuracy": teacher_accuracy,
                "student_accuracy": student_accuracy
            })
        
        return {
            "teacher_accuracy": teacher_accuracy,
            "student_accuracy": student_accuracy
        }


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
    parser.add_argument("--lr", type=float, default=0.001,
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