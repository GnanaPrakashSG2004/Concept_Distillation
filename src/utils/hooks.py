import torch


class ActivationHook:
    """
    Class to register forward hooks on specified layers of a PyTorch model
    and store their activations during forward passes.
    """

    def __init__(self, move_to_cpu_in_hook=False, move_to_cpu_every=None):
        """
        Initializes the ActivationHook.

        Args:
            move_to_cpu_in_hook (bool): If True, move layer activations to CPU within the hook function itself.
            move_to_cpu_every (int): Move layer activations to CPU every N batches to manage GPU memory.
        """
        # Dictionary to store layer activations, keys are layer names, values are lists of activation tensors
        self.layer_activations = {}
        # List to store handles of registered hooks, used for removing hooks later
        self.handles = []
        # List to store layer names for which hooks are registered
        self.layer_names = None
        # Flag to indicate whether to move activations to CPU within the hook function
        self.move_to_cpu_in_hook = move_to_cpu_in_hook
        # Interval to move activations to CPU for memory management
        self.move_to_cpu_every = move_to_cpu_every
        # Dictionary to track indices of activations currently on GPU for each layer
        self.layer_gpu_indices = {}
        # Dictionary to count activations recorded for each layer
        self.layer_count = {}

    def hook_fn_factory(self, layer_name, post_activation_fn=None):
        """
        Creates and returns a hook function for a specific layer.

        Args:
            layer_name (str): Name of the layer for which the hook is created.
            post_activation_fn (function, optional): Function to apply to the layer's output after activation. Defaults to None.

        Returns:
            function: Hook function to be registered to a layer.
        """

        def hook_fn(module, input, output):
            """
            Hook function that gets executed during forward pass.

            Args:
                module (torch.nn.Module): Layer module on which the hook is registered.
                input (tuple): Input to the layer.
                output (torch.Tensor): Output from the layer.
            """
            # Detach and clone the output tensor to avoid modifying gradients and computation graph
            if self.move_to_cpu_in_hook:
                # Move to cpu immediately in hook if specified
                x = output.detach().clone().cpu()
            else:
                # Keep on the current device (GPU if model is on GPU)
                x = output.detach().clone()

            # Apply post-activation function if provided
            if post_activation_fn is not None:
                x = post_activation_fn(x)
            # Append the activation tensor to the list associated with the layer name
            self.layer_activations[layer_name].append(x)

            # Record the index of the current activation to manage CPU offloading
            self.layer_gpu_indices[layer_name].append(self.layer_count[layer_name])
            # Increment the activation count for this layer
            self.layer_count[layer_name] += 1

            # Move activations to CPU memory if move_to_cpu_every is set and condition is met
            self.move_to_cpu(layer_name)

        return hook_fn

    def move_to_cpu(self, layer_name, finish=False):
        """
        Moves layer activations from GPU to CPU based on `move_to_cpu_every` interval.

        Args:
            layer_name (str): Name of the layer whose activations are to be moved.
            finish (bool, optional): If True, move all remaining activations to CPU regardless of count. Defaults to False.
        """
        # Check if move_to_cpu_every is set and if the count condition is met or if finishing up
        if self.move_to_cpu_every and (
            len(self.layer_gpu_indices[layer_name]) == self.move_to_cpu_every or finish
        ):
            # Iterate through the indices of activations to be moved to CPU
            for i in self.layer_gpu_indices[layer_name]:
                # Move the activation tensor at index i to CPU
                self.layer_activations[layer_name][i] = self.layer_activations[
                    layer_name
                ][i].cpu()
            # Reset the list of GPU indices for this layer as they are now on CPU
            self.layer_gpu_indices[layer_name] = []

    def register_hooks(self, layer_names, layer_modules, post_activation_fn=None):
        """
        Registers forward hooks to the specified layers.

        Args:
            layer_names (list): List of layer names to register hooks on.
            layer_modules (list): List of layer modules corresponding to layer_names.
            post_activation_fn (function, optional): Function to apply to the layer's output after activation. Defaults to None.
        """
        # Ensure that the number of layer names and modules match
        assert len(layer_modules) == len(layer_names)

        # Store layer names in the hook object
        self.layer_names = layer_names
        # Initialize storage for activations, GPU indices, and counts for each layer
        for name in layer_names:
            self.layer_activations[name] = []
            self.layer_gpu_indices[name] = []
            self.layer_count[name] = 0

        # Register hook for each layer module
        for name, module in zip(self.layer_names, layer_modules):
            # Create hook function using factory
            hook_fn = self.hook_fn_factory(name, post_activation_fn=post_activation_fn)
            # Register forward hook and get the handle
            handle = module.register_forward_hook(hook_fn)
            # Store the handle for later removal
            self.handles.append(handle)

    def remove_hooks(self):
        """
        Removes all registered forward hooks.
        """
        # Iterate through all registered hook handles
        for handle in self.handles:
            # Remove each hook
            handle.remove()

    def concatenate_layer_activations(self):
        """
        Concatenates collected activations for each layer into a single tensor.
        Moves all remaining activations to CPU before concatenation.
        """
        # Iterate through layer names
        for name in self.layer_names:
            # Ensure all remaining activations are moved to CPU before final concatenation
            self.move_to_cpu(name, finish=True)
            # Concatenate list of activation tensors along dimension 0 (batch dimension) and move to CPU
            self.layer_activations[name] = torch.cat(
                self.layer_activations[name], dim=0
            ).cpu()

    def reset_activation_dict(self):
        """
        Resets the activation dictionary, clearing stored activations and counters.
        Prepare for collecting new activations without interference from previous ones.
        """
        # Reset activation list, gpu indices and counts for each layer to empty state
        for name in self.layer_names:
            self.layer_activations[name] = []
            self.layer_gpu_indices[name] = []
            self.layer_count[name] = 0
