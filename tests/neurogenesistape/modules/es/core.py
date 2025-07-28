"""Core Evolution Strategies (ES) components for NeurogenesisTape."""

import jax
from jax import numpy as jnp
from flax import nnx

# Import shared evolutionary components
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
from neurogenesistape.modules.evolution import EvoModule


class ES_Module(EvoModule):
    """Base class for Evolution Strategies modules."""

    noise_sigma: float = 0.1
    enable_sigma_decay: bool = False
    min_sigma: float = 0.01
    
    def set_sigma_decay(self, enabled: bool = True):
        """Set sigma decay for this ES module.
        
        Args:
            enabled: Whether to enable sigma decay (default: True)
        """
        self.enable_sigma_decay = enabled


class ES_Tape(ES_Module):
    """Evolutionary parameter tensor with gradient estimation capabilities."""

    def __init__(self, shape, rngs: nnx.rnglib.Rngs, dtype=jnp.float32):
        """Initialize an ES_Tape with the given shape.
        
        Args:
            shape: Shape of the parameter tensor
            rngs: Random number generator keys
            dtype: Data type for the parameter tensor (default: float32)
        """
        self.rngs = rngs
        self.shape = shape
        kernel_key = self.rngs.params()
        
        # Default initializers
        weight_initializer = nnx.initializers.lecun_normal()
        
        # Use weight initializer for weights, zeros for biases
        if len(shape) == 1:  # Bias
            self.kernel_trainable = nnx.Param(
                jnp.zeros(shape, dtype=dtype)  # Direct zeros for bias
            )
        else:  # Weight matrix
            self.kernel_trainable = nnx.Param(
                weight_initializer(kernel_key, shape, dtype=dtype)
            )
        self.kernel_noise = Populated_Variable(
            jnp.zeros((self.popsize, *shape), dtype=dtype)
        )
        self.grad_variable = Grad_variable(
            jnp.zeros(shape)
        )
    
    def __call__(self):
        """Return the parameter tensor, with noise added during training."""
        if self.deterministic:
            return self.kernel_trainable.value
        else:
            return self.kernel_trainable.value + self.kernel_noise.value
    
    def sampling(self):
        """Generate symmetric noise for the parameter tensor."""
        # Apply sigma decay if enabled
        self.noise_sigma = jnp.where(
            self.enable_sigma_decay,
            jnp.maximum(jnp.multiply(self.noise_sigma, 0.9999), self.min_sigma),
            self.noise_sigma
        )
        # Generate symmetric noise (half positive, half negative)
        kernel_key = self.rngs.params()
        half = self.popsize // 2
        pos_noise = jax.random.normal(kernel_key, (half, *self.shape)) * self.noise_sigma
        # Concatenate positive and negative noise
        self.kernel_noise.value = jnp.concatenate([pos_noise, -pos_noise], 0)

    def estimate(self, fitness):
        """Estimate gradient from fitness values across the population.
        
        Args:
            fitness: Array of fitness values for each population member
        """
        # Convert fitness to centered ranks for fitness shaping
        w = fitness
        half = self.popsize // 2
        
        # Get the weights for positive and negative noise samples
        # In the reference implementation: w_half = w[:half] - w[half:]
        # We want to weight each noise vector by how much better/worse it performed
        w_diff = w[:half] - w[half:]  # Use symmetry to simplify
        
        # Split noise into positive and negative parts (we only need positive since negative is just -positive)
        pos_noise = self.kernel_noise.value[:half]
        
        # Following the NES gradient estimator formula: (1/sigma*n) * sum(w_i * epsilon_i)
        # where w_i is the fitness-shaped weight and epsilon_i is the noise vector
        # The axes=([0], [0]) ensures we're summing across the population dimension
        grad = jnp.tensordot(w_diff, pos_noise, axes=([0], [0])) / (self.noise_sigma * half)
        
        # Store the computed gradient
        self.grad_variable.value = grad
