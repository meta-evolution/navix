"""Utility functions for Evolution Strategies."""

import jax
import jax.numpy as jnp
from flax import nnx


def populated_noise_fwd(fitness_fn, model, popsize):
    """Evaluate fitness function across population with noise.
    
    Args:
        fitness_fn: Function that evaluates model fitness
        model: ES model with noise sampling capability
        popsize: Population size
    
    Returns:
        Array of fitness values for each population member
    """
    fitnesses = []
    
    for i in range(popsize):
        # Set the current individual index for noise selection
        for module in jax.tree.leaves(model):
            if hasattr(module, 'kernel_noise'):
                # Select the i-th noise vector for this individual
                original_noise = module.kernel_noise.value
                module.kernel_noise.value = original_noise[i:i+1]  # Keep batch dimension
        
        # Evaluate fitness for this individual
        fitness = fitness_fn()
        fitnesses.append(fitness)
        
        # Restore original noise
        for module in jax.tree.leaves(model):
            if hasattr(module, 'kernel_noise'):
                module.kernel_noise.value = original_noise
    
    return jnp.array(fitnesses)


def centered_rank(fitness_values):
    """Convert fitness values to centered ranks for fitness shaping.
    
    Args:
        fitness_values: Array of fitness values
    
    Returns:
        Array of centered rank values
    """
    # Get the ranking (argsort gives indices that would sort the array)
    ranks = jnp.argsort(jnp.argsort(fitness_values))
    
    # Convert to centered ranks: [-0.5, 0.5] range
    n = len(fitness_values)
    centered_ranks = (ranks / (n - 1)) - 0.5
    
    return centered_ranks


def fitness_shaping(fitness_values, method='centered_rank'):
    """Apply fitness shaping to raw fitness values.
    
    Args:
        fitness_values: Raw fitness values
        method: Shaping method ('centered_rank', 'rank', 'raw')
    
    Returns:
        Shaped fitness values
    """
    if method == 'centered_rank':
        return centered_rank(fitness_values)
    elif method == 'rank':
        ranks = jnp.argsort(jnp.argsort(fitness_values))
        return ranks.astype(jnp.float32)
    elif method == 'raw':
        return fitness_values
    else:
        raise ValueError(f"Unknown fitness shaping method: {method}")


def compute_es_gradient(noise, fitness_weights, sigma=1.0):
    """Compute ES gradient from noise and fitness weights.
    
    Args:
        noise: Noise vectors [popsize, ...]
        fitness_weights: Fitness-based weights [popsize]
        sigma: Noise standard deviation
    
    Returns:
        Gradient estimate
    """
    # Standard ES gradient estimator
    gradient = jnp.tensordot(fitness_weights, noise, axes=([0], [0]))
    return gradient / (sigma * len(fitness_weights))