"""Training utilities for Evolution Strategies (ES)."""

import jax
from jax import numpy as jnp
from flax import nnx
import optax
from typing import Any, Tuple
from dataclasses import dataclass

# Import shared evolutionary components
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
from neurogenesistape.modules.evolution import (
    EvoModule, state_axes, populated_noise_fwd, 
    sampling, calculate_gradients, centered_rank, evaluate
)


# ----- Configuration -----

@dataclass
class ESConfig:
    """Configuration for ES training."""
    generations: int = 2000     # Number of generations to train
    pop_size: int = 2048        # Population size (must be even)
    sigma: float = 0.05         # Initial noise standard deviation
    sigma_min: float = 0.01     # Minimum noise standard deviation
    sigma_decay: float = 0.999  # Decay rate for noise standard deviation
    lr: float = 0.05            # Learning rate
    batch_train: int = 2048     # Batch size for training


# ----- Fitness Functions -----


def compute_fitness(logits, labels):
    """Compute fitness values for classification tasks.
    
    Args:
        logits: Predicted logits from the model
        labels: Ground truth labels
        
    Returns:
        Fitness values for each population member
    """
    # Calculate negative cross-entropy for all population members
    def _fitness(_logits, _labels):
        return -jnp.mean(optax.softmax_cross_entropy_with_integer_labels(_logits, _labels))
    
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(logits, labels)
    return fitness_values


# ----- Training -----

def train_step(state, batch_x, batch_y) -> Tuple[nnx.Module, jax.Array]:
    """Training step function (not fully JIT-compiled due to model being an object).
    
    This function performs a complete ES training step:
    1. Forward pass with noise
    2. Fitness calculation
    3. Gradient computation
    4. Parameter updates
    
    Args:
        state: Optimizer state containing the model
        batch_x: Batch of input data
        batch_y: Batch of target labels
        
    Returns:
        Average fitness value for the batch
    """
    return _train_step(state, batch_x, batch_y)


@nnx.jit
def _train_step(state, batch_x, batch_y):
    """JIT-compiled implementation of the training step.
    
    Args:
        state: Optimizer state containing the model
        batch_x: Batch of input data
        batch_y: Batch of target labels
        
    Returns:
        Average fitness value for the batch
    """
    sampling(state.model)
    logits = populated_noise_fwd(state.model, batch_x)
    fitness = compute_fitness(logits, batch_y)
    avg_fitness = jnp.mean(fitness)
    fitness = centered_rank(fitness)
    grads = calculate_gradients(state.model, fitness)
    state.update(grads)
    
    return avg_fitness


# ----- The evaluate function has been moved to evolution.py -----
