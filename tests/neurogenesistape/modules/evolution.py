"""Core evolutionary module classes for NeurogenesisTape."""

import jax
from jax import numpy as jnp
from flax import nnx
from typing import Any, Optional, Tuple

from neurogenesistape.modules.variables import Populated_Variable, Grad_variable


class EvoModule(nnx.Module):
    """Base class for all evolutionary algorithm modules.
    
    This class defines the interface for all evolutionary algorithm modules.
    It provides methods for sampling perturbations and estimating gradients,
    which are common operations across many evolutionary algorithms.
    
    Attributes:
        popsize: Population size for the evolutionary algorithm
        deterministic: Whether to use deterministic evaluation (no noise)
    """

    popsize: int = 2
    deterministic = False

    def sampling(self, *args):
        """Sample perturbations for the module parameters.
        
        This method should be implemented by subclasses to define
        how parameter perturbations are generated for the specific
        evolutionary algorithm.
        
        Args:
            *args: Additional arguments specific to the algorithm
        """
        raise NotImplementedError
    
    def estimate(self, fitness, *args):
        """Estimate gradients from fitness values.
        
        This method should be implemented by subclasses to define
        how gradients are estimated from fitness values for the specific
        evolutionary algorithm.
        
        Args:
            fitness: Fitness values for each population member
            *args: Additional arguments specific to the algorithm
        """
        raise NotImplementedError


# ----- Vectorization Utilities -----

others = (nnx.RngCount, nnx.RngKey)
state_axes = nnx.StateAxes({nnx.Param: None, Populated_Variable: 0, Grad_variable: None, nnx.Variable: None, others: None})


@nnx.vmap(in_axes=(state_axes, None))
def populated_noise_fwd(model: nnx.Module, input: Any):
    """Vectorized forward pass with noise population.
    
    This utility function performs a vectorized forward pass across
    the entire population, applying the appropriate noise perturbations
    to each population member.
    
    Args:
        model: Model to run forward pass on
        input: Input data
        
    Returns:
        Outputs for each member of the population
    """
    return model(input)


# ----- Common Utility Functions -----

def sampling(model: nnx.Module):
    """Sample noise for all evolutionary modules in the model.
    
    This function iterates through all evolutionary modules in the model
    and calls their sampling method to generate perturbations.
    
    Args:
        model: Model containing evolutionary modules
    """
    for path, i in model.iter_modules():
        if isinstance(i, EvoModule):
            i.sampling()


def calculate_gradients(model: nnx.Module, fitness):
    """Calculate gradients for all evolutionary modules in the model.
    
    This function iterates through all evolutionary modules in the model
    and calls their estimate method to calculate gradients from fitness values.
    
    Args:
        model: Model containing evolutionary modules
        fitness: Fitness values for each population member
        
    Returns:
        Gradients for the model parameters
    """
    for path, i in model.iter_modules():
        if isinstance(i, EvoModule):
            i.estimate(fitness)

    return nnx.state(model, nnx.Param)


def centered_rank(x):
    """Convert fitness values to centered ranks for fitness shaping.
    
    This transformation makes evolutionary algorithms more robust to outliers
    and ensures a consistent gradient scale. It can be used with various
    evolutionary algorithms that benefit from fitness shaping.
    
    Reference: https://arxiv.org/pdf/1703.03864.pdf
    
    Args:
        x: Array of fitness values
        
    Returns:
        Array of centered ranks
    """
    x = jnp.ravel(x)
    ranks = jnp.argsort(jnp.argsort(x))
    return ranks / (x.size - 1) - 0.5


@nnx.jit
def evaluate(model, x, y):
    """Evaluate model accuracy on dataset.
    
    This function evaluates the model on a dataset and returns the accuracy.
    It temporarily sets the model to deterministic mode for evaluation.
    
    Args:
        model: Model to evaluate
        x: Input data
        y: Target labels
        
    Returns:
        Accuracy (fraction of correct predictions)
    """
    model.set_attributes(deterministic=True)
    logits = model(x)
    model.set_attributes(deterministic=False)
    return jnp.mean((jnp.argmax(logits, -1) == y))
