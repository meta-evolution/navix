"""
NeurogenesisTape: A library for evolutionary algorithms in JAX.

This package provides implementations of various evolutionary algorithms,
including Evolution Strategies (ES), Natural Evolution Strategies (NES),
and other Evolutionary Computation (EC) methods.
"""

import sys

__version__ = "0.1.0"

# Register ngt as an alias for neurogenesistape
if "ngt" not in sys.modules:
    sys.modules["ngt"] = sys.modules[__name__]

# Import and re-export core components

# Core variable types (moved to modules/variables.py)
from neurogenesistape.modules.variables import (
    Populated_Variable, Grad_variable
)

# Shared evolutionary components (moved to modules/evolution.py)
from neurogenesistape.modules.evolution import (
    EvoModule, sampling, calculate_gradients, centered_rank,
    evaluate, state_axes, populated_noise_fwd
)

# ES-specific components
from neurogenesistape.modules.es import (
    # Core ES modules
    ES_Module, ES_Tape, 
    # Neural network components
    ES_Linear, ES_MLP,
    # Optimizer
    ES_Optimizer,
    # Training utilities
    compute_fitness, train_step, train_step_with_fitness, ESConfig
)

# Import and re-export data utilities
from .data import load_cifar10, load_cifar100, load_mnist, load_tiny_imagenet

__all__ = [
    # Version
    '__version__',
    
    # Core variable types
    'Populated_Variable', 'Grad_variable',
    
    # Shared evolutionary components
    'EvoModule', 'sampling', 'calculate_gradients', 'centered_rank',
    'evaluate', 'state_axes', 'populated_noise_fwd',
    
    # ES-specific components
    'ES_Module', 'ES_Tape',
    
    # Neural network components
    'ES_Linear', 'ES_MLP',
    
    # Optimizer
    'ES_Optimizer',
    
    # Training utilities
    'compute_fitness', 'train_step', 'train_step_with_fitness', 'ESConfig',
    
    # Data utilities
    'load_cifar10', 'load_cifar100', 'load_mnist', 'load_tiny_imagenet',
]
