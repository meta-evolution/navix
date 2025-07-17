"""Modules for NeurogenesisTape's evolutionary algorithms."""

# Import shared evolutionary components
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
from neurogenesistape.modules.evolution import (
    EvoModule, sampling, calculate_gradients, centered_rank,
    evaluate, state_axes, populated_noise_fwd
)

# Import ES-specific components
from neurogenesistape.modules.es import (
    # Core ES modules
    ES_Module, ES_Tape,
    # Neural network components
    ES_Linear, ES_MLP,
    # Optimizer
    ES_Optimizer,
    # Training utilities
    compute_fitness, train_step, ESConfig
)

__all__ = [
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
    'compute_fitness', 'train_step', 'ESConfig',
]
