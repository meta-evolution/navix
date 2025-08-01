"""Evolution Strategies (ES) implementation for NeurogenesisTape."""

# Import ES-specific core components
from neurogenesistape.modules.es.core import ES_Module, ES_Tape

# Import neural network components
from neurogenesistape.modules.es.nn import ES_Linear, ES_MLP, ES_Conv2d, ES_CNN

# Import optimizer
from neurogenesistape.modules.es.optimizer import ES_Optimizer

# Import training utilities
from neurogenesistape.modules.es.training import compute_fitness, train_step, train_step_with_fitness, ESConfig

__all__ = [
    # Core ES components
    'ES_Module', 'ES_Tape',
    
    # Neural network components
    'ES_Linear', 'ES_MLP', 'ES_Conv2d', 'ES_CNN',
    
    # Optimizer
    'ES_Optimizer',
    
    # Training utilities
    'compute_fitness', 'train_step', 'train_step_with_fitness', 'ESConfig'
]
