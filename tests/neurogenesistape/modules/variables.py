"""Core variable types for evolutionary algorithms."""

from flax import nnx


class Populated_Variable(nnx.Variable):
    """Variable type for storing populated noise across a population.
    
    This variable type is used in various evolutionary algorithms to store
    parameter perturbations across a population of candidate solutions.
    It's not specific to any single algorithm, but can be used in:
    - Evolution Strategies (ES)
    - Natural Evolution Strategies (NES)
    - Population-based training methods
    - And other population-based evolutionary algorithms
    """
    pass


class Grad_variable(nnx.Variable):
    """Variable type for storing gradient information.
    
    This variable type is used in various evolutionary algorithms to store
    estimated gradient information. While classical evolution doesn't
    explicitly compute gradients, many modern approaches like:
    - Evolution Strategies (ES)
    - Natural Evolution Strategies (NES)
    - Genetic Algorithms with gradient-based local search
    
    can benefit from this gradient representation.
    """
    pass
