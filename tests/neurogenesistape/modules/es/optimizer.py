"""Optimizer components for Evolution Strategies (ES)."""

import jax
from jax import numpy as jnp
from flax import nnx
import optax
from typing import Dict

from neurogenesistape.modules.variables import Grad_variable


def _opt_state_variables_to_state(opt_state):
    """Convert optimizer variables to state representation.
    
    Args:
        opt_state: Optimizer state with variables
        
    Returns:
        Converted optimizer state
    """
    def optimizer_variable_to_state_fn(x):
        if isinstance(x, nnx.optimizer.OptVariable):
            state = x.to_state()
            state.type = x.source_type
            del state.source_type
            return state
        elif isinstance(x, nnx.optimizer.OptArray):
            return x.value
        else:
            raise TypeError(
                f'Unexpected type when converting optimizer state: {type(x)}'
            )

    return jax.tree.map(
        optimizer_variable_to_state_fn,
        opt_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )


def _update_opt_state(opt_state, updates):
    """Update optimizer state with new values.
    
    Args:
        opt_state: Current optimizer state
        updates: Updates to apply
        
    Returns:
        Updated optimizer state
    """
    def optimizer_update_variables(x, update):
        if isinstance(x, nnx.optimizer.OptVariable):
            if not isinstance(update, nnx.VariableState):
                raise TypeError(
                    f'Expected update to be VariableState, got {type(update)}'
                )
            x.value = update.value
        elif isinstance(x, nnx.optimizer.OptArray):
            if isinstance(update, nnx.VariableState):
                raise TypeError(
                    f'Expected update to not to be a VariableState, got {update}'
                )
            x.value = update
        else:
            raise TypeError(
                f'Unexpected type when updating optimizer state: {type(x)}'
            )

    return jax.tree.map(
        optimizer_update_variables,
        opt_state,
        updates,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )


def _grad_to_param(x: nnx.State) -> nnx.State:
    """Convert gradient variables to parameter state.
    
    Args:
        x: State containing gradient variables
        
    Returns:
        State with converted parameters
    """
    def _fn(x):
        param_state = nnx.VariableState(nnx.Param, x['grad_variable'].value)
        return {'kernel_trainable': param_state}
        
    def _is_leaf(x) -> bool:
        if isinstance(x, Dict):
            if 'grad_variable' in x.keys():
                return True
        return False

    res = jax.tree.map(_fn, x, is_leaf=_is_leaf)
    return res
    

class ES_Optimizer(nnx.Optimizer):
    """Optimizer for Evolution Strategies.
    
    This optimizer converts gradients estimated by ES to parameter updates
    using any compatible optax optimizer.
    """

    def update(self, grads, **kwargs):
        """Update model parameters using estimated gradients.
        
        Args:
            grads: Gradients estimated by ES
            **kwargs: Additional keyword arguments for the optimizer
        """
        params = nnx.state(self.model, self.wrt)
        # grads = _grad_to_param(grads)
        opt_state = _opt_state_variables_to_state(self.opt_state)

        # Params : [..tree..]{'kernel_trainable': {type = Param}}
        # Updates: [..tree..]{'grad_variable': {type = Grad_Variable}}
        updates, new_opt_state = self.tx.update(grads, opt_state, params, **kwargs) 
        updates = _grad_to_param(updates)
        new_params = optax.apply_updates(params, updates)
        assert isinstance(new_params, nnx.State)

        self.step.value += 1
        nnx.update(self.model, new_params)
        _update_opt_state(self.opt_state, new_opt_state)
