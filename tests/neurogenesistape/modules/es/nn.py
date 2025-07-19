"""Neural network components for Evolution Strategies (ES)."""

import jax
from jax import numpy as jnp
from flax import nnx
from typing import List, Tuple

from neurogenesistape.modules.es.core import ES_Tape


class ES_Linear(nnx.Module):
    """Linear layer using ES parameters."""
    
    def __init__(self, in_features: int, out_features: int, rngs: nnx.rnglib.Rngs):
        """Initialize an ES_Linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            rngs: Random number generator keys
        """
        self.kernel = ES_Tape((in_features, out_features), rngs)
        self.bias = ES_Tape((out_features,), rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the linear layer.
        
        Args:
            x: Input tensor of shape [..., in_features]
            
        Returns:
            Output tensor of shape [..., out_features]
        """
        y = x @ self.kernel()
        y = y + self.bias()
        return y


class ES_MLP(nnx.Module):
    """Multi-layer perceptron using ES linear layers."""
    
    def __init__(self, layer_sizes: List[int], rngs: nnx.rnglib.Rngs):
        """Initialize an ES_MLP network.
        
        Args:
            layer_sizes: List of layer sizes, including input and output dimensions
            rngs: Random number generator keys
        """
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(ES_Linear(layer_sizes[i], layer_sizes[i + 1], rngs))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape [batch_size, layer_sizes[0]]
            
        Returns:
            Output tensor of shape [batch_size, layer_sizes[-1]]
        """
        # Apply all layers with ReLU activation except the last one
        for i, layer in enumerate(self.layers[:-1]):
            x = jax.nn.relu(layer(x))
        
        # Last layer without activation (for logits)
        return self.layers[-1](x)


class ES_RNN(nnx.Module):
    """Vanilla RNN using ES linear layers with persistent hidden state."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, rngs: nnx.rnglib.Rngs):
        """Initialize an ES_RNN network.
        
        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            output_size: Output dimension
            rngs: Random number generator keys
        """
        self.hidden_size = hidden_size
        self.i2h = ES_Linear(input_size, hidden_size, rngs)
        self.h2h = ES_Linear(hidden_size, hidden_size, rngs)
        self.h2o = ES_Linear(hidden_size, output_size, rngs)
        
        # Add persistent hidden state as a non-trainable variable
        self.hidden_state = nnx.Variable(jnp.zeros((1, hidden_size)))
        
    def reset_hidden(self, batch_size: int = 1):
        """Reset the hidden state to zeros.
        
        Args:
            batch_size: Batch size for the hidden state
        """
        # Create new hidden state instead of mutating existing one
        new_hidden = jnp.zeros((batch_size, self.hidden_size))
        # Update the variable value
        self.hidden_state.value = new_hidden
        
    def get_hidden(self, batch_size: int = None) -> jnp.ndarray:
        """Get the current hidden state, expanding for batch size if needed.
        
        Args:
            batch_size: Target batch size
            
        Returns:
            Hidden state tensor
        """
        if batch_size is None or batch_size == self.hidden_state.value.shape[0]:
            return self.hidden_state.value
        else:
            # Expand hidden state to match batch size
            return jnp.broadcast_to(self.hidden_state.value[:1], (batch_size, self.hidden_size))
            
    def update_hidden(self, new_hidden: jnp.ndarray):
        """Update the persistent hidden state.
        
        Args:
            new_hidden: New hidden state of shape [batch_size, hidden_size]
        """
        # Average across batch dimension to maintain single persistent state
        averaged_hidden = jnp.mean(new_hidden, axis=0, keepdims=True)
        # Update the variable value
        self.hidden_state.value = averaged_hidden
    
    def __call__(self, x: jnp.ndarray, hidden: jnp.ndarray = None) -> jnp.ndarray:
        """Forward pass through the RNN.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size] or [batch_size, input_size]
            hidden: Initial hidden state (optional, uses persistent state if None)
        
        Returns:
            output: Output tensor
        """
        if x.ndim == 3:  # Sequence input [batch, seq, features]
            return self.forward_sequence(x)
        else:  # Single step input [batch, features]
            if hidden is None:
                hidden = self.get_hidden(x.shape[0])
            
            new_hidden = jax.nn.tanh(self.i2h(x) + self.h2h(hidden))
            output = self.h2o(new_hidden)
                
            return output
    
    def step(self, x: jnp.ndarray, hidden: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single RNN step.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            hidden: Hidden state
        
        Returns:
            output: Output tensor of shape [batch_size, output_size]
            new_hidden: New hidden state
        """
        hidden = jax.nn.tanh(self.i2h(x) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden
    
    def forward_sequence(self, sequence: jnp.ndarray) -> jnp.ndarray:
        """Process an entire sequence using persistent hidden state.
        
        Args:
            sequence: Input sequence of shape [batch_size, seq_len, input_size]
        
        Returns:
            outputs: Output sequence of shape [batch_size, seq_len, output_size]
        """
        def scan_fn(carry, x):
            # carry: [batch_size, hidden_size]
            # x: [batch_size, input_size]
            hidden = jax.nn.tanh(self.i2h(x) + self.h2h(carry))
            output = self.h2o(hidden)
            return hidden, output
        
        # Use persistent hidden state as initial state
        initial_hidden = self.get_hidden(sequence.shape[0])
        final_hidden, outputs = jax.lax.scan(scan_fn, initial_hidden, sequence.transpose(1, 0, 2))
            
        return outputs.transpose(1, 0, 2)
