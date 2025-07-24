"""Neural network components for Evolution Strategies (ES)."""

import jax
from jax import numpy as jnp
from flax import nnx
from typing import List, Tuple

from neurogenesistape.modules.es.core import ES_Tape


# JIT-optimized RNN sequence processing function
@jax.jit
def _rnn_sequence_forward(sequence: jnp.ndarray, initial_hidden: jnp.ndarray,
                         i2h_kernel: jnp.ndarray, i2h_bias: jnp.ndarray,
                         h2h_kernel: jnp.ndarray, h2h_bias: jnp.ndarray,
                         h2o_kernel: jnp.ndarray, h2o_bias: jnp.ndarray) -> jnp.ndarray:
    """JIT-optimized RNN sequence processing with inlined operations."""
    def scan_fn(carry, x):
        # Inline RNN cell computation
        i2h_out = x @ i2h_kernel + i2h_bias
        h2h_out = carry @ h2h_kernel + h2h_bias
        hidden = jax.nn.tanh(i2h_out + h2h_out)
        
        # Inline linear output computation
        output = hidden @ h2o_kernel + h2o_bias
        return hidden, output
    
    final_hidden, outputs = jax.lax.scan(scan_fn, initial_hidden, sequence.transpose(1, 0, 2))
    return outputs.transpose(1, 0, 2)


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
        return x @ self.kernel() + self.bias()


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
    
    def enable_sigma_decay(self, enabled: bool = True):
        """Enable or disable sigma decay for all ES modules in the network.
        
        Args:
            enabled: Whether to enable sigma decay (default: True)
        """
        for layer in self.layers:
            if hasattr(layer, 'kernel'):
                layer.kernel.set_sigma_decay(enabled)
            if hasattr(layer, 'bias'):
                layer.bias.set_sigma_decay(enabled)
    
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
    """Vanilla RNN using ES linear layers."""
    
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
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the RNN.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
        
        Returns:
            output: Output tensor of shape [batch_size, seq_len, output_size]
        """
        return self.forward_sequence(x)
    
    def forward_step(self, x: jnp.ndarray, hidden: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process a single time step for RL environments.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            hidden: Hidden state tensor of shape [batch_size, hidden_size]
        
        Returns:
            output: Output tensor of shape [batch_size, output_size]
            new_hidden: New hidden state tensor of shape [batch_size, hidden_size]
        """
        # RNN cell computation
        i2h_out = self.i2h(x)
        h2h_out = self.h2h(hidden)
        new_hidden = jax.nn.tanh(i2h_out + h2h_out)
        
        # Output computation
        output = self.h2o(new_hidden)
        
        return output, new_hidden
    
    def init_hidden(self, batch_size: int) -> jnp.ndarray:
        """Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Initial hidden state tensor of shape [batch_size, hidden_size]
        """
        return jnp.zeros((batch_size, self.hidden_size))
    
    def forward_sequence(self, sequence: jnp.ndarray) -> jnp.ndarray:
        """Process an entire sequence (JIT-optimized).
        
        Args:
            sequence: Input sequence of shape [batch_size, seq_len, input_size]
        
        Returns:
            outputs: Output sequence of shape [batch_size, seq_len, output_size]
        """
        # Initialize hidden state with zeros
        initial_hidden = jnp.zeros((sequence.shape[0], self.hidden_size))
        
        return _rnn_sequence_forward(sequence, initial_hidden,
                                   self.i2h.kernel(), self.i2h.bias(),
                                   self.h2h.kernel(), self.h2h.bias(),
                                   self.h2o.kernel(), self.h2o.bias())