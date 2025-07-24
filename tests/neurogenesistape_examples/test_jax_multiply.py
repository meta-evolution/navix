#!/usr/bin/env python3
"""Test script to verify JAX multiply implementation in ES_Tape."""

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
from jax import numpy as jnp
from flax import nnx

# Import the modified ES_Tape
from neurogenesistape.modules.es.core import ES_Tape

def test_jax_multiply():
    """Test that the JAX multiply implementation works correctly."""
    print("Testing JAX multiply implementation in ES_Tape...")
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    rngs = nnx.rnglib.Rngs(key)
    
    # Create an ES_Tape instance
    shape = (3, 3)  # Small matrix for testing
    es_tape = ES_Tape(shape, rngs)
    
    # Set initial values
    es_tape.popsize = 10
    es_tape.deterministic = False
    initial_sigma = 0.1
    es_tape.noise_sigma = initial_sigma
    
    print(f"Initial noise_sigma: {es_tape.noise_sigma}")
    
    # Test the sampling method (which contains our modified line)
    try:
        es_tape.sampling()
        print(f"After sampling, noise_sigma: {es_tape.noise_sigma}")
        
        # Verify the multiplication worked correctly
        expected_sigma = jnp.multiply(initial_sigma, 0.999)
        print(f"Expected sigma: {expected_sigma}")
        
        # Check if the values are close (accounting for floating point precision)
        if jnp.allclose(es_tape.noise_sigma, expected_sigma):
            print("✅ JAX multiply implementation works correctly!")
            return True
        else:
            print("❌ JAX multiply implementation failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during sampling: {e}")
        return False

def test_multiple_iterations():
    """Test multiple iterations to ensure consistent behavior."""
    print("\nTesting multiple iterations...")
    
    key = jax.random.PRNGKey(42)
    rngs = nnx.rnglib.Rngs(key)
    
    shape = (2, 2)
    es_tape = ES_Tape(shape, rngs)
    es_tape.popsize = 4
    es_tape.deterministic = False
    es_tape.noise_sigma = 1.0
    
    print(f"Initial sigma: {es_tape.noise_sigma}")
    
    # Run multiple iterations
    for i in range(5):
        es_tape.sampling()
        print(f"Iteration {i+1}: sigma = {es_tape.noise_sigma:.6f}")
    
    # Check if sigma decreased as expected
    expected_final = 1.0 * (0.999 ** 5)
    print(f"Expected final sigma: {expected_final:.6f}")
    
    if jnp.allclose(es_tape.noise_sigma, expected_final):
        print("✅ Multiple iterations work correctly!")
        return True
    else:
        print("❌ Multiple iterations failed!")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Testing JAX multiply implementation")
    print("=" * 50)
    
    success1 = test_jax_multiply()
    success2 = test_multiple_iterations()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! JAX multiply implementation is working correctly.")
    else:
        print("💥 Some tests failed. Please check the implementation.")
    print("=" * 50)