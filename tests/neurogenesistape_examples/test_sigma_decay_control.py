#!/usr/bin/env python3
"""Test script to verify sigma decay control functionality in ES_Tape."""

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
from jax import numpy as jnp
from flax import nnx

# Import the modified ES_Tape
from neurogenesistape.modules.es.core import ES_Tape

def test_sigma_decay_enabled():
    """Test that sigma decay works when enabled (default)."""
    print("Testing sigma decay enabled (default)...")
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    rngs = nnx.rnglib.Rngs(key)
    
    # Create an ES_Tape instance
    shape = (3, 3)
    es_tape = ES_Tape(shape, rngs)
    
    # Set initial values
    es_tape.popsize = 10
    es_tape.deterministic = False
    initial_sigma = 0.1
    es_tape.noise_sigma = initial_sigma
    
    print(f"Initial noise_sigma: {es_tape.noise_sigma}")
    print(f"enable_sigma_decay: {es_tape.enable_sigma_decay}")
    
    # Test the sampling method
    es_tape.sampling()
    print(f"After sampling, noise_sigma: {es_tape.noise_sigma}")
    
    # Verify the decay worked
    expected_sigma = jnp.maximum(jnp.multiply(initial_sigma, 0.999), 0.01)
    if jnp.allclose(es_tape.noise_sigma, expected_sigma):
        print("✅ Sigma decay enabled works correctly!")
        return True
    else:
        print("❌ Sigma decay enabled failed!")
        return False

def test_sigma_decay_disabled():
    """Test that sigma decay is disabled when enable_sigma_decay is False."""
    print("\nTesting sigma decay disabled...")
    
    # Initialize random key
    key = jax.random.PRNGKey(42)
    rngs = nnx.rnglib.Rngs(key)
    
    # Create an ES_Tape instance
    shape = (3, 3)
    es_tape = ES_Tape(shape, rngs)
    
    # Set initial values and disable sigma decay
    es_tape.popsize = 10
    es_tape.deterministic = False
    initial_sigma = 0.1
    es_tape.noise_sigma = initial_sigma
    es_tape.enable_sigma_decay = False  # Disable sigma decay
    
    print(f"Initial noise_sigma: {es_tape.noise_sigma}")
    print(f"enable_sigma_decay: {es_tape.enable_sigma_decay}")
    
    # Test the sampling method
    es_tape.sampling()
    print(f"After sampling, noise_sigma: {es_tape.noise_sigma}")
    
    # Verify the sigma remained unchanged
    if jnp.allclose(es_tape.noise_sigma, initial_sigma):
        print("✅ Sigma decay disabled works correctly!")
        return True
    else:
        print("❌ Sigma decay disabled failed!")
        return False

def test_multiple_iterations_with_control():
    """Test multiple iterations with decay control."""
    print("\nTesting multiple iterations with decay control...")
    
    # Test with decay enabled
    key = jax.random.PRNGKey(42)
    rngs = nnx.rnglib.Rngs(key)
    shape = (2, 2)
    es_tape_enabled = ES_Tape(shape, rngs)
    es_tape_enabled.popsize = 4
    es_tape_enabled.deterministic = False
    es_tape_enabled.noise_sigma = 1.0
    es_tape_enabled.enable_sigma_decay = True
    
    # Test with decay disabled
    key2 = jax.random.PRNGKey(42)
    rngs2 = nnx.rnglib.Rngs(key2)
    es_tape_disabled = ES_Tape(shape, rngs2)
    es_tape_disabled.popsize = 4
    es_tape_disabled.deterministic = False
    es_tape_disabled.noise_sigma = 1.0
    es_tape_disabled.enable_sigma_decay = False
    
    print("Enabled decay - Initial sigma:", es_tape_enabled.noise_sigma)
    print("Disabled decay - Initial sigma:", es_tape_disabled.noise_sigma)
    
    # Run 5 iterations
    for i in range(5):
        es_tape_enabled.sampling()
        es_tape_disabled.sampling()
        print(f"Iteration {i+1}: enabled={es_tape_enabled.noise_sigma:.6f}, disabled={es_tape_disabled.noise_sigma:.6f}")
    
    # Check results
    if es_tape_enabled.noise_sigma < 1.0 and es_tape_disabled.noise_sigma == 1.0:
        print("✅ Multiple iterations with control work correctly!")
        return True
    else:
        print("❌ Multiple iterations with control failed!")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sigma Decay Control Functionality")
    print("=" * 60)
    
    test1 = test_sigma_decay_enabled()
    test2 = test_sigma_decay_disabled()
    test3 = test_multiple_iterations_with_control()
    
    print("\n" + "=" * 60)
    if test1 and test2 and test3:
        print("🎉 All tests passed! Sigma decay control is working correctly.")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)