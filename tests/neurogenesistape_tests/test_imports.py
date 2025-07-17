"""Test imports and basic functionality of the NeurogenesisTape library."""

import unittest
import sys
import os

class TestImports(unittest.TestCase):
    """Test that all components can be properly imported."""
    
    def test_environment_setup(self):
        """Test environment setup and module registration."""
        # Check environment information
        print("\nEnvironment Diagnostics:")
        print("Current working directory:", os.getcwd())
        print("Python path:", sys.path)
        
        # Check module registration before import
        pre_import_ngt = "ngt" in sys.modules
        pre_import_neurogenesistape = "neurogenesistape" in sys.modules
        print("Before imports - neurogenesistape in sys.modules:", pre_import_neurogenesistape)
        print("Before imports - ngt in sys.modules:", pre_import_ngt)
        
        # Import the package
        import neurogenesistape
        
        # Check if ngt was registered as an alias
        post_import_ngt = "ngt" in sys.modules
        print("After importing neurogenesistape - ngt in sys.modules:", post_import_ngt)
        
        # Assert that the ngt alias is registered after importing neurogenesistape
        self.assertTrue(post_import_ngt, "ngt alias was not registered after importing neurogenesistape")
        
        # Check version consistency
        import ngt
        self.assertEqual(neurogenesistape.__version__, ngt.__version__, 
                        "Version mismatch between neurogenesistape and ngt alias")
    
    def test_core_imports(self):
        """Test imports of core components."""
        # Import core variable types
        from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
        self.assertTrue(Populated_Variable is not None)
        self.assertTrue(Grad_variable is not None)
        
        # Import core evolutionary components
        from neurogenesistape.modules.evolution import (
            EvoModule, sampling, calculate_gradients, centered_rank,
            evaluate, state_axes, populated_noise_fwd
        )
        self.assertTrue(EvoModule is not None)
        self.assertTrue(sampling is not None)
        self.assertTrue(calculate_gradients is not None)
        self.assertTrue(centered_rank is not None)
        self.assertTrue(evaluate is not None)
        self.assertTrue(state_axes is not None)
        self.assertTrue(populated_noise_fwd is not None)
    
    def test_es_imports(self):
        """Test imports of ES-specific components."""
        # Import ES components
        from neurogenesistape.modules.es import (
            ES_Module, ES_Tape, ES_Linear, ES_MLP, 
            ES_Optimizer, compute_fitness, train_step, ESConfig
        )
        self.assertTrue(ES_Module is not None)
        self.assertTrue(ES_Tape is not None)
        self.assertTrue(ES_Linear is not None)
        self.assertTrue(ES_MLP is not None)
        self.assertTrue(ES_Optimizer is not None)
        self.assertTrue(compute_fitness is not None)
        self.assertTrue(train_step is not None)
        self.assertTrue(ESConfig is not None)
    
    def test_package_imports(self):
        """Test top-level imports from the package."""
        # Import from top-level package
        import neurogenesistape
        self.assertEqual(neurogenesistape.__version__, "0.1.0")
        
        # Test ngt alias
        self.assertTrue("ngt" in sys.modules)
        import ngt
        self.assertEqual(ngt.__version__, "0.1.0")
    
    def test_ngt_imports(self):
        """Test imports using the ngt alias."""
        import ngt
        # Test that core components are available through ngt
        self.assertTrue(hasattr(ngt, "EvoModule"))
        self.assertTrue(hasattr(ngt, "Populated_Variable"))
        self.assertTrue(hasattr(ngt, "Grad_variable"))
        
        # Test that ES components are available through ngt
        self.assertTrue(hasattr(ngt, "ES_Module"))
        self.assertTrue(hasattr(ngt, "ES_Tape"))
        self.assertTrue(hasattr(ngt, "ES_Linear"))
        self.assertTrue(hasattr(ngt, "ES_MLP"))
        self.assertTrue(hasattr(ngt, "ES_Optimizer"))
        
        # Test that utility functions are available through ngt
        self.assertTrue(hasattr(ngt, "sampling"))
        self.assertTrue(hasattr(ngt, "calculate_gradients"))
        self.assertTrue(hasattr(ngt, "centered_rank"))
        self.assertTrue(hasattr(ngt, "evaluate"))


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core components."""
    
    def test_centered_rank(self):
        """Test the centered_rank function."""
        import jax.numpy as jnp
        from neurogenesistape import centered_rank
        
        # Test with an array of values
        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        ranked = centered_rank(x)
        
        # Check that the shape is preserved
        self.assertEqual(ranked.shape, x.shape)
        
        # Check that the values are between -0.5 and 0.5
        self.assertTrue(jnp.all(ranked >= -0.5))
        self.assertTrue(jnp.all(ranked <= 0.5))
        
        # Check that the sum is zero (centered)
        self.assertAlmostEqual(float(jnp.sum(ranked)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
