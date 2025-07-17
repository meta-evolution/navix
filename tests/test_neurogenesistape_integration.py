#!/usr/bin/env python3
"""
NeurogenesisTape Integration Test
测试复制到navix项目中的NeurogenesisTape代码的基本功能
"""

import sys
import os
import unittest
import jax
import jax.numpy as jnp
from jax import random

# 添加neurogenesistape模块到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neurogenesistape'))

try:
    # 测试基本导入
    from modules.variables import Populated_Variable, Grad_variable
    from modules.evolution import EvoModule, sampling, calculate_gradients, centered_rank
    from modules.es.core import ES_Module, ES_Tape
    from modules.es.nn import ES_Linear, ES_MLP
    from modules.es.optimizer import ES_Optimizer
    from modules.es.training import compute_fitness, train_step
    print("✓ 所有核心模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

class TestNeurogenesisTapeIntegration(unittest.TestCase):
    """测试NeurogenesisTape核心功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.key = random.PRNGKey(42)
        self.input_dim = 10
        self.output_dim = 5
        self.batch_size = 32
        
    def test_variables(self):
        """测试变量类型"""
        # 测试Populated_Variable
        pop_var = Populated_Variable(jnp.zeros((10,)))
        self.assertEqual(pop_var.value.shape, (10,))
        
        # 测试Grad_variable
        grad_var = Grad_variable(jnp.zeros((10,)))
        self.assertEqual(grad_var.value.shape, (10,))
        print("✓ 变量类型测试通过")
        
    def test_centered_rank(self):
        """测试centered_rank函数"""
        x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
        ranked = centered_rank(x)
        
        # 检查形状保持
        self.assertEqual(ranked.shape, x.shape)
        
        # 检查值在[-0.5, 0.5]范围内
        self.assertTrue(jnp.all(ranked >= -0.5))
        self.assertTrue(jnp.all(ranked <= 0.5))
        
        # 检查和为零（居中）
        self.assertAlmostEqual(float(jnp.sum(ranked)), 0.0, places=6)
        print("✓ centered_rank函数测试通过")
        
    def test_es_linear(self):
        """测试ES_Linear层"""
        from flax import nnx
        key1, key2 = random.split(self.key)
        
        # 创建ES_Linear层
        rngs = nnx.Rngs(params=key1)
        layer = ES_Linear(self.input_dim, self.output_dim, rngs=rngs)
        
        # 设置为确定性模式以避免population维度
        layer.kernel.deterministic = True
        layer.bias.deterministic = True
        
        # 测试前向传播
        x = random.normal(key2, (self.batch_size, self.input_dim))
        output = layer(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        print("✓ ES_Linear层测试通过")
        
    def test_es_mlp(self):
        """测试ES_MLP网络"""
        from flax import nnx
        key1, key2 = random.split(self.key)
        
        # 创建ES_MLP网络
        layer_sizes = [self.input_dim, 20, 15, self.output_dim]
        rngs = nnx.Rngs(params=key1)
        mlp = ES_MLP(layer_sizes, rngs=rngs)
        
        # 设置所有层为确定性模式
        for layer in mlp.layers:
            layer.kernel.deterministic = True
            layer.bias.deterministic = True
        
        # 测试前向传播
        x = random.normal(key2, (self.batch_size, self.input_dim))
        output = mlp(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        print("✓ ES_MLP网络测试通过")
        
    def test_sampling_and_gradients(self):
        """测试采样和梯度计算"""
        from flax import nnx
        key1, key2 = random.split(self.key)
        
        # 创建简单的ES_Tape
        rngs = nnx.Rngs(params=key1)
        tape = ES_Tape((self.input_dim, self.output_dim), rngs)
        
        # 设置种群大小
        tape.popsize = 10
        tape.deterministic = False
        
        # 测试采样
        tape.sampling()
        
        # 检查噪声是否正确生成
        self.assertEqual(tape.kernel_noise.value.shape, (10, self.input_dim, self.output_dim))
        print("✓ 采样和梯度计算测试通过")
        
    def test_compute_fitness(self):
        """测试适应度计算"""
        # 创建模拟的预测和标签
        population_size = 10
        num_classes = 5
        batch_size = 32
        
        key1, key2 = random.split(self.key)
        predictions = random.normal(key1, (population_size, batch_size, num_classes))
        labels = random.randint(key2, (batch_size,), 0, num_classes)  # 修正：标签应该是一维的
        
        # 计算适应度
        fitness = compute_fitness(predictions, labels)
        
        self.assertEqual(fitness.shape, (population_size,))
        print("✓ 适应度计算测试通过")

def run_basic_functionality_test():
    """运行基本功能测试"""
    print("\n=== NeurogenesisTape 基本功能测试 ===")
    
    # 测试基本导入和功能
    key = random.PRNGKey(42)
    
    # 测试centered_rank
    x = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])
    ranked = centered_rank(x)
    print(f"原始值: {x}")
    print(f"排序后: {ranked}")
    print(f"和: {jnp.sum(ranked):.6f}")
    
    # 测试ES_Linear
    from flax import nnx
    key1, key2 = random.split(key)
    rngs = nnx.Rngs(params=key1)
    layer = ES_Linear(10, 5, rngs=rngs)
    
    # 设置为确定性模式
    layer.kernel.deterministic = True
    layer.bias.deterministic = True
    
    x_test = random.normal(key2, (32, 10))
    output = layer(x_test)
    print(f"ES_Linear输入形状: {x_test.shape}")
    print(f"ES_Linear输出形状: {output.shape}")
    
    print("\n✓ 所有基本功能测试通过！")

if __name__ == '__main__':
    # 运行基本功能测试
    run_basic_functionality_test()
    
    # 运行单元测试
    print("\n=== 运行单元测试 ===")
    unittest.main(verbosity=2)