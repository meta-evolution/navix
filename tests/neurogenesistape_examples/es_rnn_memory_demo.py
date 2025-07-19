#!/usr/bin/env python3
"""标准ES-RNN训练程序"""

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
import jax.numpy as jnp
from neurogenesistape.modules.es.nn import ES_RNN
from neurogenesistape.modules.es.optimizer import ES_Optimizer
from neurogenesistape.modules.evolution import sampling, populated_noise_fwd, calculate_gradients, centered_rank
from flax import nnx
import optax

def generate_batch_data(key, batch_size=8, seq_len=5):
    """生成批量序列数据"""
    sequences = jax.random.uniform(key, (batch_size, seq_len, 1), minval=-0.5, maxval=0.5)
    targets = jnp.zeros_like(sequences)
    targets = targets.at[:, 0, :].set(sequences[:, 0, :])  # 第一个输出 = 第一个输入
    targets = targets.at[:, 1:, :].set(sequences[:, 1:, :] + sequences[:, :-1, :])  # 其余 = 当前 + 前一个
    return sequences, targets

def compute_fitness(outputs, targets):
    """计算适应度 - 负MSE"""
    def _fitness(_outputs, _targets):
        return -jnp.mean((_outputs - _targets) ** 2)
    
    # 对每个种群成员计算适应度
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(outputs, targets)
    return fitness_values

@nnx.jit
def train_step(state, batch_x, batch_y):
    """标准ES训练步骤"""
    # 重置隐藏状态
    state.model.reset_hidden(batch_x.shape[0])
    
    # 1. 采样噪声
    sampling(state.model)
    
    # 2. 前向传播（种群）
    outputs = populated_noise_fwd(state.model, batch_x)
    
    # 3. 计算适应度
    fitness = compute_fitness(outputs, batch_y)
    avg_fitness = jnp.mean(fitness)
    
    # 4. 适应度排序
    fitness_ranked = centered_rank(fitness)
    
    # 5. 计算梯度
    grads = calculate_gradients(state.model, fitness_ranked)
    
    # 6. 更新参数
    state.update(grads)
    
    return avg_fitness, grads

def main():
    """ES-RNN训练"""
    # 配置
    config = {
        'input_size': 1, 'hidden_size': 4, 'output_size': 1,
        'popsize': 1000, 'generations': 1000, 'learning_rate': 0.1, 'sigma': 0.1
    }
    
    # 初始化模型
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    model = ES_RNN(config['input_size'], config['hidden_size'], config['output_size'], rngs)
    
    # 设置ES参数
    model.set_attributes(popsize=config['popsize'], noise_sigma=config['sigma'])
    model.deterministic = False
    
    # 初始化优化器 - 使用nnx.Param来优化所有参数（hidden_state现在是Variable，不会被优化）
    tx = optax.chain(optax.scale(-1), optax.sgd(config['learning_rate']))
    state = ES_Optimizer(model, tx, wrt=nnx.Param)
    
    # 生成固定的训练数据（只生成一次）
    key, data_key = jax.random.split(key)
    train_data, train_targets = generate_batch_data(data_key, batch_size=8)
    
    # ES训练
    for gen in range(1, config['generations'] + 1):
        # 记录训练前的参数
        if gen == 1:
            old_params = state.model.i2h.kernel.grad_variable.value.copy()
        
        avg_fitness, grads = train_step(state, train_data, train_targets)
        
        # 检查参数是否变化
        if gen <= 5:
            new_params = state.model.i2h.kernel.grad_variable.value
            param_change = jnp.max(jnp.abs(new_params - old_params))
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}, param_change={param_change:.8f}, grad_norm={grad_norm:.8f}")
            old_params = new_params.copy()
        else:
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}")

if __name__ == "__main__":
    main()