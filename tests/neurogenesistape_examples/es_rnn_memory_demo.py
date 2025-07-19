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

def generate_batch_data(key, batch_size=2, seq_len=3):
    """生成有规律的序列分类数据"""
    # 使用固定的有规律的序列模式
    patterns = jnp.array([
        [[1], [2], [3]],  # 模式1: 1-2-3 -> 类别0
        [[3], [2], [1]]   # 模式2: 3-2-1 -> 类别1
    ])
    
    # 重复模式以填充batch
    sequences = jnp.tile(patterns, (batch_size // 2 + 1, 1, 1))[:batch_size]
    
    # 分类标签：根据序列模式确定类别
    # 模式 [1,2,3] -> 类别0, 模式 [3,2,1] -> 类别1
    targets = jnp.array([0, 1] * (batch_size // 2 + 1))[:batch_size]
    
    return sequences.astype(jnp.float32), targets

def compute_fitness(outputs, targets):
    """计算分类任务适应度 - 使用MSE损失的负值"""
    def _fitness(_outputs, _targets):
        # 取最后一个时间步的输出
        logits = _outputs[:, -1, :]  # [batch_size, num_classes]
        
        # 将目标标签转换为one-hot编码
        targets_onehot = jax.nn.one_hot(_targets, num_classes=logits.shape[-1])
        
        # 计算MSE损失
        mse_loss = jnp.mean((logits - targets_onehot) ** 2)
        
        # 返回负MSE作为适应度（损失越小，适应度越高）
        fitness = -mse_loss
        return fitness
    
    # 对每个种群成员计算适应度
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(outputs, targets)
    return fitness_values

def train_step(state, batch_x, batch_y, generation=0):
    """标准ES训练步骤"""
    # 重置隐藏状态
    state.model.reset_hidden(batch_x.shape[0])
    
    # 1. 采样噪声
    sampling(state.model)
    
    # 2. 前向传播（种群）
    outputs = populated_noise_fwd(state.model, batch_x)
    
    # 打印网络输出详情（前几代）
    if generation <= 3:  # 只在前3代打印详细信息
        print(f"\n--- 第{generation}代种群输出详情 ---")
        print(f"输入数据: {batch_x}")
        print(f"目标标签: {batch_y}")
        print(f"种群大小: {outputs.shape[0]}")
        print(f"\n所有种群成员的输出:")
        
        # 显示所有种群成员的输出
        for i in range(outputs.shape[0]):
            logits = outputs[i, :, -1, :]  # [batch_size, output_size]
            predictions = jnp.argmax(logits, axis=-1)
            # 计算该成员的适应度（MSE损失的负值）
            targets_onehot = jax.nn.one_hot(batch_y, num_classes=logits.shape[-1])
            mse_loss = jnp.mean((logits - targets_onehot) ** 2)
            member_fitness = -mse_loss
            
            # 简化输出格式，每行显示一个成员
            print(f"  成员{i:2d}: logits={logits.flatten()}, 预测={predictions}, 适应度={member_fitness:.4f}")
            
            # 每10个成员换行，便于阅读
            if (i + 1) % 10 == 0 and i < outputs.shape[0] - 1:
                print("")
    
    # 3. 计算适应度
    fitness = compute_fitness(outputs, batch_y)
    avg_fitness = jnp.mean(fitness)
    
    # 4. 适应度排序
    fitness_ranked = centered_rank(fitness)
    
    # 5. 计算梯度
    grads = calculate_gradients(state.model, fitness_ranked)
    
    # 6. 更新参数
    state.update(grads)
    
    return avg_fitness, grads, outputs, fitness

def main():
    """ES-RNN训练"""
    # 配置 - 多分类任务（进一步优化ES参数）
    config = {
        'input_size': 1, 'hidden_size': 20, 'output_size': 2,  # 2分类任务
        'popsize': 200, 'generations': 50, 'learning_rate': 0.05, 'sigma': 0.05  # 增大sigma以提高种群多样性
    }
    
    # 初始化模型
    key = jax.random.PRNGKey(21)
    rngs = nnx.Rngs(key)
    model = ES_RNN(config['input_size'], config['hidden_size'], config['output_size'], rngs)
    
    # 设置ES参数
    model.set_attributes(popsize=config['popsize'], noise_sigma=config['sigma'])
    model.deterministic = False
    
    # 初始化优化器 - 使用nnx.Param来优化所有参数（hidden_state现在是Variable，不会被优化）
    tx = optax.chain(optax.scale(-1), optax.sgd(config['learning_rate']))
    state = ES_Optimizer(model, tx, wrt=nnx.Param)
    
    # 生成固定的训练数据（只生成一次）- 缩小数据规模
    key, data_key = jax.random.split(key)
    train_data, train_targets = generate_batch_data(data_key, batch_size=2)
    
    print("训练数据 (train_data):")
    print(train_data)
    print("\n目标数据 (train_targets):")
    print(train_targets)
    print("\n开始ES训练...\n")
    
    # ES训练
    for gen in range(1, config['generations'] + 1):
        # 记录训练前的参数
        if gen == 1:
            old_params = state.model.i2h.kernel.grad_variable.value.copy()
        
        avg_fitness, grads, outputs, fitness = train_step(state, train_data, train_targets, gen)
        
        # # 打印所有模型参数
        # print(f"\n=== Gen {gen} 参数详情 ===")
        # print(f"i2h.kernel: {state.model.i2h.kernel.grad_variable.value}")
        # print(f"i2h.bias: {state.model.i2h.bias.grad_variable.value}")
        # print(f"h2h.kernel: {state.model.h2h.kernel.grad_variable.value}")
        # print(f"h2h.bias: {state.model.h2h.bias.grad_variable.value}")
        # print(f"h2o.kernel: {state.model.h2o.kernel.grad_variable.value}")
        # print(f"h2o.bias: {state.model.h2o.bias.grad_variable.value}")
        
        # 检查参数是否变化
        if gen <= 10000:
            new_params = state.model.i2h.kernel.grad_variable.value
            param_change = jnp.max(jnp.abs(new_params - old_params))
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            
            # 调试信息：显示种群中每个成员的预测结果
            if gen <= 500:
                # print(f"\n种群适应度: {fitness}")
                # 显示第一个种群成员的输出
                first_member_output = outputs[0]  # [batch_size, seq_len, output_size]
                last_step_logits = first_member_output[:, -1, :]  # [batch_size, output_size]
                predictions = jnp.argmax(last_step_logits, axis=-1)
                # print(f"第一个成员最后时间步输出: {last_step_logits}")
                # print(f"预测类别: {predictions}, 真实类别: {train_targets}")
                # print(f"预测正确性: {predictions == train_targets}")
            
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}, param_change={param_change:.8f}, grad_norm={grad_norm:.8f}")
            old_params = new_params.copy()
        else:
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}")

if __name__ == "__main__":
    main()