#!/usr/bin/env python3

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
import jax.numpy as jnp
from neurogenesistape.modules.es.nn import ES_RNN
from neurogenesistape.modules.es.optimizer import ES_Optimizer
from neurogenesistape.modules.evolution import sampling, calculate_gradients, centered_rank
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
from flax import nnx
import optax

others = (nnx.RngCount, nnx.RngKey)
state_axes = nnx.StateAxes({nnx.Param: None, Populated_Variable: 0, Grad_variable: None, nnx.Variable: None, others: None})

@nnx.vmap(in_axes=(state_axes, None))
def populated_noise_fwd(model: nnx.Module, input):
    return model(input)

def generate_batch_data(key, batch_size=16, seq_len=5):
    """
    生成包含4个类别的简化序列数据集
    每个类别对应不同的数学序列模式
    """
    patterns = []
    
    # 类别0: 算术序列 (等差数列) [1, 2, 3, 4, 5]
    seq = [[1 + j] for j in range(seq_len)]
    patterns.append(seq)
    
    # 类别1: 几何序列 (等比数列) [1, 2, 4, 8, 16]
    seq = [[2 ** j] for j in range(seq_len)]
    patterns.append(seq)
    
    # 类别2: 平方序列 [1, 4, 9, 16, 25]
    seq = [[(j + 1) ** 2] for j in range(seq_len)]
    patterns.append(seq)
    
    # 类别3: 斐波那契序列 [1, 1, 2, 3, 5]
    seq = [[1], [1]]
    for j in range(seq_len - 2):
        next_val = seq[-1][0] + seq[-2][0]
        seq.append([next_val])
    patterns.append(seq)
    
    patterns = jnp.array(patterns, dtype=jnp.float32)
    
    # 生成批次数据
    num_classes = len(patterns)
    samples_per_class = batch_size // num_classes
    
    sequences = []
    targets = []
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            sequences.append(patterns[class_id])
            targets.append(class_id)
    
    # 填充剩余样本
    remaining = batch_size - len(sequences)
    for i in range(remaining):
        class_id = i % num_classes
        sequences.append(patterns[class_id])
        targets.append(class_id)
    
    sequences = jnp.array(sequences)
    targets = jnp.array(targets)
    
    return sequences, targets

def compute_fitness(outputs, targets):
    def _fitness(_outputs, _targets):
        logits = _outputs[:, -1, :]
        targets_onehot = jax.nn.one_hot(_targets, num_classes=logits.shape[-1])
        mse_loss = jnp.mean((logits - targets_onehot) ** 2)
        fitness = -mse_loss
        return fitness
    
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(outputs, targets)
    return fitness_values

def train_step(state, batch_x, batch_y, generation=0):
    state.model.reset_hidden(batch_x.shape[0])
    sampling(state.model)
    outputs = populated_noise_fwd(state.model, batch_x)
    
    fitness = compute_fitness(outputs, batch_y)
    avg_fitness = jnp.mean(fitness)
    fitness_ranked = centered_rank(fitness)
    grads = calculate_gradients(state.model, fitness_ranked)
    state.update(grads)
    
    return avg_fitness, grads, outputs, fitness

def main():
    config = {
        'input_size': 1, 'hidden_size': 128, 'output_size': 4,
        'popsize': 2000, 'generations': 1000, 'learning_rate': 0.05, 'sigma': 0.05
    }
    
    key = jax.random.PRNGKey(21)
    rngs = nnx.Rngs(key)
    model = ES_RNN(config['input_size'], config['hidden_size'], config['output_size'], rngs)
    
    model.set_attributes(popsize=config['popsize'], noise_sigma=config['sigma'])
    model.deterministic = False
    
    tx = optax.chain(optax.scale(-1), optax.sgd(config['learning_rate']))
    state = ES_Optimizer(model, tx, wrt=nnx.Param)
    
    key, data_key = jax.random.split(key)
    train_data, train_targets = generate_batch_data(data_key, batch_size=16)
    
    print("训练数据 (train_data):")
    print(train_data)
    print("\n目标数据 (train_targets):")
    print(train_targets)
    print("\n开始ES训练...\n")
    
    for gen in range(1, config['generations'] + 1):
        if gen == 1:
            old_params = state.model.i2h.kernel.grad_variable.value.copy()
        
        avg_fitness, grads, outputs, fitness = train_step(state, train_data, train_targets, gen)
        
        best_idx = jnp.argmax(fitness)
        best_output = outputs[best_idx]
        best_logits = best_output[:, -1, :]
        
        if gen <= 20 or gen % 10 == 0:
            new_params = state.model.i2h.kernel.grad_variable.value
            param_change = jnp.max(jnp.abs(new_params - old_params))
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            
            # 计算准确率
            best_predictions = jnp.argmax(best_logits, axis=-1)
            accuracy = jnp.mean(best_predictions == train_targets)
            
            # 调试信息：检查维度和数值
            if gen <= 3:
                print(f"调试信息 - Gen {gen}:")
                print(f"  best_logits.shape: {best_logits.shape}")
                print(f"  train_targets.shape: {train_targets.shape}")
                print(f"  best_predictions[:5]: {best_predictions[:5]}")
                print(f"  train_targets[:5]: {train_targets[:5]}")
                print(f"  best_logits[:3]: {best_logits[:3]}")
                
                # 重新计算MSE来验证
                targets_onehot = jax.nn.one_hot(train_targets, num_classes=4)
                mse_manual = jnp.mean((best_logits - targets_onehot) ** 2)
                print(f"  手动计算MSE: {mse_manual:.6f}")
                print(f"  最佳适应度: {fitness[best_idx]:.6f}")
                print()
            
            print(f"Gen {gen:3d}: avg={avg_fitness:8.6f}, accuracy={accuracy:.4f}, param_change={param_change:.8f}, grad_norm={grad_norm:.8f}")
            print(f"         最佳适应度: {fitness[best_idx]:.6f}")
            old_params = new_params.copy()
        else:
            best_predictions = jnp.argmax(best_logits, axis=-1)
            accuracy = jnp.mean(best_predictions == train_targets)
            print(f"Gen {gen:3d}: avg={avg_fitness:8.6f}, accuracy={accuracy:.4f}, 最佳适应度: {fitness[best_idx]:.6f}")

if __name__ == "__main__":
    main()