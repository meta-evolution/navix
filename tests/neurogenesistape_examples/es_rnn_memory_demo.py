#!/usr/bin/env python3

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt
import numpy as np
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

@jax.jit
def compute_fitness(outputs, targets):
    def _fitness(_outputs, _targets):
        logits = _outputs[:, -1, :]
        targets_onehot = jax.nn.one_hot(_targets, num_classes=logits.shape[-1])
        mse_loss = jnp.mean((logits - targets_onehot) ** 2)
        fitness = -mse_loss
        return fitness
    
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(outputs, targets)
    return fitness_values

@jax.jit
def compute_accuracy(logits, targets):
    """JIT-compiled accuracy computation."""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)

@jax.jit
def compute_metrics(outputs, targets, fitness):
    """JIT-compiled metrics computation."""
    best_idx = jnp.argmax(fitness)
    best_output = outputs[best_idx]
    best_logits = best_output[:, -1, :]
    accuracy = compute_accuracy(best_logits, targets)
    return best_idx, best_logits, accuracy

@jax.jit
def compute_fitness_and_grads(outputs, targets, fitness_ranked):
    """JIT-compiled fitness and gradient computation."""
    fitness = compute_fitness(outputs, targets)
    avg_fitness = jnp.mean(fitness)
    return avg_fitness, fitness

def create_training_visualization(generations, best_fitness_history, accuracy_history):
    """
    创建训练过程的可视化图表，显示best fitness和accuracy moving average的变化趋势
    
    Args:
        generations: 代数列表
        best_fitness_history: 最佳适应度历史记录
        accuracy_history: 准确率历史记录
    """
    # 计算准确率的移动平均值
    def moving_average(data, window_size=10):
        """计算移动平均值"""
        if len(data) < window_size:
            window_size = len(data)
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            avg = np.mean(data[start_idx:end_idx])
            moving_avg.append(avg)
        return moving_avg
 
    # 计算准确率的移动平均值（窗口大小为20）
    accuracy_ma = moving_average(accuracy_history, window_size=20)
    
    # 设置图表样式
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制error曲线（左轴）- 将fitness乘以-1转换为error
    error_history = [-fitness for fitness in best_fitness_history]
    color1 = '#E74C3C'  # 现代红色
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Error', color=color1, fontsize=12)
    line1 = ax1.plot(generations, error_history, color=color1, linewidth=2.5, label='Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # 创建右轴绘制accuracy moving average曲线
    ax2 = ax1.twinx()
    color2 = '#3498DB'  # 现代蓝色
    ax2.set_ylabel('Accuracy (Moving Average)', color=color2, fontsize=12)
    line2 = ax2.plot(generations, accuracy_ma, color=color2, linewidth=2.5, label='Accuracy (MA-20)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 设置标题和图例
    plt.title('ES-RNN Training Progress: Error and Accuracy Moving Average', fontsize=14, fontweight='bold')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_path = 'es_rnn_training_progress.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练过程可视化图表已保存至: {output_path}")
    
    # 显示最终统计信息
    final_fitness = best_fitness_history[-1]
    final_accuracy = accuracy_history[-1]
    final_accuracy_ma = accuracy_ma[-1]
    max_fitness = max(best_fitness_history)
    max_accuracy = max(accuracy_history)
    max_accuracy_ma = max(accuracy_ma)
    
    print(f"\n训练统计:")
    print(f"最终 Error: {-final_fitness:.6f}")
    print(f"最终 Accuracy: {final_accuracy:.4f}")
    print(f"最终 Accuracy (MA-20): {final_accuracy_ma:.4f}")
    print(f"最低 Error: {-max_fitness:.6f}")
    print(f"最高 Accuracy: {max_accuracy:.4f}")
    print(f"最高 Accuracy (MA-20): {max_accuracy_ma:.4f}")
    
    plt.close()

def train_step(state, batch_x, batch_y, generation=0):
    # 1. 采样噪声
    sampling(state.model)
    
    # 2. 前向传播
    outputs = populated_noise_fwd(state.model, batch_x)
    
    # 3. 计算适应度
    fitness = compute_fitness(outputs, batch_y)
    avg_fitness = jnp.mean(fitness)
    
    # 4. 排序适应度
    fitness_ranked = centered_rank(fitness)
    
    # 5. 计算梯度
    grads = calculate_gradients(state.model, fitness_ranked)
    
    # 6. 更新参数
    state.update(grads)
    
    return avg_fitness, grads, outputs, fitness

def main():
    # 极端测试配置：更大的种群和更多代数
    config = {
        'input_size': 1, 'hidden_size': 256, 'output_size': 4,
        'popsize': 2000, 'generations': 500, 'learning_rate': 0.05, 'sigma': 0.04
    }
    
    print(f"开始极端规模测试: 种群大小={config['popsize']}, 代数={config['generations']}, 隐藏层大小={config['hidden_size']}")
    print("JIT优化已启用，预期显著提升性能...\n")
    
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
    
    # 记录训练过程中的指标
    generations = []
    best_fitness_history = []
    accuracy_history = []
    
    for gen in range(1, config['generations'] + 1):
        avg_fitness, grads, outputs, fitness = train_step(state, train_data, train_targets, gen)
        
        # 使用JIT优化的指标计算
        best_idx, best_logits, accuracy = compute_metrics(outputs, train_targets, fitness)
        
        # 记录数据用于可视化
        generations.append(gen)
        best_fitness_history.append(float(fitness[best_idx]))
        accuracy_history.append(float(accuracy))
        
        if gen % 10 == 0 or gen <= 5:
            print(f"Gen {gen:3d}: avg={avg_fitness:8.6f}, accuracy={accuracy:.4f}, best={fitness[best_idx]:.6f}")
        elif gen == config['generations']:
            print(f"Final Gen {gen}: avg={avg_fitness:8.6f}, accuracy={accuracy:.4f}, best={fitness[best_idx]:.6f}")
    
    # 创建可视化图表
    print("\n生成训练过程可视化图表...")
    create_training_visualization(generations, best_fitness_history, accuracy_history)

if __name__ == "__main__":
    main()