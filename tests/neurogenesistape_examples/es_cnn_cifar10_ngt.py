"""Example of using ES-CNN on CIFAR-10 with the NGT package alias."""

import argparse
import sys
import os
from pathlib import Path

# Add the tests directory to Python path to use local neurogenesistape instead of pip-installed version
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# GPU设备配置（必须在JAX导入前）
def setup_gpu_from_args():
    """从命令行参数设置GPU，必须在JAX导入前调用"""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            try:
                gpu_id = int(sys.argv[i + 1])
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")
                return gpu_id
            except ValueError:
                print(f"Warning: Invalid GPU ID '{sys.argv[i + 1]}', using default device")
    return None

# 设置GPU（必须在JAX导入前）
setup_gpu_from_args()

import jax
import matplotlib.pyplot as plt
import numpy as np

# Import neurogenesistape from local tests directory as ngt
import neurogenesistape as ngt

# Import components
from ngt import (
    ES_CNN, ES_Optimizer, ESConfig,
    train_step, train_step_with_fitness, evaluate, load_cifar10
)
import numpy as np


def plot_fitness_histogram(fitness_scores, generation, bins=20):
    """绘制当前代的fitness分布直方图并返回直方图数据。
    
    Args:
        fitness_scores: 当前代所有个体的适应度分数
        generation: 当前代数
        bins: 直方图区间数
        
    Returns:
        tuple: (hist, bin_edges, min_val, max_val)
    """
    fitness_array = np.array(fitness_scores)
    total_count = len(fitness_array)
    
    # 使用固定范围[-7, 0]来保持直方图的一致性
    min_val = -20.0
    max_val = 0.0
    
    # 计算直方图（使用固定范围）
    hist, bin_edges = np.histogram(fitness_array, bins=bins, range=(min_val, max_val))
    
    # 获取实际数据的最小值和最大值用于显示
    actual_min = float(fitness_array.min())
    actual_max = float(fitness_array.max())
    
    max_count = hist.max()
    
    print(f"[Gen {generation}] Fitness Histogram (fixed range: {min_val:.2f} to {max_val:.2f}, actual: {actual_min:.2f} to {actual_max:.2f}):")
    
    # 绘制直方图
    bar_width = 40  # 最大条形宽度
    for i in range(bins):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]
        count = hist[i]
        percentage = (count / total_count) * 100 if total_count > 0 else 0
        
        # 计算条形长度
        if max_count > 0:
            bar_length = int((count / max_count) * bar_width)
        else:
            bar_length = 0
        
        # 绘制条形（使用固定宽度格式对齐）
        bar = '█' * bar_length + '░' * (bar_width - bar_length)
        print(f"  [{left_edge:7.2f}-{right_edge:7.2f}] {bar} {percentage:5.1f}% ({count})")
    
    print()
    
    # 返回直方图数据：(hist, bin_edges, actual_min, actual_max)
    return hist, bin_edges, actual_min, actual_max


def main():
    # ----- Command Line Arguments -----
    parser = argparse.ArgumentParser(description='CIFAR-10 ES-CNN Classifier using NGT package')
    parser.add_argument("--gen", type=int, default=200, help="Number of generations (default: 200)")
    parser.add_argument("--pop", type=int, default=2000, help="Population size (default: 2000, must be even)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="Initial noise std")
    parser.add_argument("--min_sigma", type=float, default=0.01, help="Minimum noise std for sigma decay (default: 0.01)")
    parser.add_argument("--enable_sigma_decay", action="store_true", help="Enable sigma decay during training (default: False)")
    parser.add_argument("--gpu", type=int, help="GPU device ID to use (e.g., 0, 1, 2, 3)")

    args = parser.parse_args()

    # ----- Device Information -----
    devices = jax.devices()
    print(f"JAX available devices: {devices}")

    # ----- Configuration -----
    cfg = ESConfig(generations=args.gen, pop_size=args.pop, lr=args.lr, sigma=args.sigma)
    print("----- Configuration -----")
    print(cfg)
    print("-------------------------")
    
    # ----- Data Loading -----
    print("Loading CIFAR-10 dataset...")
    x_train, y_train, x_test, y_test = load_cifar10()
    
    # Reshape data for CNN: (N, H, W, C) format
    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    
    # ----- Model Initialization -----
    print(f"\nInitializing CNN model: input_channels=3, num_classes=10")
    
    # Use fixed random seed for reproducibility
    from flax import nnx
    rngs = nnx.Rngs(42)
    model = ES_CNN(input_channels=3, num_classes=10, rngs=rngs)
    model.set_attributes(popsize=cfg.pop_size, noise_sigma=cfg.sigma, min_sigma=args.min_sigma)
    
    # Enable sigma decay for all ES modules (configurable via command line)
    model.enable_sigma_decay(args.enable_sigma_decay)
    
    # Initialize optimizer
    import optax
    tx = optax.chain(optax.scale(-1), optax.sgd(cfg.lr))
    state = ES_Optimizer(model, tx, wrt=nnx.Param)

    print(f"\nStarting ES training: generations={cfg.generations}, population={cfg.pop_size}, learning rate={cfg.lr}")
    print("=" * 70)

    # ----- Main Training Loop -----
    best_acc = 0.0
    
    # Lists to store training data for plotting
    generations = []
    train_fitnesses = []
    test_accuracies = []
    best_accuracies = []
    
    # 用于保存每一代的直方图数据
    histogram_data = []
    bins = 60  # 直方图区间数
    
    for g in range(1, cfg.generations + 1):

        # 打印当前代的sigma值（从ES模块中提取）
        current_sigma = model.conv1.kernel.noise_sigma  # 从第一层的ES_Tape中获取sigma值

        # # 在训练后期（90%进度时）将sigma增大一倍
        # if current_sigma < 0.05:
        #     # 将所有层的sigma值增大
        #     for layer in model.all_layers:
        #         if hasattr(layer, 'kernel') and hasattr(layer.kernel, 'noise_sigma'):
        #             layer.kernel.noise_sigma = args.sigma
        #         if hasattr(layer, 'bias') and hasattr(layer.bias, 'noise_sigma'):
        #             layer.bias.noise_sigma = args.sigma
        #     print(f"[Gen {g:4d}] *** SIGMA BOOST: Increased sigma by 2x at 90% progress ***")
        
        # 打印当前代的sigma值（从ES模块中提取）
        current_sigma = model.conv1.kernel.noise_sigma  # 从第一层的ES_Tape中获取sigma值
        print(f"[Gen {g:4d}] Current sigma: {current_sigma:.6f}")
        
        # Create batch
        rng = jax.random.PRNGKey(g)  # New key for each generation
        idx = jax.random.choice(rng, x_train.shape[0], (cfg.batch_train,), replace=False)
        batch_x = x_train[idx]
        batch_y = y_train[idx]
        
        # Perform single training step using JIT-compiled function with fitness scores
        train_fitness, fitness_scores = train_step_with_fitness(state, batch_x, batch_y)
        
        # 绘制当前代的fitness分布直方图并收集数据
        hist, bin_edges, min_val, max_val = plot_fitness_histogram(fitness_scores, g, bins)
        
        # 保存直方图数据
        histogram_data.append({
            'generation': g,
            'hist': hist,
            'bin_edges': bin_edges,
            'min_val': min_val,
            'max_val': max_val,
            'fitness_scores': np.array(fitness_scores)
        })
        
        # Periodically evaluate and report
        if g % 10 == 0 or g == 1:
            # Evaluate on test set
            acc = evaluate(state.model, x_test, y_test)
            if acc > best_acc:
                best_acc = acc
            
            print(f"[Gen {g:4d}]     fitness={float(train_fitness):.4f}   "
                  f"test-acc={acc*100:5.2f}%   best={best_acc*100:.2f}%")
            
            # Store data for plotting
            generations.append(g)
            train_fitnesses.append(float(train_fitness))
            test_accuracies.append(float(acc))
            best_accuracies.append(float(best_acc))
        
        print("max_val ======= ", max_val)

    print("=" * 70)
    print(f"Training completed! Best test accuracy: {best_acc*100:.2f}%")
    
    # ----- Plot and Save Results -----
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training fitness
    ax1.plot(generations, train_fitnesses, 'b-', linewidth=2, label='Training Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Training Fitness Over Generations')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Test accuracy
    ax2.plot(generations, [acc * 100 for acc in test_accuracies], 'g-', linewidth=2, label='Test Accuracy')
    ax2.plot(generations, [acc * 100 for acc in best_accuracies], 'r--', linewidth=2, label='Best Accuracy')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy Over Generations')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Generate filename with parameters
    filename = f"es_cnn_g{args.gen}_p{args.pop}_results.png"
    filepath = results_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Training results plot saved to: {filepath}")
    
    # Also save the raw data as a text file
    data_filename = f"es_cnn_g{args.gen}_p{args.pop}_data.txt"
    data_filepath = results_dir / data_filename
    
    with open(data_filepath, 'w') as f:
        f.write("# ES-CNN CIFAR-10 Training Results\n")
        f.write(f"# Generations: {args.gen}, Population: {args.pop}\n")
        f.write(f"# Final best accuracy: {best_acc*100:.2f}%\n")
        f.write("# Generation\tTrain_Fitness\tTest_Accuracy\tBest_Accuracy\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]}\t{train_fitnesses[i]:.6f}\t{test_accuracies[i]:.6f}\t{best_accuracies[i]:.6f}\n")
    
    print(f"Training data saved to: {data_filepath}")
    
    # 保存直方图数据到results文件夹
    # 创建包含所有直方图数据的numpy数组
    histogram_tensor = np.array([data['hist'] for data in histogram_data])
    
    # 保存直方图张量和相关元数据
    histogram_file = results_dir / f"fitness_histograms_cifar10_cnn_{args.gen}gen_{args.pop}pop.npz"
    np.savez(histogram_file, 
             histogram_tensor=histogram_tensor,
             generations=[data['generation'] for data in histogram_data],
             bin_edges=[data['bin_edges'] for data in histogram_data],
             min_vals=[data['min_val'] for data in histogram_data],
             max_vals=[data['max_val'] for data in histogram_data],
             all_fitness_scores=[data['fitness_scores'] for data in histogram_data])
    
    print(f"Fitness histograms saved to: {histogram_file}")
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()