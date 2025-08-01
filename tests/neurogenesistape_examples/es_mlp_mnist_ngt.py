"""Example of using ES-MLP on MNIST with the NGT package alias."""

import argparse
import sys
import os

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
from pathlib import Path

# Add the tests directory to Python path to use local neurogenesistape instead of pip-installed version
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import neurogenesistape from local tests directory as ngt
import neurogenesistape as ngt

# Import components
from ngt import (
    ES_MLP, ES_Optimizer, ESConfig,
    train_step, evaluate, load_mnist
)


def main():
    # ----- Command Line Arguments -----
    parser = argparse.ArgumentParser(description='MNIST ES-MLP Classifier using NGT package')
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size (default: 256)")
    parser.add_argument("--gen", type=int, default=200, help="Number of generations (default: 200)")
    parser.add_argument("--pop", type=int, default=2000, help="Population size (default: 2000, must be even)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="Initial noise std")
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
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # ----- Model Initialization -----
    print(f"\nInitializing model: input={x_train.shape[1]}, hidden={args.hidden}, output=10")
    
    # Create model with 3 layers: input -> hidden -> hidden -> output
    layer_sizes = [x_train.shape[1], args.hidden, args.hidden, 10]
    
    # Use fixed random seed for reproducibility
    from flax import nnx
    rngs = nnx.Rngs(42)
    model = ES_MLP(layer_sizes, rngs)
    model.set_attributes(popsize=cfg.pop_size, noise_sigma=cfg.sigma)
    
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
    
    for g in range(1, cfg.generations + 1):
        # Create batch
        rng = jax.random.PRNGKey(g)  # New key for each generation
        idx = jax.random.choice(rng, x_train.shape[0], (cfg.batch_train,), replace=False)
        batch_x = x_train[idx]
        batch_y = y_train[idx]
        
        # Perform single training step using JIT-compiled function
        train_fitness = train_step(state, batch_x, batch_y)
        
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
    filename = f"es_mlp_mnist_h{args.hidden}_g{args.gen}_p{args.pop}_results.png"
    filepath = results_dir / filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Training results plot saved to: {filepath}")
    
    # Also save the raw data as a text file
    data_filename = f"es_mlp_mnist_h{args.hidden}_g{args.gen}_p{args.pop}_data.txt"
    data_filepath = results_dir / data_filename
    
    with open(data_filepath, 'w') as f:
        f.write("# ES-MLP MNIST Training Results\n")
        f.write(f"# Hidden size: {args.hidden}, Generations: {args.gen}, Population: {args.pop}\n")
        f.write(f"# Final best accuracy: {best_acc*100:.2f}%\n")
        f.write("# Generation\tTrain_Fitness\tTest_Accuracy\tBest_Accuracy\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]}\t{train_fitnesses[i]:.6f}\t{test_accuracies[i]:.6f}\t{best_accuracies[i]:.6f}\n")
    
    print(f"Training data saved to: {data_filepath}")
    
    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()