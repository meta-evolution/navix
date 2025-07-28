"""ES algorithm for Navix navigation training using neurogenesistape."""

import argparse
import sys
import os

# GPU设备配置（必须在JAX导入前）
def setup_gpu_from_args():
    """Set GPU from command-line args before JAX import."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            try:
                gpu_id = int(sys.argv[i + 1])
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                print(f"Using GPU {gpu_id}")
                return gpu_id
            except ValueError:
                print(f"Invalid GPU ID: {sys.argv[i + 1]}")
    return None

setup_gpu_from_args()

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import navix as nx
from flax import nnx
import optax
import numpy as np

try:
    import ngt
except ImportError:
    import neurogenesistape as ngt

from neurogenesistape.modules.es.nn import ES_RNN
from neurogenesistape.modules.es.optimizer import ES_Optimizer
from neurogenesistape.modules.es.training import ESConfig
from neurogenesistape.modules.evolution import sampling, calculate_gradients, centered_rank
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable

# Define state_axes and populated_noise_fwd functions for RNN with hidden state
others = (nnx.RngCount, nnx.RngKey)
state_axes = nnx.StateAxes({nnx.Param: None, Populated_Variable: 0, Grad_variable: None, nnx.Variable: None, others: None, ...: None})

@nnx.vmap(in_axes=(state_axes, 0, 0))
def populated_noise_fwd_step(model: nnx.Module, input, hidden):
    """Single-step forward pass for RNN with hidden state maintenance."""
    return model.forward_step(input, hidden)

@nnx.vmap(in_axes=(state_axes, None))
def populated_noise_fwd(model: nnx.Module, input):
    # Reshape single-step input to sequence format: (batch_size, 1, input_size)
    input_seq = jnp.expand_dims(input, axis=1)
    output_seq = model(input_seq)
    # Return only the output for the single timestep: (batch_size, output_size)
    return output_seq[:, 0, :]

# Use NGT's ES_RNN directly for agent
def initialize_batch_environments(env, popsize, base_key):
    """Initialize batch of environments in parallel."""
    env_keys = jax.random.split(base_key, popsize)
    return jax.vmap(env.reset)(env_keys)

def evaluate_population_fitness(env, agent, batch_timesteps, max_steps=500, generation=None, seed=42):
    """Evaluate population fitness in batch with proper RNN hidden state maintenance."""
    pop_size = agent.i2h.kernel.popsize
    step_counts = jnp.zeros(pop_size, dtype=jnp.int32)
    done_flags = jnp.zeros(pop_size, dtype=jnp.bool_)
    current_timesteps = batch_timesteps
    
    # Initialize hidden states for all population members
    # Shape: (pop_size, batch_size, hidden_size)
    hidden_states = agent.init_hidden(pop_size)
    
    @jax.jit
    def preprocess_single(timestep):
        obs = timestep.observation.flatten()
        pos = timestep.state.entities['player'].position.squeeze().astype(jnp.float32) / 15.0
        dir_ = jnp.array([timestep.state.entities['player'].direction.squeeze() / 3.0])
        return jnp.concatenate([obs, pos, dir_])

    @jax.jit
    def step_batch(timesteps, actions):
        next_timesteps = jax.vmap(env.step)(timesteps, actions)
        goals_reached = jax.vmap(lambda ts: ts.state.events.goal_reached.happened)(next_timesteps)
        return next_timesteps, goals_reached

    # Visualization removed for performance

    for step in range(max_steps):
        if step % 50 == 0 or step == max_steps - 1:
            print(f"\rStep {step + 1}/{max_steps}", end="", flush=True)

        batch_obs = jax.vmap(preprocess_single)(current_timesteps)
        
        # Use single-step RNN forward pass with hidden state maintenance
        logits, new_hidden_states = populated_noise_fwd_step(agent, batch_obs, hidden_states)
        
        # Update hidden states only for agents that are still active (not done)
        hidden_states = jnp.where(done_flags[:, None], hidden_states, new_hidden_states)
        
        # logits shape is (pop_size, action_size) - direct output from RNN
        member_logits = logits

        # 测试代码（已注释）
        # other_prob = 0.1 / 6
        # forced_probs = jnp.full_like(member_logits, other_prob)
        # forced_probs = forced_probs.at[:, 2].set(0.9)
        
        # 正常的基于RNN网络输出的概率采样
        action_probs = jax.nn.softmax(member_logits, axis=-1)
        action_keys = jax.random.split(jax.random.PRNGKey(step + (generation * 1000 if generation else 0)), pop_size)
        actions = jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(action_keys, member_logits)

        # Visualization code removed

        next_timesteps, goals_reached = step_batch(current_timesteps, actions)
        next_timesteps = jax.vmap(lambda o, n, d: jax.tree.map(lambda old, new: jnp.where(d, old, new), o, n))(current_timesteps, next_timesteps, done_flags)

        # 更新step_counts：如果agent刚刚到达目标，记录步数
        newly_done = goals_reached & ~done_flags
        step_counts = jnp.where(newly_done, step + 1, step_counts)
        done_flags = done_flags | goals_reached
        current_timesteps = next_timesteps

        if jnp.all(done_flags):
            print()
            break

    if not jnp.all(done_flags):
        print()

    path_lengths = jnp.where(done_flags, step_counts, max_steps)
    return -path_lengths


def plot_fitness_histogram(fitness_scores, generation, bins=10):
    """绘制fitness分布的命令行直方图并返回直方图数据"""
    fitness_array = np.array(fitness_scores)
    # min_val, max_val = fitness_array.min(), fitness_array.max()
    min_val, max_val = -500, 0
    total_count = len(fitness_array)
    
    # 强制计算直方图，即使所有值相同
    if min_val == max_val:
        # 当所有值相同时，创建一个单一区间的直方图
        hist = np.array([total_count] + [0] * (bins - 1))
        bin_edges = np.linspace(min_val - 0.1, max_val + 0.1, bins + 1)
        print(f"[Gen {generation}] Fitness Histogram: All values = {min_val:.2f}")
    else:
        # 计算直方图
        hist, bin_edges = np.histogram(fitness_array, bins=bins, range=(min_val, max_val))
    
    max_count = hist.max()
    
    print(f"[Gen {generation}] Fitness Histogram (range: {min_val:.2f} to {max_val:.2f}):")
    
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
    
    # 返回直方图数据：(hist, bin_edges, min_val, max_val)
    return hist, bin_edges, min_val, max_val


def train_step_navix(env, state, max_steps=500, generation=None, seed=42):
    """Perform single ES training step."""
    popsize = state.model.i2h.kernel.popsize

    if generation is None:
        numpy_seed = np.random.randint(0, 2**31)
    else:
        temp_state = np.random.get_state()
        np.random.seed(42 + generation * 1000)
        numpy_seed = np.random.randint(0, 2**31)
        np.random.set_state(temp_state)

    base_key = jax.random.PRNGKey(numpy_seed)
    batch_timesteps = initialize_batch_environments(env, popsize, base_key)

    fitness_scores = evaluate_population_fitness(env, state.model, batch_timesteps, max_steps, generation, seed)
    
    fitness_ranked = centered_rank(fitness_scores)
    
    grads = calculate_gradients(state.model, fitness_ranked)
    

    state.update(grads)
    
    return jnp.mean(fitness_scores), jnp.max(fitness_scores), fitness_scores


# Simplified ES using NGT components
def main():
    """Main ES training function."""
    parser = argparse.ArgumentParser(description='ES Navix Navigation with NGT')
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--gen", type=int, default=100, help="Generations")
    parser.add_argument("--pop", type=int, default=200, help="Population size (even)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="Noise std")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps")
    parser.add_argument("--test", action='store_true', help="Test mode")
    parser.add_argument("--gpu", type=int, help="GPU ID")
    # Visualization parameter removed
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--env", type=str, default="Navix-Empty-Random-6x6-v0", help="Environment name")

    args = parser.parse_args()
    np.random.seed(args.seed)
    print(f"Numpy seed: {args.seed}")

    if args.test:
        args.pop = 20
        args.gen = 10
        args.max_steps = 5
        print("Test mode enabled")

    if args.pop % 2 != 0:
        args.pop += 1
        print(f"Adjusted pop to {args.pop} (even)")

    print(f"JAX on {jax.default_backend()} ({len(jax.devices('gpu'))} GPUs)")
    print(f"Devices: {jax.devices()}")

    cfg = ESConfig(generations=args.gen, pop_size=args.pop, lr=args.lr, sigma=args.sigma)
    print("----- Config -----")
    print(f"Env: {args.env}")
    print(f"Hidden: {args.hidden}")
    print(f"Gens: {args.gen}")
    print(f"Pop: {args.pop}")
    print(f"LR: {args.lr}")
    print(f"Sigma: {args.sigma}")
    print(f"Max steps: {args.max_steps}")
    print("------------------")

    from navix import transitions
    env = nx.make(args.env, observation_fn=nx.observations.symbolic,
    transitions_fn=transitions.deterministic_transition,
    )
    numpy_seed = np.random.randint(0, 2**31)
    sample_key = jax.random.PRNGKey(numpy_seed)
    sample_timestep = env.reset(sample_key)

    # Visualization check removed

    total_obs_size = sample_timestep.observation.size + 2 + 1
    action_size = env.action_space.maximum.item() + 1
    print(f"Obs size: {total_obs_size}, Actions: {action_size}")

    numpy_seed = np.random.randint(0, 2**31)
    rngs = nnx.Rngs(numpy_seed)
    agent = ES_RNN(total_obs_size, args.hidden, action_size, rngs)
    agent.set_attributes(popsize=cfg.pop_size, noise_sigma=cfg.sigma)

    tx = optax.chain(optax.scale(-1), optax.sgd(cfg.lr))
    state = ES_Optimizer(agent, tx, wrt=nnx.Param)

    print(f"\nTraining: gens={cfg.generations}, pop={cfg.pop_size}")
    print("=" * 70)

    best_fitness = float('-inf')
    generations = []
    fitnesses = []
    best_fitnesses = []
    
    # 用于保存每一代的直方图数据
    histogram_data = []
    bins = 30  # 直方图区间数

    for g in range(1, cfg.generations + 1):
        sampling(state.model)
        avg_fitness, current_best_fitness, fitness_scores = train_step_navix(env, state, args.max_steps, g, args.seed)

        # 记录当前种群中的最佳个体适应度
        best_fitness = current_best_fitness

        print(f"[Gen {g:4d}] avg={avg_fitness:8.4f} best={best_fitness:8.4f}")
        
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

        generations.append(g)
        fitnesses.append(float(avg_fitness))
        best_fitnesses.append(float(best_fitness))

    print("=" * 70)
    print(f"Completed! Best: {best_fitness:.4f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Plot generation removed - keeping only data saving
    print(f"Training completed - no plot generated")

    data_filename = f"es_navix_h{args.hidden}_g{args.gen}_p{args.pop}_data.txt"
    with open(results_dir / data_filename, 'w') as f:
        f.write("# ES Navix Results\n")
        f.write(f"# Hidden: {args.hidden}, Gens: {args.gen}, Pop: {args.pop}\n")
        f.write(f"# Best: {best_fitness:.4f}\n")
        f.write("# Gen\tAvg\tBest\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]}\t{fitnesses[i]:.6f}\t{best_fitnesses[i]:.6f}\n")
    print(f"Data saved: {data_filename}")
    
    # 保存直方图数据到results文件夹
    # 创建包含所有直方图数据的numpy数组
    histogram_tensor = np.array([data['hist'] for data in histogram_data])
    
    # 保存直方图张量和相关元数据
    histogram_file = results_dir / f"fitness_histograms_{args.env.replace('-', '_')}_{args.gen}gen_{args.pop}pop.npz"
    np.savez(histogram_file, 
             histogram_tensor=histogram_tensor,
             generations=[data['generation'] for data in histogram_data],
             bin_edges=[data['bin_edges'] for data in histogram_data],
             min_vals=[data['min_val'] for data in histogram_data],
             max_vals=[data['max_val'] for data in histogram_data],
             all_fitness_scores=[data['fitness_scores'] for data in histogram_data])
    
    print(f"Fitness histograms saved to: {histogram_file}")


if __name__ == '__main__':
    main()