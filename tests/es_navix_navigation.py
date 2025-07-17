"""ES algorithm for Navix navigation training using neurogenesistape."""

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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import navix as nx
from flax import nnx
import optax

# Try to import using the ngt alias, fall back to neurogenesistape if needed
try:
    import ngt
except ImportError:
    print("Warning: Could not import ngt directly, using neurogenesistape instead")
    import neurogenesistape as ngt

# Import ES components
from ngt import (
    ES_MLP, ES_Optimizer, ESConfig,
    sampling, populated_noise_fwd, calculate_gradients, centered_rank
)


# Using NGT's ES_MLP directly - no need for custom agent class


def initialize_batch_environments(env, popsize, base_key):
    """Initialize a batch of environments for parallel evaluation.
    
    Args:
        env: Navix environment instance
        popsize: Population size (number of environments)
        base_key: Base random key for environment initialization
        
    Returns:
        Batch of initialized environment timesteps
    """
    env_keys = jax.random.split(base_key, popsize)
    
    @jax.jit
    def reset_single_env(key):
        return env.reset(key)
    
    # Initialize all environments in parallel
    batch_timesteps = jax.vmap(reset_single_env)(env_keys)
    return batch_timesteps


def evaluate_population_fitness(env, agent, batch_timesteps, max_steps=500):
    """Evaluate fitness for the entire population using batch processing.
    
    Args:
        env: Navix environment instance
        agent: ES agent model
        batch_timesteps: Initial timesteps for all environments
        max_steps: Maximum steps per episode
        
    Returns:
        Array of fitness scores for each population member
    """
    # Get population size
    pop_size = agent.layers[0].kernel.popsize
    
    # Initialize step counters and done flags for all population members
    step_counts = jnp.zeros(pop_size, dtype=jnp.int32)
    done_flags = jnp.zeros(pop_size, dtype=jnp.bool_)
    
    # Current timesteps for all population members
    current_timesteps = batch_timesteps
    
    @jax.jit
    def preprocess_single(timestep):
        """Preprocess a single observation."""
        obs = timestep.observation
        position = timestep.state.entities['player'].position.squeeze()
        direction = timestep.state.entities['player'].direction.squeeze()
        
        flat_obs = obs.flatten()
        norm_pos = position.astype(jnp.float32) / 15.0
        norm_dir = jnp.array([direction / 3.0], dtype=jnp.float32)
        return jnp.concatenate([flat_obs, norm_pos, norm_dir])
    
    @jax.jit
    def step_single(timestep, action):
        """Step a single environment."""
        return env.step(timestep, action)
    
    @jax.jit
    def process_step_batch(timesteps, actions, done_flags):
        """Combined function for preprocessing, stepping, and updating batch environments."""
        # Preprocess observations
        batch_obs = jax.vmap(preprocess_single)(timesteps)
        
        # Step environments
        next_timesteps = jax.vmap(step_single)(timesteps, actions)
        
        # Check goal reached
        goals_reached = jax.vmap(lambda ts: ts.state.events.goal_reached.happened)(next_timesteps)
        
        # Update timesteps only for non-done environments
        updated_timesteps = jax.vmap(
            lambda old_ts, new_ts, done: jax.tree.map(
                lambda old, new: jnp.where(done, old, new), old_ts, new_ts
            )
        )(timesteps, next_timesteps, done_flags)
        
        return batch_obs, updated_timesteps, goals_reached
    
    # Pre-compile functions with static shapes to reduce compilation overhead
    sample_obs = jax.vmap(preprocess_single)(current_timesteps)
    _ = populated_noise_fwd(agent, sample_obs)  # Trigger initial compilation
    
    # Main evaluation loop using optimized batch processing
    for step in range(max_steps):
        # Display progress every 10 steps to reduce I/O overhead
        if step % 10 == 0 or step == max_steps - 1:
            print(f"\rStep {step + 1}/{max_steps}", end="", flush=True)
        
        # Preprocess observations for all environments
        batch_obs = jax.vmap(preprocess_single)(current_timesteps)
        
        # Get actions for all population members using populated_noise_fwd
        logits = populated_noise_fwd(agent, batch_obs)  # (pop_size, pop_size, action_size)
        
        # Extract diagonal elements to get each member's action for its own environment
        member_logits = jnp.diagonal(logits, axis1=0, axis2=1).T  # (pop_size, action_size)
        actions = jnp.argmax(member_logits, axis=-1)  # (pop_size,)
        
        # Use combined batch processing function
        _, next_timesteps, goals_reached = process_step_batch(current_timesteps, actions, done_flags)
        
        # Update done flags and step counts
        new_done = done_flags | goals_reached
        step_counts = jnp.where(~done_flags & goals_reached, step + 1, step_counts)
        done_flags = new_done
        
        # Update current timesteps
        current_timesteps = next_timesteps
        
        # Early termination if all environments are done
        if jnp.all(done_flags):
            print()  # New line after progress display
            break
    
    # Ensure new line after progress display if loop completes normally
    if not jnp.all(done_flags):
        print()
    
    # Return fitness (negative path length, so shorter paths have higher fitness)
    path_lengths = jnp.where(done_flags, step_counts, max_steps)
    return -path_lengths


def train_step_navix(env, state, max_steps=500):
    """Single training step for Navix navigation using NGT components.
    
    Args:
        env: Navix environment instance
        state: ES optimizer state
        max_steps: Maximum steps per episode
        
    Returns:
        Average fitness
    """
    # Get population size
    popsize = state.model.layers[0].kernel.popsize
    
    # Initialize batch environments with fixed seed for consistency
    base_key = jax.random.PRNGKey(42)
    batch_timesteps = initialize_batch_environments(env, popsize, base_key)
    
    # Evaluate population fitness
    fitness_scores = evaluate_population_fitness(env, state.model, batch_timesteps, max_steps)
    
    # Apply centered rank transformation
    fitness_ranked = centered_rank(fitness_scores)
    
    # Calculate gradients using NGT's built-in function
    grads = calculate_gradients(state.model, fitness_ranked)
    
    # Update parameters
    state.update(grads)
    
    return jnp.mean(fitness_scores)


# Simplified ES implementation using NGT's built-in components


def main():
    """Main training function using NGT ES components."""
    # Command line arguments
    parser = argparse.ArgumentParser(description='ES-based Navix Navigation Training using NGT')
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size (default: 128)")
    parser.add_argument("--gen", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--pop", type=int, default=200, help="Population size (default: 200, must be even)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Initial noise std (default: 0.1)")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode (default: 500)")
    parser.add_argument("--test", action='store_true', help="Run test mode (pop=20, gen=10)")
    parser.add_argument("--gpu", type=int, help="GPU device ID to use")
    
    args = parser.parse_args()
    
    # Test mode override
    if args.test:
        args.pop = 20
        args.gen = 10
        args.max_steps = 50
        print("Running in test mode: pop=20, gen=10, max_steps=50")
    
    # Device information
    devices = jax.devices()
    print(f"JAX {jax.__version__} on {jax.default_backend()} ({len(jax.devices('gpu'))} GPUs)")
    print(f"Available devices: {devices}")
    
    # Configuration
    cfg = ESConfig(generations=args.gen, pop_size=args.pop, lr=args.lr, sigma=args.sigma)
    print("----- Configuration -----")
    print(f"Environment: Navix-Dynamic-Obstacles-16x16-v0")
    print(f"Hidden size: {args.hidden}")
    print(f"Generations: {args.gen}")
    print(f"Population size: {args.pop}")
    print(f"Learning rate: {args.lr}")
    print(f"Noise sigma: {args.sigma}")
    print(f"Max steps per episode: {args.max_steps}")
    print("-------------------------")
    
    # Environment setup for dimension calculation
    env = nx.make('Navix-Dynamic-Obstacles-16x16-v0', observation_fn=nx.observations.symbolic)
    sample_key = jax.random.PRNGKey(0)
    sample_timestep = env.reset(sample_key)
    
    # Calculate observation size
    obs_size = sample_timestep.observation.size
    agent_pos_size = 2  # x, y position
    agent_dir_size = 1  # direction
    total_obs_size = obs_size + agent_pos_size + agent_dir_size
    action_size = 4  # Navix has 4 actions: forward, left, right, stay
    
    print(f"Observation size: {total_obs_size}, Action size: {action_size}")
    
    # Create ES agent using NGT's ES_MLP
    layer_sizes = [total_obs_size, args.hidden, args.hidden, action_size]
    rngs = nnx.Rngs(42)
    agent = ES_MLP(layer_sizes, rngs)
    agent.set_attributes(popsize=cfg.pop_size, noise_sigma=cfg.sigma)
    
    # Create optimizer
    tx = optax.chain(optax.scale(-1), optax.sgd(cfg.lr))
    state = ES_Optimizer(agent, tx, wrt=nnx.Param)
    
    print(f"\nStarting ES training: generations={cfg.generations}, population={cfg.pop_size}")
    print("=" * 70)
    
    # Training loop
    best_fitness = float('-inf')
    generations = []
    fitnesses = []
    best_fitnesses = []
    
    for g in range(1, cfg.generations + 1):
        # Sample noise for population at the beginning of each generation
        sampling(state.model)
        
        # Training step (evaluate the sampled population)
        avg_fitness = train_step_navix(env, state, args.max_steps)
        
        # Track best fitness
        if avg_fitness > best_fitness:
            best_fitness = avg_fitness
        
        # Report progress
        avg_path_length = -float(avg_fitness)
        best_path_length = -float(best_fitness)
        
        print(f"[Gen {g:4d}] avg_fitness={float(avg_fitness):8.4f} best_fitness={float(best_fitness):8.4f}")
        
        generations.append(g)
        fitnesses.append(float(avg_fitness))
        best_fitnesses.append(float(best_fitness))
    
    print("=" * 70)
    print(f"Training completed! Best fitness: {best_fitness:.4f}")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average fitness
    ax1.plot(generations, fitnesses, 'b-', linewidth=2, label='Average Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Average Fitness Over Generations')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Best fitness
    ax2.plot(generations, best_fitnesses, 'r--', linewidth=2, label='Best Fitness')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Best Fitness Over Generations')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    mode_str = "test" if args.test else "train"
    filename = f"es_navix_h{args.hidden}_g{args.gen}_p{args.pop}_results.png"
    filepath = results_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Results plot saved to: {filepath}")
    
    # Save data
    data_filename = f"es_navix_h{args.hidden}_g{args.gen}_p{args.pop}_data.txt"
    data_filepath = results_dir / data_filename
    
    with open(data_filepath, 'w') as f:
        f.write("# ES Navix Navigation Training Results\n")
        f.write(f"# Hidden size: {args.hidden}, Generations: {args.gen}, Population: {args.pop}\n")
        f.write(f"# Final best fitness: {best_fitness:.4f}\n")
        f.write("# Generation\tAvg_Fitness\tBest_Fitness\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]}\t{fitnesses[i]:.6f}\t{best_fitnesses[i]:.6f}\n")
    
    print(f"Training data saved to: {data_filepath}")
    plt.show()


if __name__ == '__main__':
    main()