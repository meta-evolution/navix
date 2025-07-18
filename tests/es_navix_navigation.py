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
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
import navix as nx
from flax import nnx
import optax

import numpy as np

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


def evaluate_population_fitness(env, agent, batch_timesteps, max_steps=500, generation=None, seed=42):
    """Evaluate fitness for the entire population using batch processing.
    
    Args:
        env: Navix environment instance
        agent: ES agent model
        batch_timesteps: Initial timesteps for all environments
        max_steps: Maximum steps per episode
        generation: Current generation number for visualization
        
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
    
    # Setup visualization if enabled
    viz_env_id = None
    viz_dir = None
    viz_timestep = None
    if generation is not None:
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        # Create visualization directory
        viz_dir = Path("results") / f"visualization_gen_{generation}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a seed-based environment ID for visualization
        # This ensures different seeds visualize different environments
        temp_state = np.random.get_state()
        np.random.seed(seed)  # Use provided seed
        viz_env_id = np.random.randint(0, pop_size)  # Random environment based on seed
        np.random.set_state(temp_state)
        print(f"Visualizing environment {viz_env_id} for generation {generation} (seed-based selection)")
        
        # Initialize dedicated visualization timestep to track continuous state evolution
        viz_timestep = jax.tree.map(lambda x: x[viz_env_id], batch_timesteps)
    
    # Main evaluation loop using optimized batch processing
    for step in range(max_steps):
        # Display progress every 50 steps to reduce I/O overhead
        if step % 50 == 0 or step == max_steps - 1:
            print(f"\rStep {step + 1}/{max_steps}", end="", flush=True)
        
        # Preprocess observations for all environments
        batch_obs = jax.vmap(preprocess_single)(current_timesteps)
        
        # Get actions for all population members using populated_noise_fwd
        logits = populated_noise_fwd(agent, batch_obs)  # (pop_size, pop_size, action_size)
        
        # Extract diagonal elements to get each member's action for its own environment
        member_logits = jnp.diagonal(logits, axis1=0, axis2=1).T  # (pop_size, action_size)
        
        # EXPERIMENT: Force forward action (index 2) to have 90% probability for testing
        # Create custom probability distribution: forward=0.9, others=0.1/6 each
        forced_probs = jnp.zeros_like(member_logits)
        forced_probs = forced_probs.at[:, 2].set(0.9)  # Forward action gets 90%
        other_prob = 0.1 / 6  # Remaining 10% split among other 6 actions
        for i in range(7):
            if i != 2:  # Skip forward action
                forced_probs = forced_probs.at[:, i].set(other_prob)
        
        # Use forced probabilities instead of model logits
        action_keys = jax.random.split(jax.random.PRNGKey(step + generation * 1000 if generation else step), pop_size)
        actions = jax.vmap(lambda key, probs: jax.random.categorical(key, jnp.log(probs)))(action_keys, forced_probs)
        
        # Command line output for agent status (show every step for debugging)
        if viz_env_id is not None:
            current_pos = current_timesteps.state.entities['player'].position[viz_env_id].squeeze()
            current_dir = current_timesteps.state.entities['player'].direction[viz_env_id].squeeze()
            selected_action = actions[viz_env_id]
            print(f"\n[Step {step+1}] Env {viz_env_id}: Pos=({current_pos[0]},{current_pos[1]}), Dir={current_dir}, Action={selected_action}, Probs={forced_probs[viz_env_id]}")
            print(f"[Step {step+1}] Forward action (index 2) probability: {forced_probs[viz_env_id][2]:.1%}")
            print(f"[Step {step+1}] Original model forward probability: {jax.nn.softmax(member_logits[viz_env_id])[2]:.4%}")
        
        # Visualization: save image for the selected environment
        if viz_env_id is not None and not done_flags[viz_env_id]:
            # Get RGB observation from the dedicated visualization timestep
            rgb_obs = nx.observations.rgb(viz_timestep.state)
            
            # Save visualization image
            img_path = viz_dir / f"step_{step + 1:03d}.png"
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_obs)
            plt.axis('off')
            plt.title(f"Generation {generation}, Step {step + 1}, Environment #{viz_env_id}")
            plt.savefig(img_path, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Update visualization timestep with the action from the visualization environment
            viz_action = actions[viz_env_id]
            viz_timestep = env.step(viz_timestep, viz_action)
        
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


def train_step_navix(env, state, max_steps=500, generation=None, seed=42):
    """Single training step for Navix navigation using NGT components.
    
    Args:
        env: Navix environment instance
        state: ES optimizer state
        max_steps: Maximum steps per episode
        generation: Current generation number for visualization
        
    Returns:
        Average fitness
    """
    # Get population size
    popsize = state.model.layers[0].kernel.popsize
    
    # Initialize batch environments with generation-specific seed for consistency
    # Use different seeds for different generations but same seed within a generation
    # This ensures the same initial conditions for visualization tracking
    if generation is None:
        numpy_seed = np.random.randint(0, 2**31)
    else:
        # Use numpy to generate deterministic but different seeds for each generation
        temp_state = np.random.get_state()
        np.random.seed(42 + generation * 1000)
        numpy_seed = np.random.randint(0, 2**31)
        np.random.set_state(temp_state)
    base_key = jax.random.PRNGKey(numpy_seed)
    batch_timesteps = initialize_batch_environments(env, popsize, base_key)
    
    # Evaluate population fitness
    fitness_scores = evaluate_population_fitness(env, state.model, batch_timesteps, max_steps, generation, seed)
    
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
    parser.add_argument("--visualize", action='store_true', help="Enable visualization during training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set numpy random seed for reproducibility
    np.random.seed(args.seed)
    print(f"Set numpy random seed to: {args.seed}")
    
    # Test mode override
    if args.test:
        args.pop = 20
        args.gen = 10
        args.max_steps = 50
        print("Running in test mode: pop=20, gen=10, max_steps=50")
    
    # Ensure population size is even (required for symmetric noise in ES)
    if args.pop % 2 != 0:
        args.pop += 1
        print(f"Population size adjusted to {args.pop} (must be even for symmetric noise)")
    
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
    
    # Environment setup for dimension calculation with deterministic transitions
    from navix import transitions
    env = nx.make('Navix-Dynamic-Obstacles-16x16-v0', 
                  observation_fn=nx.observations.symbolic,
                  transitions_fn=transitions.deterministic_transition)
    # Use numpy to generate seed for environment sampling
    numpy_seed = np.random.randint(0, 2**31)
    sample_key = jax.random.PRNGKey(numpy_seed)
    sample_timestep = env.reset(sample_key)
    
    # Check if visualization is enabled
    if args.visualize:
        print("Visualization mode enabled - tracking environment 0 for consistent episode visualization")
    
    # Calculate observation size
    obs_size = sample_timestep.observation.size
    agent_pos_size = 2  # x, y position
    agent_dir_size = 1  # direction
    total_obs_size = obs_size + agent_pos_size + agent_dir_size
    
    # Get actual action space size from environment
    actual_action_size = env.action_space.maximum.item() + 1
    action_size = actual_action_size
    

    
    print(f"Observation size: {total_obs_size}, Action size: {action_size}")
    
    # Create ES agent using NGT's ES_MLP
    layer_sizes = [total_obs_size, args.hidden, args.hidden, action_size]
    # Use numpy to generate seed for model initialization
    numpy_seed = np.random.randint(0, 2**31)
    rngs = nnx.Rngs(numpy_seed)
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
        # Pass generation number for visualization if enabled
        avg_fitness = train_step_navix(env, state, args.max_steps, g if args.visualize else None, args.seed)
        
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