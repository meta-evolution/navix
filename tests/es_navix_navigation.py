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

from ngt import (
    ES_MLP, ES_Optimizer, ESConfig,
    sampling, populated_noise_fwd, calculate_gradients, centered_rank
)

# Use NGT's ES_MLP directly for agent
def initialize_batch_environments(env, popsize, base_key):
    """Initialize batch of environments in parallel."""
    env_keys = jax.random.split(base_key, popsize)
    return jax.vmap(env.reset)(env_keys)

def evaluate_population_fitness(env, agent, batch_timesteps, max_steps=500, generation=None, seed=42):
    """Evaluate population fitness in batch."""
    pop_size = agent.layers[0].kernel.popsize
    step_counts = jnp.zeros(pop_size, dtype=jnp.int32)
    done_flags = jnp.zeros(pop_size, dtype=jnp.bool_)
    current_timesteps = batch_timesteps

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

    sample_obs = jax.vmap(preprocess_single)(current_timesteps)
    _ = populated_noise_fwd(agent, sample_obs)

    viz_env_id = None
    viz_dir = None
    viz_timestep = None
    if generation is not None:
        viz_dir = Path("results") / f"visualization_gen_{generation}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        temp_state = np.random.get_state()
        np.random.seed(seed)
        viz_env_id = np.random.randint(0, pop_size)
        np.random.set_state(temp_state)
        print(f"Visualizing env {viz_env_id} for gen {generation}")
        viz_timestep = jax.tree.map(lambda x: x[viz_env_id], batch_timesteps)

    for step in range(max_steps):
        if step % 50 == 0 or step == max_steps - 1:
            print(f"\rStep {step + 1}/{max_steps}", end="", flush=True)

        batch_obs = jax.vmap(preprocess_single)(current_timesteps)
        logits = populated_noise_fwd(agent, batch_obs)
        member_logits = jnp.diagonal(logits, axis1=0, axis2=1).T

        other_prob = 0.1 / 6
        forced_probs = jnp.full_like(member_logits, other_prob)
        forced_probs = forced_probs.at[:, 2].set(0.9)

        action_keys = jax.random.split(jax.random.PRNGKey(step + (generation * 1000 if generation else 0)), pop_size)
        actions = jax.vmap(jax.random.categorical)(action_keys, jnp.log(forced_probs))

        if viz_env_id is not None:
            pos = current_timesteps.state.entities['player'].position[viz_env_id].squeeze()
            dir_ = current_timesteps.state.entities['player'].direction[viz_env_id].squeeze()
            act = actions[viz_env_id]
            print(f"\n[Step {step+1}] Env {viz_env_id}: Pos=({pos[0]},{pos[1]}), Dir={dir_}, Action={act}, Probs={forced_probs[viz_env_id]}")
            print(f"Forward prob: {forced_probs[viz_env_id][2]:.1%}, Model: {jax.nn.softmax(member_logits[viz_env_id])[2]:.4%}")

            if not done_flags[viz_env_id]:
                rgb = nx.observations.rgb(viz_timestep.state)
                img_path = viz_dir / f"step_{step + 1:03d}.png"
                plt.figure(figsize=(8, 8))
                plt.imshow(rgb)
                plt.axis('off')
                plt.title(f"Gen {generation}, Step {step + 1}, Env #{viz_env_id}")
                plt.savefig(img_path, bbox_inches='tight', dpi=100)
                plt.close()
                viz_timestep = env.step(viz_timestep, actions[viz_env_id])

        next_timesteps, goals_reached = step_batch(current_timesteps, actions)
        next_timesteps = jax.vmap(lambda o, n, d: jax.tree.map(lambda old, new: jnp.where(d, old, new), o, n))(current_timesteps, next_timesteps, done_flags)

        done_flags = done_flags | goals_reached
        step_counts = jnp.where(~done_flags & goals_reached, step + 1, step_counts)
        current_timesteps = next_timesteps

        if jnp.all(done_flags):
            print()
            break

    if not jnp.all(done_flags):
        print()

    path_lengths = jnp.where(done_flags, step_counts, max_steps)
    return -path_lengths


def train_step_navix(env, state, max_steps=500, generation=None, seed=42):
    """Perform single ES training step."""
    popsize = state.model.layers[0].kernel.popsize

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
    return jnp.mean(fitness_scores)


# Simplified ES using NGT components


def main():
    """Main ES training function."""
    parser = argparse.ArgumentParser(description='ES Navix Navigation with NGT')
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--gen", type=int, default=100, help="Generations")
    parser.add_argument("--pop", type=int, default=200, help="Population size (even)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise std")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps")
    parser.add_argument("--test", action='store_true', help="Test mode")
    parser.add_argument("--gpu", type=int, help="GPU ID")
    parser.add_argument("--visualize", action='store_true', help="Visualize")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    args = parser.parse_args()
    np.random.seed(args.seed)
    print(f"Numpy seed: {args.seed}")

    if args.test:
        args.pop = 20
        args.gen = 10
        args.max_steps = 50
        print("Test mode enabled")

    if args.pop % 2 != 0:
        args.pop += 1
        print(f"Adjusted pop to {args.pop} (even)")

    print(f"JAX on {jax.default_backend()} ({len(jax.devices('gpu'))} GPUs)")
    print(f"Devices: {jax.devices()}")

    cfg = ESConfig(generations=args.gen, pop_size=args.pop, lr=args.lr, sigma=args.sigma)
    print("----- Config -----")
    print(f"Env: Navix-Dynamic-Obstacles-16x16-v0")
    print(f"Hidden: {args.hidden}")
    print(f"Gens: {args.gen}")
    print(f"Pop: {args.pop}")
    print(f"LR: {args.lr}")
    print(f"Sigma: {args.sigma}")
    print(f"Max steps: {args.max_steps}")
    print("------------------")

    from navix import transitions
    env = nx.make('Navix-Dynamic-Obstacles-16x16-v0', observation_fn=nx.observations.symbolic, transitions_fn=transitions.deterministic_transition)
    numpy_seed = np.random.randint(0, 2**31)
    sample_key = jax.random.PRNGKey(numpy_seed)
    sample_timestep = env.reset(sample_key)

    if args.visualize:
        print("Visualization enabled")

    total_obs_size = sample_timestep.observation.size + 2 + 1
    action_size = env.action_space.maximum.item() + 1
    print(f"Obs size: {total_obs_size}, Actions: {action_size}")

    layer_sizes = [total_obs_size, args.hidden, args.hidden, action_size]
    numpy_seed = np.random.randint(0, 2**31)
    rngs = nnx.Rngs(numpy_seed)
    agent = ES_MLP(layer_sizes, rngs)
    agent.set_attributes(popsize=cfg.pop_size, noise_sigma=cfg.sigma)

    tx = optax.chain(optax.scale(-1), optax.sgd(cfg.lr))
    state = ES_Optimizer(agent, tx, wrt=nnx.Param)

    print(f"\nTraining: gens={cfg.generations}, pop={cfg.pop_size}")
    print("=" * 70)

    best_fitness = float('-inf')
    generations = []
    fitnesses = []
    best_fitnesses = []

    for g in range(1, cfg.generations + 1):
        sampling(state.model)
        avg_fitness = train_step_navix(env, state, args.max_steps, g if args.visualize else None, args.seed)

        if avg_fitness > best_fitness:
            best_fitness = avg_fitness

        print(f"[Gen {g:4d}] avg={avg_fitness:8.4f} best={best_fitness:8.4f}")

        generations.append(g)
        fitnesses.append(float(avg_fitness))
        best_fitnesses.append(float(best_fitness))

    print("=" * 70)
    print(f"Completed! Best: {best_fitness:.4f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(generations, fitnesses, 'b-', label='Avg Fitness')
    ax1.set_xlabel('Gen')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Avg Fitness')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(generations, best_fitnesses, 'r--', label='Best Fitness')
    ax2.set_xlabel('Gen')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Best Fitness')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    mode = "test" if args.test else "train"
    filename = f"es_navix_h{args.hidden}_g{args.gen}_p{args.pop}_results.png"
    plt.savefig(results_dir / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")

    data_filename = f"es_navix_h{args.hidden}_g{args.gen}_p{args.pop}_data.txt"
    with open(results_dir / data_filename, 'w') as f:
        f.write("# ES Navix Results\n")
        f.write(f"# Hidden: {args.hidden}, Gens: {args.gen}, Pop: {args.pop}\n")
        f.write(f"# Best: {best_fitness:.4f}\n")
        f.write("# Gen\tAvg\tBest\n")
        for i in range(len(generations)):
            f.write(f"{generations[i]}\t{fitnesses[i]:.6f}\t{best_fitnesses[i]:.6f}\n")
    print(f"Data saved: {data_filename}")
    plt.show()


if __name__ == '__main__':
    main()