#!/usr/bin/env python3
"""NAVIX GPU Performance Benchmark with Internal JIT Optimization"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import navix as nx


def benchmark_env(env_name, episodes=1000, steps=500):
    """Benchmark environment with internal JIT optimization."""
    env = nx.make(env_name, observation_fn=nx.observations.categorical_first_person)
    
    # JIT-compiled functions
    @jax.jit
    def step_batch(timesteps, keys):
        actions = jax.vmap(lambda k: jax.random.randint(k, (), 0, env.action_space.n))(keys)
        return jax.vmap(env.step)(timesteps, actions)
    
    @jax.jit
    def reset_batch(keys):
        return jax.vmap(env.reset)(keys)
    
    # Warmup
    warmup_keys = jax.random.split(jax.random.PRNGKey(0), 10)
    _ = step_batch(reset_batch(warmup_keys), warmup_keys)
    
    # 收集观测数据的列表
    collected_observations = []
    
    # Benchmark
    start_time = time.time()
    keys = jax.random.split(jax.random.PRNGKey(42), episodes)
    timesteps = reset_batch(keys)
    
    # 收集初始观测 (已通过环境初始化设置为局部观测)
    collected_observations.append(timesteps.observation)
    
    for step in range(steps):
        step_keys = jax.random.split(jax.random.PRNGKey(step), episodes)
        timesteps = step_batch(timesteps, step_keys)
        # 收集每一步的观测 (已通过环境初始化设置为局部观测)
        collected_observations.append(timesteps.observation)
    
    duration = time.time() - start_time
    steps_per_sec = (episodes * steps) / duration
    episodes_per_sec = episodes / duration
    
    # 分析并打印观测tensor的整体形状
    if collected_observations:
        # 将所有观测堆叠成一个大的tensor
        stacked_observations = jnp.stack(collected_observations, axis=0)
        print(f"{env_name:20s}: {steps_per_sec:8,.0f} steps/sec, {episodes_per_sec:6,.0f} eps/sec")
        print(f"{'':20s}  观测tensor整体形状: {stacked_observations.shape} (时间步数×批次大小×观测维度)")
    else:
        print(f"{env_name:20s}: {steps_per_sec:8,.0f} steps/sec, {episodes_per_sec:6,.0f} eps/sec")
    
    return steps_per_sec, episodes_per_sec


def test_environments():
    """Test multiple environments."""
    envs = ['Navix-Empty-5x5-v0', 'Navix-Empty-8x8-v0', 
            'Navix-DoorKey-5x5-v0', 'Navix-DoorKey-8x8-v0', 'Navix-Dynamic-Obstacles-16x16-v0']
    
    print("\nEnvironment Performance:")
    print("-" * 50)
    
    results = {}
    for env_name in envs:
        try:
            results[env_name] = benchmark_env(env_name, 1000, 500)
        except Exception as e:
            print(f"{env_name:20s}: FAILED - {e}")
            results[env_name] = None
    
    return results


def test_batch_scaling():
    """Test batch size scaling."""
    print("\nBatch Scaling (Navix-Dynamic-Obstacles-16x16-v0):")
    print("-" * 40)
    
    for batch_size in [100, 1000, 5000, 10000]:
        try:
            _, eps_per_sec = benchmark_env('Navix-Dynamic-Obstacles-16x16-v0', batch_size, 20)
            print(f"Batch {batch_size:5d}: {eps_per_sec:8,.0f} eps/sec")
        except Exception as e:
            print(f"Batch {batch_size:5d}: FAILED - {e}")


def main():
    """Main benchmark runner."""
    print(f"JAX {jax.__version__} on {jax.default_backend()} ({len(jax.devices('gpu'))} GPUs)")
    
    test_environments()
    test_batch_scaling()
    
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()