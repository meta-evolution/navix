#!/usr/bin/env python3
"""
NAVIX GPU性能测试 - 优化版本
在for循环内部使用JIT和VMAP优化
"""

import time
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np

def benchmark_environment_optimized(env_name, num_episodes=1000, max_steps=500):
    """在for循环内部使用JIT和VMAP的优化版本"""
    print(f"\n[内部JIT优化] 测试环境: {env_name}")
    print(f"测试参数: {num_episodes} episodes, {max_steps} steps each")
    
    # 创建环境
    env = nx.make(env_name, observation_fn=nx.observations.symbolic)
    
    # 定义单步操作函数
    def single_step(timestep, key):
        """单步环境交互"""
        action = jax.random.randint(key, (), 0, env.action_space.n)
        new_timestep = env.step(timestep, action)
        return new_timestep
    
    # 定义批量重置函数
    def batch_reset(keys):
        """批量重置环境"""
        return jax.vmap(env.reset)(keys)
    
    # JIT编译关键函数
    jit_single_step = jax.jit(single_step)
    jit_batch_step = jax.jit(jax.vmap(single_step))
    jit_batch_reset = jax.jit(batch_reset)
    
    print("预热JIT编译...")
    # 预热JIT编译
    test_keys = jax.random.split(jax.random.PRNGKey(0), 10)
    test_timesteps = jit_batch_reset(test_keys)
    _ = jit_batch_step(test_timesteps, test_keys)
    
    print("开始性能测试...")
    start_time = time.time()
    
    # 生成随机种子
    main_key = jax.random.PRNGKey(42)
    episode_keys = jax.random.split(main_key, num_episodes)
    
    # 批量重置所有环境
    timesteps = jit_batch_reset(episode_keys)
    total_rewards = jnp.zeros(num_episodes)
    
    # 在for循环内部使用JIT和VMAP
    for step in range(max_steps):
        # 为每个环境生成新的随机key
        step_keys = jax.random.split(jax.random.PRNGKey(step), num_episodes)
        
        # 批量执行一步，使用JIT编译的VMAP函数
        timesteps = jit_batch_step(timesteps, step_keys)
        
        # 累积奖励
        total_rewards += timesteps.reward
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 统计结果
    total_steps = num_episodes * max_steps
    steps_per_second = total_steps / duration
    episodes_per_second = num_episodes / duration
    
    print(f"执行时间: {duration:.3f} 秒")
    print(f"步数/秒: {steps_per_second:,.0f}")
    print(f"回合/秒: {episodes_per_second:,.0f}")
    print(f"平均奖励: {jnp.mean(total_rewards):.3f}")
    print(f"完成率: {jnp.mean(jax.vmap(lambda ts: ts.is_done())(timesteps)):.1%}")
    
    return steps_per_second, episodes_per_second

def test_different_environments():
    """测试不同环境的性能"""
    environments = [
        'Navix-Empty-5x5-v0',
        'Navix-Empty-8x8-v0', 
        'Navix-DoorKey-5x5-v0',
        'Navix-DoorKey-8x8-v0'
    ]
    
    results = {}
    
    for env_name in environments:
        try:
            steps_per_sec, episodes_per_sec = benchmark_environment_optimized(env_name, 1000, 500)
            results[env_name] = {
                'steps_per_sec': steps_per_sec,
                'episodes_per_sec': episodes_per_sec
            }
        except Exception as e:
            print(f"环境 {env_name} 测试失败: {e}")
            results[env_name] = None
    
    return results

def test_batch_scaling():
    """测试批处理规模对性能的影响"""
    print("\n" + "=" * 60)
    print("批处理规模性能测试")
    print("=" * 60)
    
    env_name = 'Navix-Empty-8x8-v0'
    max_steps = 20
    batch_sizes = [100, 1000, 5000, 10000]
    
    for batch_size in batch_sizes:
        try:
            steps_per_sec, episodes_per_sec = benchmark_environment_optimized(env_name, batch_size, max_steps)
            print(f"批大小 {batch_size:5d}: {episodes_per_sec:8,.0f} episodes/sec")
        except Exception as e:
            print(f"批大小 {batch_size} 测试失败: {e}")

def explain_optimization_approach():
    """解释优化方法"""
    print("\n" + "=" * 80)
    print("内部JIT优化方法说明")
    print("=" * 80)
    
    explanation = """
优化策略: 在for循环内部使用JIT和VMAP

1. 传统方法的问题:
   - 将整个episode函数用JIT包装，导致大计算图
   - for循环被完全展开，编译时间长
   - 内存占用高，不利于大规模并行

2. 内部JIT优化的优势:
   - 只对关键操作(单步交互)进行JIT编译
   - for循环在Python层面执行，避免展开
   - 每次循环内部使用VMAP实现批量并行
   - 编译图更小，编译时间更短
   - 内存使用更高效

3. 具体实现:
   - 将单步操作函数进行JIT编译
   - 使用VMAP实现批量处理
   - 在Python for循环中调用JIT函数
   - 保持JAX的并行化优势，同时避免大计算图

4. 性能优势:
   - 减少JIT编译开销
   - 保持GPU并行化效率
   - 更好的内存管理
   - 适合大规模批处理
    """
    
    print(explanation)

def main():
    print("=" * 80)
    print("NAVIX GPU 性能优化测试 - 内部JIT版本")
    print("=" * 80)
    
    # 显示系统信息
    print(f"JAX 版本: {jax.__version__}")
    print(f"默认后端: {jax.default_backend()}")
    print(f"GPU 设备: {len(jax.devices('gpu'))} 个")
    print(f"设备列表: {jax.devices()}")
    
    # 解释优化方法
    explain_optimization_approach()
    
    # 测试不同环境
    print("\n" + "=" * 60)
    print("不同环境性能测试")
    print("=" * 60)
    
    results = test_different_environments()
    
    # 显示汇总结果
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    
    for env_name, result in results.items():
        if result:
            print(f"{env_name:25s}: {result['steps_per_sec']:8,.0f} steps/sec")
        else:
            print(f"{env_name:25s}: 测试失败")
    
    # 批处理规模测试
    test_batch_scaling()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("结论: 在for循环内部使用JIT和VMAP可以有效优化性能")
    print("=" * 80)

if __name__ == "__main__":
    main()