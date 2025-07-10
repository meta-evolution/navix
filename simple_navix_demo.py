#!/usr/bin/env python3
"""
NAVIX 最简单的可视化 demo
基于 JAX 的 MiniGrid 环境，运行在 CPU 模式下
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np

def render_observation(obs, title):
    """渲染观察结果"""
    plt.figure(figsize=(6, 6))
    plt.imshow(obs)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 60)
    print("NAVIX 简单可视化 Demo")
    print("=" * 60)
    
    # 检查 JAX 版本和设备
    print(f"JAX 版本: {jax.__version__}")
    print(f"使用设备: {jax.default_backend()}")
    print(f"可用设备: {jax.devices()}")
    
    # 创建环境 - 使用 Empty 环境，这是最简单的
    print("\n创建 NAVIX 环境...")
    env = nx.make('Navix-Empty-5x5-v0', observation_fn=nx.observations.rgb)
    
    print(f"环境名称: Navix-Empty-5x5-v0")
    print(f"动作空间: {env.action_space.n} 个动作")
    print(f"观察空间: {env.observation_space.shape}")
    
    # 初始化环境
    key = jax.random.PRNGKey(42)
    timestep = env.reset(key)
    
    print(f"\n初始观察形状: {timestep.observation.shape}")
    render_observation(timestep.observation, "初始环境状态")
    
    # 生成一些随机动作
    num_steps = 10
    actions = jax.random.randint(key, (num_steps,), 0, env.action_space.n)
    
    print(f"\n执行 {num_steps} 步随机动作...")
    
    # 逐步执行动作并可视化关键步骤
    for step, action in enumerate(actions):
        timestep = env.step(timestep, action)
        
        # 每隔几步显示一次状态
        if step % 3 == 0 or step == num_steps - 1:
            print(f"步骤 {step + 1}: 动作={action}, 奖励={timestep.reward}")
            render_observation(timestep.observation, f"步骤 {step + 1} 后的环境状态")
    
    print("\n使用 JAX JIT 编译优化...")
    
    # 演示 JIT 编译的性能优化
    @jax.jit
    def step_jit(timestep, action):
        return env.step(timestep, action)
    
    # 重置环境
    timestep = env.reset(key)
    
    # 使用 JIT 编译的步骤函数
    print("执行 JIT 编译的步骤...")
    for i in range(3):
        action = jax.random.randint(key, (), 0, env.action_space.n)
        timestep = step_jit(timestep, action)
        print(f"JIT 步骤 {i+1}: 动作={action}, 奖励={timestep.reward}")
    
    render_observation(timestep.observation, "JIT 编译后的最终状态")
    
    # 演示批处理环境
    print("\n演示批处理环境...")
    
    def run_episode(seed):
        key = jax.random.PRNGKey(seed)
        timestep = env.reset(key)
        total_reward = 0
        
        for _ in range(5):
            action = jax.random.randint(key, (), 0, env.action_space.n)
            timestep = env.step(timestep, action)
            total_reward += timestep.reward
        
        return total_reward
    
    # 编译批处理函数
    run_batch = jax.jit(jax.vmap(run_episode))
    
    # 运行多个并行环境
    seeds = jnp.arange(8)
    rewards = run_batch(seeds)
    
    print(f"8 个并行环境的总奖励: {rewards}")
    print(f"平均奖励: {jnp.mean(rewards):.2f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成！")
    print("NAVIX 成功运行在 CPU 模式下，展示了:")
    print("1. 基本环境创建和交互")
    print("2. 可视化观察结果")
    print("3. JAX JIT 编译优化")
    print("4. 批处理并行环境")
    print("=" * 60)

if __name__ == "__main__":
    main()
