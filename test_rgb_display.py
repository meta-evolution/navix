#!/usr/bin/env python3
"""
简单的 RGB 显示测试
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np

def test_rgb_display():
    """测试 RGB 图像显示"""
    print("测试 NAVIX RGB 图像显示...")
    
    # 创建不同类型的环境来测试
    env_names = [
        "Navix-Empty-5x5-v0",
        "Navix-DoorKey-5x5-v0", 
        "Navix-FourRooms-v0"
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, env_name in enumerate(env_names):
        try:
            # 创建带 RGB 观察的环境
            env = nx.make(env_name, observation_fn=nx.observations.rgb)
            
            # 重置环境
            key = jax.random.PRNGKey(42)
            timestep = env.reset(key)
            obs = timestep.observation
            
            # 显示图像
            axes[i].imshow(obs)
            axes[i].set_title(f"{env_name}\n形状: {obs.shape}")
            axes[i].axis('off')
            
            # 打印图像统计信息
            print(f"\n{env_name}:")
            print(f"  形状: {obs.shape}")
            print(f"  类型: {obs.dtype}")
            print(f"  范围: {obs.min()} - {obs.max()}")
            print(f"  非零像素: {np.count_nonzero(obs)}/{obs.size}")
            
        except Exception as e:
            print(f"❌ {env_name} 失败: {e}")
            axes[i].text(0.5, 0.5, f"错误:\n{env_name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{env_name} - 错误")
    
    plt.tight_layout()
    plt.savefig('navix_rgb_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ RGB 图像测试完成！")
    print("图像已保存为 navix_rgb_test.png")

if __name__ == "__main__":
    test_rgb_display()
