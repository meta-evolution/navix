#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import navix as nx
from navix.observations import RADIUS


def save_image(obs, title, filename):
    """保存观察结果为图像"""
    plt.figure(figsize=(8, 6))
    if len(obs.shape) == 3:
        plt.imshow(obs)
    else:
        plt.imshow(obs, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def test_categorical_first_person():
    """测试第一人称分类观察"""
    # 创建Navix-Dynamic-Obstacles-16x16-v0环境
    env = nx.make('Navix-Dynamic-Obstacles-16x16-v0')
    
    # 重置环境获取初始状态
    key = jax.random.PRNGKey(0)
    timestep = env.reset(key)
    state = timestep.state
    
    # 获取观察数据
    categorical_obs = nx.observations.categorical_first_person(state)
    global_rgb = nx.observations.rgb(state)
    
    # 保存图像到output文件夹
    save_image(global_rgb, "Global RGB", "tests/output/navix_global_rgb.png")
    save_image(categorical_obs, "First Person Categorical", "tests/output/navix_categorical_obs.png")
    
    # 验证
    expected_shape = (RADIUS + 1, RADIUS * 2 + 1)
    assert categorical_obs.shape == expected_shape
    
    return categorical_obs, global_rgb


if __name__ == "__main__":
    categorical_obs, global_rgb = test_categorical_first_person()
    print(f"Test completed. Shapes: {categorical_obs.shape}, {global_rgb.shape}")