#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NES‑trained 3‑layer MLP on CIFAR-10 (JAX, clean‑code version)
---------------------------------------------------------
核心流程：
1. 载入 & 预处理 CIFAR-10
2. 定义三层 MLP 与前向传播
3. NES：
   • 对中心参数采样对称噪声 ϵ/‑ϵ
   • 评估全部个体得到 fitness（这里用 −CE loss）
   • Centered‑Rank 转换 → 权重 w
   • 根据 ∑ w·ϵ 更新中心参数
4. 每 N 代在验证集上评估

基于 cifar100_nes_mlp_jax_opt.py 改编，主要变化：
- 数据源：CIFAR-100 → CIFAR-10
- 输出类别：100 → 10
- 默认参数：种群数量2000、网络尺寸256、200代
"""
from __future__ import annotations
import os, time, argparse, functools, pickle, sys
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
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

# 执行GPU设置
setup_gpu_from_args()

# ──────────────────────────────────────────────────────────────────────────────
# 0. 环境变量
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

# 导入JAX（此时CUDA_VISIBLE_DEVICES已设置）
import jax, jax.numpy as jnp
import optax
from tqdm import tqdm
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据加载 CIFAR-10
def unpickle(file):
    """解压 CIFAR-10 数据文件"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10(data_dir="cifar-10-batches-py"):
    """加载完整的 CIFAR-10 数据集"""
    # 训练数据 - 5个批次
    x_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        x_train_list.append(batch[b'data'])
        y_train_list.append(batch[b'labels'])
    
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # 测试数据
    test_batch = unpickle(f"{data_dir}/test_batch")
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # 数据预处理：reshape 和归一化
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_train = x_train.reshape(-1, 3072).astype('float32') / 255.0
    
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3072).astype('float32') / 255.0
    
    # 转换为 JAX 数组
    x_train = jnp.asarray(x_train)
    y_train = jnp.asarray(y_train)
    x_test = jnp.asarray(x_test)
    y_test = jnp.asarray(y_test)
    
    print(f"Training set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set: {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel value range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]")
    
    return x_train, y_train, x_test, y_test

# ──────────────────────────────────────────────────────────────────────────────
# 2. 模型定义
def glorot(k, shape):
    fan_in, fan_out = shape
    lim = jnp.sqrt(6 / (fan_in + fan_out))
    return jax.random.uniform(k, (fan_in, fan_out), minval=-lim, maxval=lim)

def init_params(rng, in_dim, hid, out_dim):
    k1, k2, k3 = jax.random.split(rng, 3)
    return dict(
        W1=glorot(k1, (in_dim, hid)), b1=jnp.zeros(hid),
        W2=glorot(k2, (hid, hid)),   b2=jnp.zeros(hid),
        W3=glorot(k3, (hid, out_dim)), b3=jnp.zeros(out_dim),
    )

def forward(params, x):
    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])
    return h2 @ params["W3"] + params["b3"]

# batched / jit‑compiled版本
batched_forward = jax.jit(jax.vmap(forward, in_axes=(None, 0)))

# ──────────────────────────────────────────────────────────────────────────────
# 3. NES 实用函数
@functools.partial(jax.jit, static_argnums=(2,))
def sample_noise(rng, params, pop_size: int, sigma: float):
    """对称噪声采样：返回 same treedef 的 eps+, eps- 拼接结果"""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    rngs = jax.random.split(rng, len(leaves))
    half = pop_size // 2

    def _one(e, k):
        n = jax.random.normal(k, (half, *e.shape)) * sigma
        return jnp.concatenate([n, -n], 0)
    noisy = [ _one(e, k) for e, k in zip(leaves, rngs) ]
    return jax.tree_util.tree_unflatten(treedef, noisy)

def add_noise(params, noise_tree, idx):
    """取第 idx 个噪声向量并加到中心参数上"""
    return jax.tree_util.tree_map(lambda p, n: p + n[idx], params, noise_tree)

def centered_rank(x):
    """Paper: https://arxiv.org/pdf/1703.03864.pdf"""
    x = jnp.ravel(x)
    ranks = jnp.argsort(jnp.argsort(x))
    return ranks / (x.size - 1) - 0.5

def logit_cross_entropy(logits, labels, num_classes=10):
    """交叉熵损失，适配 CIFAR-10 的 10 个类别"""
    y_onehot = jax.nn.one_hot(labels, num_classes)
    return -jnp.mean(jnp.sum(y_onehot * jax.nn.log_softmax(logits), axis=-1))

# ──────────────────────────────────────────────────────────────────────────────
# 4. 主训练循环
@dataclass
class ESConfig:
    generations: int = 200       # 指定的200代
    pop_size: int = 2000        # 指定的种群数量2000
    sigma: float = 0.05
    sigma_min: float = 0.01
    sigma_decay: float = 0.999
    lr: float = 0.05
    batch_train: int = 2048     # 每代用多少训练样本估计 fitness
    batch_eval: int = 4096      # 验证评估批

def main():
    # ── 命令行参数 ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='CIFAR-10 NES MLP Classifier')
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size (default: 256)")
    parser.add_argument("--gen", type=int, default=200, help="Number of generations (default: 200)")
    parser.add_argument("--pop", type=int, default=2000, help="Population size (default: 2000)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="Initial noise std")
    parser.add_argument("--gpu", type=int, default=None, help="Specify GPU device ID (e.g., 0, 1, 2, 3)")
    args = parser.parse_args()

    # ── 显示设备信息 ──────────────────────────────────────────────────────────
    devices = jax.devices()
    print(f"JAX available devices: {devices}")

    cfg = ESConfig(generations=args.gen, pop_size=args.pop, lr=args.lr, sigma=args.sigma)
    
    # ── 数据加载 ──────────────────────────────────────────────────────────────
    print("Loading CIFAR-10 dataset...")
    x_train, y_train, x_test, y_test = load_cifar10()
    
    # ── 模型初始化 ────────────────────────────────────────────────────────────
    print(f"\nInitializing model: input={x_train.shape[1]}, hidden={args.hidden}, output=10")
    rng = jax.random.PRNGKey(42)  # 固定随机种子以便复现
    params = init_params(rng, x_train.shape[1], args.hidden, 10)

    # ★ 使用简单 SGD 优化器（NES 标准做法）
    print(f"Using simple SGD optimizer with NES, learning rate: {cfg.lr}")
    
    # 计算参数总数
    total_params = sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    print(f"\nStarting NES training: generations={cfg.generations}, population={cfg.pop_size}, learning rate={cfg.lr}")
    print("=" * 70)

    # ── 主训练循环 ────────────────────────────────────────────────────────────
    best_acc = 0.0
    for g in range(1, cfg.generations + 1):
        # 1) 采样对称噪声
        rng, sub = jax.random.split(rng)
        noise_tree = sample_noise(sub, params, cfg.pop_size, cfg.sigma)
        
        # 2) 随机抽 batch 评估 fitness（负交叉熵）
        rng, idx_key = jax.random.split(rng)
        idx = jax.random.choice(idx_key, x_train.shape[0], (cfg.batch_train,), replace=False)

        def eval_individual(i):
            p_i = add_noise(params, noise_tree, i)
            logits = batched_forward(p_i, x_train[idx])
            return -logit_cross_entropy(logits, y_train[idx], 10)  # 越大越好
        
        fitness = jax.vmap(eval_individual)(jnp.arange(cfg.pop_size))

        # 3) Fit‑shaping & 求梯度
        w = centered_rank(fitness)
        w_half = w[:cfg.pop_size // 2] - w[cfg.pop_size // 2:]  # 利用对称性简化
        
        
        def tree_weighted_sum(noise_branch):
            pos, neg = jnp.split(noise_branch, 2, 0)
            # NES梯度公式: (1/σ) * Σ[(w_i^+ - w_i^-) * ε_i] 
            # 对称采样: pos是ε, neg是-ε, 所以用pos就行
            return jnp.tensordot(w_half, pos, axes=1) / (cfg.sigma * (cfg.pop_size // 2))
        
        grad = jax.tree_util.tree_map(tree_weighted_sum, noise_tree)

        # 4) 更新中心参数 (NES 使用简单SGD)
        params = jax.tree_util.tree_map(lambda p, g: p + cfg.lr * g, params, grad)

        # 5) 学习率 / σ 衰减
        cfg.sigma = max(cfg.sigma * cfg.sigma_decay, cfg.sigma_min)

        # 6) 定期评估和报告
        if g % 10 == 0 or g == 1:
            # 在测试集上评估
            acc = evaluate(params, x_test, y_test, cfg.batch_eval)
            if acc > best_acc:
                best_acc = acc
            
            # 也在训练集样本上评估 fitness
            train_fitness = float(jnp.mean(fitness))
            
            print(f"[Gen {g:4d}]   σ={cfg.sigma:.4f}   fitness={train_fitness:.4f}   "
                  f"test‑acc={acc*100:5.2f}%   best={best_acc*100:.2f}%")

    print("=" * 70)
    print(f"Training completed! Best test accuracy: {best_acc*100:.2f}%")

# ── 工具：评估准确率 ───────────────────────────────────────────────────────────
def evaluate(params, x, y, batch):
    """使用标准循环避免JIT中的动态形状问题"""
    correct = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        yb = y[i:i + batch]
        logits = batched_forward(params, xb)
        correct += (jnp.argmax(logits, -1) == yb).sum()
    return correct / x.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()