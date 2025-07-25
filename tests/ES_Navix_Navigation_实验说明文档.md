# ES Navix Navigation 实验说明文档

## 概述

本实验使用进化策略（Evolution Strategies, ES）算法训练智能体在 Navix 导航环境中学习路径规划和避障行为。实验基于 `neurogenesistape` (NGT) 框架实现，采用批量并行评估和可视化追踪功能。

## 实验环境

### 环境配置
- **环境名称**: `Navix-Dynamic-Obstacles-16x16-v0`
- **观察类型**: 符号观察 (`nx.observations.symbolic`)
- **转换函数**: 确定性转换 (`transitions.deterministic_transition`)
- **网格大小**: 16x16
- **动态障碍**: 启用

### 状态空间
- **观察维度**: 环境符号观察 + 智能体位置(x,y) + 智能体方向
- **位置归一化**: 位置坐标除以15.0进行归一化
- **方向归一化**: 方向值除以3.0进行归一化
- **总观察维度**: `obs_size + 2 + 1`

### 动作空间
- **动作类型**: 离散动作空间
- **动作数量**: 7个动作（前进、后退、左转、右转等）
- **实验设置**: 强制前进动作概率为90%，其他动作各占1.67% （暂时调试用）

## 算法实现

### 进化策略 (ES) 算法

#### 核心组件
1. **ES_MLP**: 多层感知机网络，使用ES参数
2. **ES_Optimizer**: ES优化器，基于optax优化器链
3. **ESConfig**: ES配置类，管理超参数

#### 算法流程
1. **种群初始化**: 使用对称噪声生成种群
2. **批量评估**: 并行评估所有种群成员的适应度
3. **适应度计算**: 使用负路径长度作为适应度（路径越短，适应度越高）
4. **梯度计算**: 使用中心排名变换和梯度估计
5. **参数更新**: 基于SGD优化器更新网络参数

### 网络架构
- **输入层**: 观察维度（环境观察 + 位置 + 方向）
- **隐藏层**: 2层，默认128个神经元
- **输出层**: 动作空间大小（7个动作）
- **激活函数**: 默认激活函数（ReLU等）

## 实验参数

### 默认超参数
```python
--hidden 128        # 隐藏层大小
--gen 100           # 训练代数
--pop 200           # 种群大小（必须为偶数）
--lr 0.01           # 学习率
--sigma 0.1         # 噪声标准差
--max_steps 500     # 每回合最大步数
--seed 42           # 随机种子
```

### 测试模式参数
```python
--test              # 启用测试模式
# 测试模式下：pop=20, gen=10, max_steps=50
```

### 可选参数
```python
--gpu <id>          # 指定GPU设备ID
--visualize         # 启用可视化追踪
--seed <value>      # 设置随机种子
```

## 实验特性

### 1. 批量并行评估
- **并行环境**: 同时评估整个种群的所有成员
- **JIT编译**: 使用JAX JIT加速计算
- **内存优化**: 批量处理减少内存开销

### 2. 可视化追踪
- **环境可视化**: 保存每步的RGB图像
- **路径追踪**: 记录智能体的移动轨迹
- **状态监控**: 实时显示位置、方向和动作信息
- **输出目录**: `results/visualization_gen_<generation>/`

### 3. 实验控制
- **强制动作概率**: 前进动作90%概率（用于测试）
- **确定性种子**: 基于代数的确定性随机种子
- **早期终止**: 所有环境完成时提前结束

### 4. 结果保存
- **训练曲线**: 平均适应度和最佳适应度曲线
- **数据文件**: 详细的训练数据（.txt格式）
- **图像文件**: 高分辨率结果图表（.png格式）

## 运行方式

### 基本训练
```bash
python es_navix_navigation.py
```

### 自定义参数训练
```bash
python es_navix_navigation.py --hidden 256 --gen 200 --pop 400 --lr 0.005
```

### 测试模式
```bash
python es_navix_navigation.py --test
```

### 启用可视化
```bash
python es_navix_navigation.py --visualize --gen 50
```

### GPU训练
```bash
python es_navix_navigation.py --gpu 0 --pop 500
```

### 小参数快速训练（适合调试和初步验证）
```bash
# 最小参数配置 - 快速测试
python es_navix_navigation.py --gen 1 --pop 5 --max_steps 100

# 轻量级训练 - 适合资源受限环境
python es_navix_navigation.py --gen 2 --pop 5 --lr 0.02 --sigma 0.05

# 小参数可视化训练
python es_navix_navigation.py --gen 2 --pop 5 --visualize --max_steps 200
```

## 输出结果

### 训练过程输出
```
[Gen    1] avg_fitness= -45.2500 best_fitness= -23.0000
[Gen    2] avg_fitness= -42.1800 best_fitness= -20.0000
...
```

### 结果文件
1. **图表文件**: `results/es_navix_h128_g100_p200_results.png`
2. **数据文件**: `results/es_navix_h128_g100_p200_data.txt`
3. **可视化文件**: `results/visualization_gen_<N>/step_<XXX>.png`

### 性能指标
- **适应度**: 负路径长度（越高越好）
- **路径长度**: 到达目标所需步数（越短越好）
- **成功率**: 在最大步数内到达目标的比例

## 技术细节

### 依赖库
- **JAX**: 高性能数值计算和自动微分
- **Flax NNX**: 神经网络框架
- **Optax**: 优化器库
- **Navix**: 导航环境框架
- **neurogenesistape (NGT)**: ES算法实现
- **Matplotlib**: 结果可视化

### 性能优化
- **JIT编译**: 关键函数使用@jax.jit装饰器
- **向量化操作**: 使用jax.vmap进行批量处理
- **内存管理**: 避免不必要的数据复制
- **进度显示**: 减少I/O开销的进度更新

### 实验设计考虑
- **可重现性**: 固定随机种子确保结果可重现
- **稳定性**: 对称噪声和偶数种群大小
- **可扩展性**: 支持不同网络架构和超参数
- **调试友好**: 详细的日志输出和可视化支持

## 注意事项

1. **种群大小**: 必须为偶数（ES算法要求对称噪声）
2. **GPU内存**: 大种群可能需要更多GPU内存
3. **可视化开销**: 启用可视化会显著增加运行时间
4. **文件权限**: 确保有写入results目录的权限
5. **依赖版本**: 确保所有依赖库版本兼容

## 扩展建议

1. **超参数调优**: 使用网格搜索或贝叶斯优化
2. **网络架构**: 尝试不同的隐藏层配置
3. **环境变体**: 测试不同的Navix环境
4. **算法改进**: 实现自适应噪声或其他ES变体
5. **多目标优化**: 同时优化路径长度和能耗等多个指标

---

**文档版本**: 1.0  
**最后更新**: 2024年  
**维护者**: ES Navix Navigation 项目组