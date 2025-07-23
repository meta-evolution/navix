#!/usr/bin/env python3
"""
验证 ES 算法中 state.update(grads) 是否成功更新模型参数
"""

import jax
import jax.numpy as jnp
import numpy as np
from neurogenesistape.modules.es.nn import ES_RNN
from neurogenesistape.modules.es.optimizer import ES_Optimizer
from neurogenesistape.modules.variables import Grad_variable
from flax import nnx
import optax

def test_parameter_update():
    """测试参数更新是否生效"""
    print("=== 测试 ES 参数更新 ===")
    
    # 创建小规模 RNN
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    input_size = 10
    hidden_size = 8
    output_size = 4
    
    # 初始化 RNN
    agent = ES_RNN(input_size, hidden_size, output_size, rngs)
    
    # 初始化 ES 优化器
    tx = optax.adam(learning_rate=0.01)
    state = ES_Optimizer(agent, tx)
    
    print(f"模型已初始化")
    print(f"优化器类型: {type(state)}")
    
    # 获取初始参数
    initial_params = nnx.state(agent)
    print(f"\n初始参数结构: {jax.tree.map(lambda x: x.shape, initial_params)}")
    
    # 将参数展平以便分析
    flat_params = jax.tree.leaves(initial_params)
    flat_initial = jnp.concatenate([p.flatten() for p in flat_params])
    print(f"展平后参数数量: {len(flat_initial)}")
    print(f"初始参数前5个值: {flat_initial[:5]}")
    print(f"初始参数均值: {jnp.mean(flat_initial):.6f}")
    print(f"初始参数标准差: {jnp.std(flat_initial):.6f}")
    
    # 创建模拟梯度（与参数结构相同）
    # 使用简单的梯度：让所有参数朝零方向移动
    mock_grads = jax.tree.map(lambda x: -0.1 * x, initial_params)
    
    # 将梯度转换为 Grad_variable 格式
    grad_state = jax.tree.map(lambda x: Grad_variable(x), mock_grads)
    
    flat_grads = jax.tree.leaves(mock_grads)
    flat_grad_values = jnp.concatenate([g.flatten() for g in flat_grads])
    print(f"\n梯度前5个值: {flat_grad_values[:5]}")
    print(f"梯度均值: {jnp.mean(flat_grad_values):.6f}")
    print(f"梯度标准差: {jnp.std(flat_grad_values):.6f}")
    print(f"梯度范数: {jnp.linalg.norm(flat_grad_values):.6f}")
    
    # 更新参数前的状态
    pre_update_params = nnx.state(agent)
    pre_update_step = state.step.value
    
    # 执行参数更新
    print("\n执行 state.update(grads)...")
    state.update(grad_state)
    
    # 检查更新后的参数
    post_update_params = nnx.state(agent)
    post_update_step = state.step.value
    
    # 将更新后的参数展平以便比较
    flat_post_params = jax.tree.leaves(post_update_params)
    flat_post = jnp.concatenate([p.flatten() for p in flat_post_params])
    
    print(f"\n更新后参数前5个值: {flat_post[:5]}")
    print(f"更新后参数均值: {jnp.mean(flat_post):.6f}")
    print(f"更新后参数标准差: {jnp.std(flat_post):.6f}")
    
    # 计算参数变化
    param_diff = flat_post - flat_initial
    print(f"\n参数变化前5个值: {param_diff[:5]}")
    print(f"参数变化均值: {jnp.mean(param_diff):.6f}")
    print(f"参数变化标准差: {jnp.std(param_diff):.6f}")
    print(f"参数变化范数: {jnp.linalg.norm(param_diff):.6f}")
    
    # 检查优化器步数是否更新了
    print(f"\n更新前步数: {pre_update_step}")
    print(f"更新后步数: {post_update_step}")
    print(f"步数变化: {post_update_step - pre_update_step}")
    
    # 验证参数确实发生了变化
    params_changed = not jnp.allclose(flat_initial, flat_post, atol=1e-10)
    step_changed = post_update_step != pre_update_step
    
    print(f"\n=== 验证结果 ===")
    print(f"参数是否发生变化: {params_changed}")
    print(f"优化器步数是否更新: {step_changed}")
    
    if params_changed:
        print("✅ 参数更新成功！")
    else:
        print("❌ 参数没有更新！")
        
    if step_changed:
        print("✅ 优化器步数更新成功！")
    else:
        print("❌ 优化器步数没有更新！")
    
    # 额外验证：检查梯度方向是否正确
    # 我们的梯度是 -0.1 * params，所以参数变化应该与梯度方向一致
    expected_direction = flat_grad_values  # 期望的变化方向
    actual_direction = param_diff  # 实际的变化
    
    # 计算方向相关性
    if jnp.linalg.norm(expected_direction) > 0 and jnp.linalg.norm(actual_direction) > 0:
        correlation = jnp.dot(expected_direction, actual_direction) / (jnp.linalg.norm(expected_direction) * jnp.linalg.norm(actual_direction))
        print(f"\n梯度方向相关性: {correlation:.4f}")
        
        if correlation > 0.5:
            print("✅ 梯度方向正确！")
        else:
            print("⚠️  梯度方向可能不正确")
    else:
        print("\n⚠️  无法计算梯度方向相关性（零向量）")
    
    return params_changed, step_changed

if __name__ == "__main__":
    test_parameter_update()