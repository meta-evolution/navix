#!/usr/bin/env python3
"""
测试程序：逐步验证ES_RNN的计算正确性
使用最简单的配置：每层只有一个神经元，整数参数和输入
"""

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
import jax.numpy as jnp
from neurogenesistape.modules.es.nn import ES_RNN
from flax import nnx
import numpy as np

def manual_calculation_verification():
    """
    手动计算验证函数
    """
    print("=== 手动计算验证 ===")
    
    # 设置简单的整数参数
    i2h_weight = 2.0
    i2h_bias = 1.0
    h2h_weight = 0.5
    h2h_bias = 0.0
    h2o_weight = 1.0
    h2o_bias = 0.0
    
    # 简单的输入序列
    input_seq = [1.0, 2.0, 3.0]
    
    print(f"输入序列: {input_seq}")
    print(f"i2h权重: {i2h_weight}, 偏置: {i2h_bias}")
    print(f"h2h权重: {h2h_weight}, 偏置: {h2h_bias}")
    print(f"h2o权重: {h2o_weight}, 偏置: {h2o_bias}")
    print()
    
    h = 0.0  # 初始隐藏状态
    
    for t, x in enumerate(input_seq):
        print(f"时间步 {t}:")
        print(f"  输入 x: {x}")
        print(f"  当前隐藏状态 h: {h}")
        
        # 计算 i2h
        i2h_out = x * i2h_weight + i2h_bias
        print(f"  i2h输出: {x} * {i2h_weight} + {i2h_bias} = {i2h_out}")
        
        # 计算 h2h
        h2h_out = h * h2h_weight + h2h_bias
        print(f"  h2h输出: {h} * {h2h_weight} + {h2h_bias} = {h2h_out}")
        
        # 组合并应用tanh
        combined = i2h_out + h2h_out
        h_new = np.tanh(combined)
        print(f"  组合: {i2h_out} + {h2h_out} = {combined}")
        print(f"  新隐藏状态: tanh({combined}) = {h_new}")
        
        # 计算输出
        output = h_new * h2o_weight + h2o_bias
        print(f"  输出: {h_new} * {h2o_weight} + {h2o_bias} = {output}")
        
        h = h_new
        print()

def test_es_rnn():
    """
    测试ES_RNN的计算过程
    """
    print("=== 测试ES_RNN ===")
    
    # 创建模型
    key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(key)
    model = ES_RNN(input_size=1, hidden_size=1, output_size=1, rngs=rngs)
    
    # 手动设置简单的整数参数
    # 注意：ES_Tape的参数存储在.grad_variable.value中
    model.i2h.kernel.grad_variable.value = jnp.array([[2.0]])
    model.i2h.bias.grad_variable.value = jnp.array([1.0])
    model.h2h.kernel.grad_variable.value = jnp.array([[0.5]])
    model.h2h.bias.grad_variable.value = jnp.array([0.0])
    model.h2o.kernel.grad_variable.value = jnp.array([[1.0]])
    model.h2o.bias.grad_variable.value = jnp.array([0.0])
    
    print("模型参数设置完成:")
    print(f"i2h权重: {model.i2h.kernel.grad_variable.value}, 偏置: {model.i2h.bias.grad_variable.value}")
    print(f"h2h权重: {model.h2h.kernel.grad_variable.value}, 偏置: {model.h2h.bias.grad_variable.value}")
    print(f"h2o权重: {model.h2o.kernel.grad_variable.value}, 偏置: {model.h2o.bias.grad_variable.value}")
    print()
    
    # 创建简单的输入
    inputs = jnp.array([[[1.0], [2.0], [3.0]]])  # [batch_size=1, seq_len=3, input_size=1]
    
    print(f"输入形状: {inputs.shape}")
    print(f"输入内容: {inputs}")
    print()
    
    # 重置隐藏状态
    model.reset_hidden(batch_size=1)
    print(f"初始隐藏状态: {model.hidden_state.value}")
    print()
    
    # 逐步前向传播
    print("逐步前向传播:")
    outputs = []
    
    for t in range(inputs.shape[1]):
        x_t = inputs[:, t, :]  # [batch_size, input_size]
        
        print(f"时间步 {t}:")
        print(f"  输入 x_t: {x_t}")
        print(f"  当前隐藏状态: {model.hidden_state.value}")
        
        # 计算 i2h 输出
        i2h_out = model.i2h(x_t)
        print(f"  i2h输出: {i2h_out}")
        
        # 计算 h2h 输出
        h2h_out = model.h2h(model.hidden_state.value)
        print(f"  h2h输出: {h2h_out}")
        
        # 组合并应用tanh
        combined = i2h_out + h2h_out
        new_hidden = jnp.tanh(combined)
        print(f"  组合输入: {combined}")
        print(f"  新隐藏状态: {new_hidden}")
        
        # 计算输出
        output = model.h2o(new_hidden)
        print(f"  输出: {output}")
        
        # 更新隐藏状态
        model.hidden_state.value = new_hidden
        outputs.append(output)
        print()
    
    # 使用模型的forward_sequence方法验证
    print("使用forward_sequence方法验证:")
    model.reset_hidden(batch_size=1)
    sequence_outputs = model.forward_sequence(inputs)
    print(f"序列输出形状: {sequence_outputs.shape}")
    print(f"序列输出: {sequence_outputs}")
    
    return jnp.stack(outputs, axis=1), sequence_outputs

def compare_with_manual():
    """
    比较模型输出与手动计算结果
    """
    print("\n" + "="*50)
    print("比较模型输出与手动计算")
    print("="*50)
    
    # 先进行手动计算
    manual_calculation_verification()
    
    print("\n" + "-"*30)
    
    # 再运行模型
    step_outputs, sequence_outputs = test_es_rnn()
    
    print("\n=== 结果比较 ===")
    print(f"逐步计算输出: {step_outputs}")
    print(f"序列方法输出: {sequence_outputs}")
    print(f"两种方法是否一致: {jnp.allclose(step_outputs, sequence_outputs)}")
    
    print("\n验证完成！请比较上述手动计算结果与模型输出结果。")

if __name__ == "__main__":
    print("ES_RNN 逐步计算验证程序")
    print("="*50)
    
    compare_with_manual()