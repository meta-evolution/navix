#!/usr/bin/env python3

import sys
sys.path.append('/root/workspace/navix/tests')

import jax
import jax.numpy as jnp
from neurogenesistape.modules.es.nn import ES_RNN
from neurogenesistape.modules.es.optimizer import ES_Optimizer
from neurogenesistape.modules.evolution import sampling, calculate_gradients, centered_rank
from neurogenesistape.modules.variables import Populated_Variable, Grad_variable
from flax import nnx
import optax

others = (nnx.RngCount, nnx.RngKey)
state_axes = nnx.StateAxes({nnx.Param: None, Populated_Variable: 0, Grad_variable: None, nnx.Variable: None, others: None})

@nnx.vmap(in_axes=(state_axes, None))
def populated_noise_fwd(model: nnx.Module, input):
    return model(input)

def generate_batch_data(key, batch_size=2, seq_len=3):
    patterns = jnp.array([
        [[1], [2], [3]],
        [[3], [2], [1]]
    ])
    
    sequences = jnp.tile(patterns, (batch_size // 2 + 1, 1, 1))[:batch_size]
    targets = jnp.array([0, 1] * (batch_size // 2 + 1))[:batch_size]
    
    return sequences.astype(jnp.float32), targets

def compute_fitness(outputs, targets):
    def _fitness(_outputs, _targets):
        logits = _outputs[:, -1, :]
        targets_onehot = jax.nn.one_hot(_targets, num_classes=logits.shape[-1])
        mse_loss = jnp.mean((logits - targets_onehot) ** 2)
        fitness = -mse_loss
        return fitness
    
    fitness_values = jax.vmap(_fitness, in_axes=(0, None))(outputs, targets)
    return fitness_values

def train_step(state, batch_x, batch_y, generation=0):
    state.model.reset_hidden(batch_x.shape[0])
    sampling(state.model)
    outputs = populated_noise_fwd(state.model, batch_x)
    
    fitness = compute_fitness(outputs, batch_y)
    avg_fitness = jnp.mean(fitness)
    fitness_ranked = centered_rank(fitness)
    grads = calculate_gradients(state.model, fitness_ranked)
    state.update(grads)
    
    return avg_fitness, grads, outputs, fitness

def main():
    config = {
        'input_size': 1, 'hidden_size': 20, 'output_size': 2,
        'popsize': 200, 'generations': 50, 'learning_rate': 0.05, 'sigma': 0.05
    }
    
    key = jax.random.PRNGKey(21)
    rngs = nnx.Rngs(key)
    model = ES_RNN(config['input_size'], config['hidden_size'], config['output_size'], rngs)
    
    model.set_attributes(popsize=config['popsize'], noise_sigma=config['sigma'])
    model.deterministic = False
    
    tx = optax.chain(optax.scale(-1), optax.sgd(config['learning_rate']))
    state = ES_Optimizer(model, tx, wrt=nnx.Param)
    
    key, data_key = jax.random.split(key)
    train_data, train_targets = generate_batch_data(data_key, batch_size=2)
    
    print("训练数据 (train_data):")
    print(train_data)
    print("\n目标数据 (train_targets):")
    print(train_targets)
    print("\n开始ES训练...\n")
    
    for gen in range(1, config['generations'] + 1):
        if gen == 1:
            old_params = state.model.i2h.kernel.grad_variable.value.copy()
        
        avg_fitness, grads, outputs, fitness = train_step(state, train_data, train_targets, gen)
        
        best_idx = jnp.argmax(fitness)
        best_output = outputs[best_idx]
        best_logits = best_output[:, -1, :]
        
        if gen <= 10000:
            new_params = state.model.i2h.kernel.grad_variable.value
            param_change = jnp.max(jnp.abs(new_params - old_params))
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}, param_change={param_change:.8f}, grad_norm={grad_norm:.8f}")
            print(f"        最佳子代输出: {best_logits}, 真实标签: {train_targets}, 适应度: {fitness[best_idx]:.6f}")
            old_params = new_params.copy()
        else:
            print(f"Gen {gen:2d}: avg={avg_fitness:8.6f}")
            print(f"        最佳子代输出: {best_logits}, 真实标签: {train_targets}, 适应度: {fitness[best_idx]:.6f}")

if __name__ == "__main__":
    main()