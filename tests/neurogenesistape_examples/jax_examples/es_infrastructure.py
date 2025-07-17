"""
es_infrastructure.py
~~~~~~~~~~~~~~~~~~~~
Reusable utilities for large‑scale Evolution Strategies with JAX.

Author: <your name>, 2025‑06‑13
"""

from __future__ import annotations
from functools import partial
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax

__all__ = [
    "ESConfig",
    "reorg_obs",
    "get_noise_for_model",
    "model_forward",
    "model_forward_vmap",
    "init_fn",
    "init_fn_rnd",
    "params_add",
    "get_action_deterministic",
    "get_action_deterministic_vmap",
    "get_rnd_act",
    "centered_rank_transform",
    "get_fitness_multy_objective",
    "get_fitness_multy_objective_vmap",
    "get_fitness_final_perf",
    "get_fitness_final_perf_vmap",
    "build_adamw",
]

# ---------------------------------------------------------------------
# 1. Configuration dataclass
# ---------------------------------------------------------------------

@dataclass
class ESConfig:
    """Collection of hyper‑parameters used by Evolution Strategies."""
    pop_size: int = 12000
    sigma: float = 0.04
    sigma_low: float = 0.02
    sigma_anneal: float = 0.999
    lr: float = 0.01
    weight_decay: float = 0.02
    steps_per_episode: int = 100
    max_generations: int = 20_000
    seed: int | None = None
    # ---- model / env‑agnostic extras ----
    action_threshold: float = 0.2
    k1: float = 1.0       # weight: mean reward
    k2: float = 2.0       # weight: worst‑case reward
    k3: float = 0.0       # weight: skill improvement term


# ---------------------------------------------------------------------
# 2. Data‑layout helpers
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_landscapes",))
def reorg_obs(obs: jnp.ndarray, n_landscapes: int) -> jnp.ndarray:
    """
    Reshape a flat observation batch of shape (N_envs, F)
    into (N_envs // n_landscapes, n_landscapes, F).

    Parameters
    ----------
    obs : jnp.ndarray
        Concatenated observations.
    n_landscapes : int
        Number of distinct landscapes bundled in the batch.
    """
    return jnp.reshape(
        obs, (obs.shape[0] // n_landscapes, n_landscapes, obs.shape[1])
    )


# ---------------------------------------------------------------------
# 3. Gaussian noise sampler (mirrored sampling)
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnums=(2,))
def get_noise_for_model(
    key: jax.Array,
    params: jax.PyTree,
    pop_size: int,
    std: float = 0.02,
) -> jax.PyTree:
    """
    Generate **mirrored Gaussian** noise for every leaf in a parameter PyTree.

    The function first samples `pop_size // 2` i.i.d. normal tensors ε,
    then concatenates [+ε, –ε] along a new leading axis so that the final
    population size matches `pop_size`.

    Returns
    -------
    noise : PyTree
        Tree with leading dimension `pop_size`.
    """
    batch_shape = (pop_size // 2,)
    leaves = jax.tree_util.tree_leaves(params)
    treedef = jax.tree_util.tree_structure(params)

    # Split the master key into one sub‑key per leaf
    leaf_keys = jax.random.split(key, num=len(leaves))                   # :contentReference[oaicite:0]{index=0}
    noise_half = jax.tree_util.tree_map(                                 # :contentReference[oaicite:1]{index=1}
        lambda g, k: std
        * jax.random.normal(k, shape=(*batch_shape, *g.shape), dtype=g.dtype),
        params,
        jax.tree_util.tree_unflatten(treedef, leaf_keys),
    )
    # Build mirrored population: [+ε, –ε]
    return jax.tree_util.tree_map(lambda n: jnp.concatenate([n, -n], axis=0), noise_half)


# ---------------------------------------------------------------------
# 4. Model forward‑pass helpers (works with Flax, Haiku, custom apply)
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnums=(3,))
def model_forward(
    variables: jax.PyTree,
    state: jax.Array,
    x: jnp.ndarray,
    model,
):
    """Single forward pass wrapper; `model` is treated as a static argument."""
    return model.apply(variables, state, x)                              # :contentReference[oaicite:2]{index=2}


# Vectorised over population (axis 0)
model_forward_vmap = jax.vmap(
    model_forward, in_axes=(0, 0, 0, None), out_axes=0
)                                                                        # :contentReference[oaicite:3]{index=3}


# ---------------------------------------------------------------------
# 5. Parameter initialisation boilerplate
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnums=(2,))
def init_fn(seed: jax.Array, sample_obs: jnp.ndarray, model):
    """Initialise parameters + state for a recurrent agent."""
    return model.init(seed, model.initial_state(sample_obs.shape[0]), sample_obs)


@partial(jax.jit, static_argnums=(2,))
def init_fn_rnd(seed: jax.Array, sample_obs: jnp.ndarray, model):
    """Initialise parameters with randomised initial RNN state."""
    return model.init(
        seed, model.initial_state_rnd(sample_obs.shape[0]), sample_obs
    )


# ---------------------------------------------------------------------
# 6. Low‑level tensor ops
# ---------------------------------------------------------------------

@partial(jax.jit, donate_argnums=(0, 1))
def params_add(params_center: jax.PyTree, pop_noise: jax.PyTree) -> jax.PyTree:
    """Element‑wise add two parameter PyTrees (in‑place donation)."""     # :contentReference[oaicite:4]{index=4}
    return jax.tree_util.tree_map(lambda x, y: x + y, params_center, pop_noise)


# ---------------------------------------------------------------------
# 7. Action helpers
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1,))
def get_action_deterministic(
    logits: jnp.ndarray, action_threshold: float = 0.2
) -> jnp.ndarray:
    """
    Greedy action with confidence threshold.

    If max(logits) < threshold → return fallback action 4.
    """
    argmax = jnp.argmax(logits)
    return jnp.where(jnp.max(logits) >= action_threshold, argmax, 4)


get_action_deterministic_vmap = jax.vmap(get_action_deterministic, in_axes=(0, None))


@partial(jax.jit, static_argnums=(0,))
def get_rnd_act(n_envs: int) -> jnp.ndarray:
    """Uniform random actions in [0, 3] per env."""
    return jax.random.randint(
        jax.random.PRNGKey(0), shape=(n_envs,), minval=0, maxval=4
    )                                                                     # :contentReference[oaicite:5]{index=5}


# ---------------------------------------------------------------------
# 8. Fitness transformations
# ---------------------------------------------------------------------

@jax.jit
def centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """
    Centered‑rank transform (OpenES, Salimans et al. 2017).

    Maps raw scores to (‑0.5, 0.5] to stabilise gradient estimates.
    """
    flat = x.ravel()
    ranks = jnp.argsort(jnp.argsort(flat))
    return (ranks / (flat.size - 1) - 0.5).reshape(x.shape)              # :contentReference[oaicite:6]{index=6}


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_fitness_multy_objective(
    rewards: jnp.ndarray,
    skill_improvement: jnp.ndarray,
    k1: float = 1.0,
    k2: float = 2.0,
    k3: float = 0.0,
) -> jnp.ndarray:
    """Combine mean, worst‑case reward and skill term."""
    rew_mean = k1 * jnp.mean(rewards)
    rew_min = k2 * jnp.min(rewards)
    return rew_mean + rew_min + k3 * jnp.mean(skill_improvement)


get_fitness_multy_objective_vmap = jax.vmap(
    get_fitness_multy_objective, in_axes=(0, 0, None, None, None)
)


@jax.jit
def get_fitness_final_perf(final_perf: jnp.ndarray) -> jnp.ndarray:
    """Scalar fitness from final‑task performance."""
    return jnp.mean(final_perf)


get_fitness_final_perf_vmap = jax.vmap(get_fitness_final_perf)


# ---------------------------------------------------------------------
# 9. Optimiser factory
# ---------------------------------------------------------------------

def build_adamw(
    lr: float = 1e-2, weight_decay: float = 2e-2
) -> tuple[optax.GradientTransformation, optax.OptState]:
    """
    Convenience wrapper around Optax AdamW.                      # :contentReference[oaicite:7]{index=7}

    Returns
    -------
    optim : optax.GradientTransformation
    opt_state : optax.OptState
    """
    optim = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    # The caller must call `optim.init(params)` after parameters are known
    return optim
