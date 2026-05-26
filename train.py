"""
PPO training for Go2 joystick locomotion using MJX + Flax.

Usage:
    .venv/bin/python3 train.py [--xml PATH] [--n_envs 4096] [--steps 50_000_000]
                               [--checkpoint_dir checkpoints/]

GPU: requires cuDNN. On Arch Linux: sudo pacman -S cudnn
Without cuDNN, training runs on CPU (correct but slow — reduce --n_envs).
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from envs.go2_joystick import Go2JoystickEnv
from policy import ActorCritic

# ---------------------------------------------------------------------------
# GAE (vectorised over the time axis via lax.scan)
# ---------------------------------------------------------------------------

def compute_gae(rewards: jax.Array, values: jax.Array, dones: jax.Array,
                last_value: jax.Array,
                gamma: float = 0.99, lam: float = 0.95) -> tuple:
    """
    rewards, values, dones: (T, n_envs)
    last_value:             (n_envs,)  — bootstrap value for the last step
    Returns: advantages, returns  each (T, n_envs)
    """
    T = rewards.shape[0]

    def _scan_fn(gae, x):
        r, v, next_v, d = x
        delta = r + gamma * next_v * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        return gae, gae

    # Build next-values: shift values by one step, use last_value for final
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)  # (T, E)

    # scan over time in reverse
    _, advantages = jax.lax.scan(
        _scan_fn,
        jnp.zeros_like(last_value),
        (rewards[::-1], values[::-1], next_values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]   # (T, E)
    returns = advantages + values   # (T, E)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss (fully jittable)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("model",))
def ppo_update(train_state: TrainState, model: ActorCritic,
               obs_batch: jax.Array, act_batch: jax.Array,
               old_log_probs: jax.Array, advantages: jax.Array,
               returns: jax.Array,
               clip_eps: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.01):

    def loss_fn(params):
        action_mean, log_std, values = jax.vmap(
            lambda o: model.apply(params, o)
        )(obs_batch)

        std = jnp.exp(log_std)
        diff = (act_batch - action_mean) / (std + 1e-8)
        log_probs = -0.5 * jnp.sum(
            jnp.square(diff) + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
            axis=-1,
        )
        entropy = jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1).mean()

        ratio = jnp.exp(log_probs - old_log_probs)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg1 = -adv * ratio
        pg2 = -adv * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg1, pg2).mean()
        vf_loss = jnp.square(values - returns).mean()
        loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy
        return loss, (pg_loss, vf_loss, entropy)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss, aux


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    rng = jax.random.PRNGKey(42)
    print(f"JAX devices: {jax.devices()}")

    env = Go2JoystickEnv(xml_path=args.xml, n_envs=args.n_envs)
    model = ActorCritic()

    rng, rng_init = jax.random.split(rng)
    params = model.init(rng_init, jnp.zeros(45))

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=3e-4),
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Vectorised reset / step / obs
    rng, *env_rngs = jax.random.split(rng, args.n_envs + 1)
    states = jax.vmap(env.reset)(jnp.array(env_rngs))

    @jax.jit
    def policy_step_batch(params, obs_batch, rng):
        """Single-device batched policy inference."""
        def single(o, r):
            action_mean, log_std, value = model.apply(params, o)
            std = jnp.exp(log_std)
            noise = jax.random.normal(r, shape=action_mean.shape)
            action = action_mean + std * noise
            log_prob = -0.5 * jnp.sum(
                jnp.square(noise) + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            return action, log_prob, value
        rngs = jax.random.split(rng, obs_batch.shape[0])
        return jax.vmap(single)(obs_batch, rngs)

    @jax.jit
    def env_step_batch(states, actions, rng):
        rngs = jax.random.split(rng, args.n_envs)
        return jax.vmap(env.step)(states, actions, rngs)

    @jax.jit
    def get_obs_batch(states):
        return jax.vmap(env.get_obs)(states)

    @jax.jit
    def get_value_batch(params, obs_batch):
        _, _, values = jax.vmap(lambda o: model.apply(params, o))(obs_batch)
        return values

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    rollout_len = 64
    n_epochs = 4
    batch_size = 2048
    total_steps = 0
    iteration = 0

    print(f"n_envs={args.n_envs}  rollout_len={rollout_len}  "
          f"target_steps={args.steps:,}")

    while total_steps < args.steps:
        t0 = time.time()

        # ---- Collect rollout ------------------------------------------------
        obs_list, act_list, lp_list, val_list, rew_list, done_list = [], [], [], [], [], []

        for _ in range(rollout_len):
            obs_batch = get_obs_batch(states)           # (E, 45)

            rng, rng_act = jax.random.split(rng)
            actions, log_probs, values = policy_step_batch(
                train_state.params, obs_batch, rng_act
            )                                           # (E, 12), (E,), (E,)

            rng, rng_step = jax.random.split(rng)
            states = env_step_batch(states, actions, rng_step)

            obs_list.append(obs_batch)
            act_list.append(actions)
            lp_list.append(log_probs)
            val_list.append(values)
            rew_list.append(states.reward)
            done_list.append(states.done.astype(jnp.float32))

        total_steps += rollout_len * args.n_envs

        # Stack: (T, E, ...)
        obs_arr  = jnp.stack(obs_list)   # (T, E, 45)
        act_arr  = jnp.stack(act_list)   # (T, E, 12)
        lp_arr   = jnp.stack(lp_list)    # (T, E)
        val_arr  = jnp.stack(val_list)   # (T, E)
        rew_arr  = jnp.stack(rew_list)   # (T, E)
        done_arr = jnp.stack(done_list)  # (T, E)

        # Bootstrap value for last step
        last_obs = get_obs_batch(states)
        last_val = get_value_batch(train_state.params, last_obs)  # (E,)

        advantages, returns = compute_gae(rew_arr, val_arr, done_arr, last_val)

        # Flatten (T, E, ...) → (T*E, ...)
        N = rollout_len * args.n_envs
        obs_flat  = obs_arr.reshape(N, 45)
        act_flat  = act_arr.reshape(N, 12)
        lp_flat   = lp_arr.reshape(N)
        adv_flat  = advantages.reshape(N)
        ret_flat  = returns.reshape(N)

        # ---- PPO update -----------------------------------------------------
        loss_val = 0.0
        for _ in range(n_epochs):
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, N)
            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                train_state, loss_val, _ = ppo_update(
                    train_state, model,
                    obs_flat[idx], act_flat[idx], lp_flat[idx],
                    adv_flat[idx], ret_flat[idx],
                )

        dt = time.time() - t0
        fps = rollout_len * args.n_envs / dt
        print(f"iter={iteration:4d}  steps={total_steps:>12,}  "
              f"mean_rew={float(rew_arr.mean()):.3f}  "
              f"fps={fps:.0f}  loss={float(loss_val):.4f}")

        if iteration % 50 == 0:
            ckpt = {"params": train_state.params, "step": total_steps}
            ckpt_path = checkpoint_dir / f"ckpt_{iteration:05d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  -> saved {ckpt_path}")

        iteration += 1

    final_path = checkpoint_dir / "policy_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"params": train_state.params, "step": total_steps}, f)
    print(f"Training complete. Saved: {final_path}")
    return train_state.params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=None)
    parser.add_argument("--n_envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
