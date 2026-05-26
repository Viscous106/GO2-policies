"""Shared policy network used by train.py, export_policy.py, and deploy/."""
import flax.linen as nn
import jax
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """ELU MLP actor-critic for 45→12 locomotion."""
    action_dim: int = 12

    @nn.compact
    def __call__(self, obs: jax.Array):
        x = nn.Dense(512)(obs)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)

        action_mean = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        value = nn.Dense(1)(x).squeeze(-1)
        return action_mean, log_std, value
