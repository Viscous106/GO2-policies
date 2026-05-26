"""Domain randomization for Go2 MJX training."""
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def randomize_model(model: mjx.Model, rng: jax.Array) -> mjx.Model:
    """Apply per-episode domain randomization to a batched MJX model."""
    rng, r1, r2, r3, r4 = jax.random.split(rng, 5)

    # Floor friction: uniform [0.4, 1.0]
    friction = jax.random.uniform(r1, shape=model.geom_friction[:, 0:1].shape,
                                   minval=0.4, maxval=1.0)
    new_friction = model.geom_friction.at[:, 0:1].set(friction)

    # Link mass scale: × uniform [0.9, 1.1]
    mass_scale = jax.random.uniform(r2, shape=model.body_mass.shape,
                                     minval=0.9, maxval=1.1)
    new_mass = model.body_mass * mass_scale

    # Torso mass offset: uniform [-1.0, 1.0] kg  (body 1 = torso)
    torso_offset = jax.random.uniform(r3, shape=(), minval=-1.0, maxval=1.0)
    new_mass = new_mass.at[1].add(torso_offset)

    # Armature scale: × uniform [1.0, 1.05]
    arm_scale = jax.random.uniform(r4, shape=model.dof_armature.shape,
                                    minval=1.0, maxval=1.05)
    new_armature = model.dof_armature * arm_scale

    return model.replace(
        geom_friction=new_friction,
        body_mass=new_mass,
        dof_armature=new_armature,
    )


def randomize_init_qpos(qpos: jax.Array, rng: jax.Array,
                         noise: float = 0.05) -> jax.Array:
    """Add uniform noise to initial joint positions (joints only, not root)."""
    noise_vec = jax.random.uniform(rng, shape=qpos.shape,
                                   minval=-noise, maxval=noise)
    # Don't perturb the root position/orientation (first 7 elements)
    noise_vec = noise_vec.at[:7].set(0.0)
    return qpos + noise_vec
