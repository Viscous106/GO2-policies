"""
Go2 joystick locomotion environment for MJX.

Adapted from MuJoCo Playground Go1 joystick env.
Observation (45-dim): ang_vel(3) | proj_gravity(3) | cmd_vel(3) |
                       q-q0(12) | dq(12) | last_action(12)
Action (12-dim): target joint positions (clipped to limits)

MuJoCo model actuator order (indices 0-11):
  FL_hip, FL_thigh, FL_calf,
  FR_hip, FR_thigh, FR_calf,
  RL_hip, RL_thigh, RL_calf,
  RR_hip, RR_thigh, RR_calf

SDK motor order (for deployment, see deploy/run_policy.py):
  FR_hip, FR_thigh, FR_calf,
  FL_hip, FL_thigh, FL_calf,
  RR_hip, RR_thigh, RR_calf,
  RL_hip, RL_thigh, RL_calf

Permutation to convert model → SDK: [3,4,5, 0,1,2, 9,10,11, 6,7,8]
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from randomize import randomize_model, randomize_init_qpos

# ---------------------------------------------------------------------------
# Constants from XML keyframe + CONTEXT.md
# ---------------------------------------------------------------------------

# MuJoCo model actuator order: FL, FR, RL, RR
JOINT_ORDER_MODEL = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Default joint angles from go2_mjx.xml keyframe "home" (ctrl=...)
Q_DEFAULT = jnp.array([
    0.0,  0.9, -1.8,   # FL
    0.0,  0.9, -1.8,   # FR
    0.0,  0.9, -1.8,   # RL
    0.0,  0.9, -1.8,   # RR
])

# Joint limits in model order (from CONTEXT.md, same per leg)
JOINT_LIMITS_LOW = jnp.array([
    -1.047, -1.571, -2.723,   # FL
    -1.047, -1.571, -2.723,   # FR
    -1.047, -1.571, -2.723,   # RL
    -1.047, -1.571, -2.723,   # RR
])

JOINT_LIMITS_HIGH = jnp.array([
    1.047, 3.491, -0.838,   # FL
    1.047, 3.491, -0.838,   # FR
    1.047, 3.491, -0.838,   # RL
    1.047, 3.491, -0.838,   # RR
])

# Permutation: model-order → SDK-order  (FL,FR,RL,RR → FR,FL,RR,RL)
MODEL_TO_SDK = jnp.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8], dtype=jnp.int32)
# Permutation: SDK-order → model-order  (same permutation — it's self-inverse)
SDK_TO_MODEL = MODEL_TO_SDK

KP = 50.0
KD = 0.5
TORQUE_LIMIT = 23.7

CTRL_HZ = 50
SIM_HZ = 500
N_SUBSTEPS = SIM_HZ // CTRL_HZ   # = 10

MAX_STEPS = 1000   # 20 s at 50 Hz
STANDING_HEIGHT = 0.27   # from XML keyframe


class State(NamedTuple):
    mjx_state: mjx.Data
    last_action: jax.Array   # (12,)  delta from Q_DEFAULT
    step: jax.Array          # scalar int
    cmd_vel: jax.Array       # (3,)  [vx, vy, yaw_rate]
    done: jax.Array          # scalar bool
    reward: jax.Array        # scalar float


class Go2JoystickEnv:
    """Vectorised Go2 joystick environment running on MJX."""

    def __init__(self, xml_path: str | None = None, n_envs: int = 4096):
        self.n_envs = n_envs
        xml_path = xml_path or _find_xml()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.timestep = 1.0 / SIM_HZ
        self.mjx_model = mjx.put_model(self.mj_model)
        self._q0 = _build_default_qpos(self.mj_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        rng, rng_model, rng_qpos, rng_cmd = jax.random.split(rng, 4)

        model = randomize_model(self.mjx_model, rng_model)
        q0 = randomize_init_qpos(self._q0, rng_qpos)

        mj_data = mjx.make_data(model)
        mj_data = mj_data.replace(qpos=q0, qvel=jnp.zeros_like(mj_data.qvel))
        mj_data = mjx.forward(model, mj_data)

        cmd_vel = _sample_cmd(rng_cmd)

        return State(
            mjx_state=mj_data,
            last_action=jnp.zeros(12),
            step=jnp.int32(0),
            cmd_vel=cmd_vel,
            done=jnp.bool_(False),
            reward=jnp.float32(0.0),
        )

    def step(self, state: State, action: jax.Array, rng: jax.Array) -> State:
        """Advance one control step (N_SUBSTEPS physics steps).

        action: (12,) target joint angles in model order, expressed as
                delta from Q_DEFAULT (policy output convention).

        Auto-resets the physics state when done=True so the returned State
        is always valid for the next step (PPO collects one long trajectory
        per worker — resets happen in-place).
        """
        model = self.mjx_model

        # --- Physics step (on current state, regardless of done) ---
        mj_data = state.mjx_state
        target_q = jnp.clip(Q_DEFAULT + action, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        torques = KP * (target_q - mj_data.qpos[7:]) - KD * mj_data.qvel[6:]
        torques = jnp.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        mj_data = mj_data.replace(ctrl=torques)

        def substep(d, _):
            return mjx.step(model, d), None

        mj_data, _ = jax.lax.scan(substep, mj_data, None, length=N_SUBSTEPS)

        reward = _compute_reward(mj_data, state.cmd_vel, action, state.last_action)
        done = _is_terminated(mj_data, state.step)

        # --- Auto-reset: rebuild a fresh state when episode ends ---
        rng, rng_model, rng_qpos, rng_cmd = jax.random.split(rng, 4)
        fresh_q0 = randomize_init_qpos(self._q0, rng_qpos)
        fresh_data = mjx.make_data(randomize_model(model, rng_model))
        fresh_data = fresh_data.replace(qpos=fresh_q0,
                                        qvel=jnp.zeros_like(fresh_data.qvel))
        fresh_data = mjx.forward(model, fresh_data)
        fresh_cmd = _sample_cmd(rng_cmd)

        # jnp.where over PyTree fields — select fresh vs. continuing
        def _select(fresh, cont):
            return jnp.where(done, fresh, cont)

        next_data = jax.tree.map(_select, fresh_data, mj_data)
        next_cmd = jnp.where(done, fresh_cmd, state.cmd_vel)
        next_step = jnp.where(done, jnp.int32(0), state.step + 1)
        next_action = jnp.where(done, jnp.zeros(12), action)

        return State(
            mjx_state=next_data,
            last_action=next_action,
            step=next_step,
            cmd_vel=next_cmd,
            done=done,
            reward=reward,
        )

    def get_obs(self, state: State) -> jax.Array:
        """Return 45-dim observation vector."""
        d = state.mjx_state
        quat = d.qpos[3:7]
        gravity_body = _rotate_vec_by_quat_inv(jnp.array([0.0, 0.0, -1.0]), quat)
        ang_vel = d.qvel[3:6]
        q_joint = d.qpos[7:] - Q_DEFAULT
        dq_joint = d.qvel[6:]

        return jnp.concatenate([
            ang_vel,            # 3
            gravity_body,       # 3
            state.cmd_vel,      # 3
            q_joint,            # 12
            dq_joint,           # 12
            state.last_action,  # 12
        ])  # = 45


# ---------------------------------------------------------------------------
# Pure functions (easier to JIT independently)
# ---------------------------------------------------------------------------

def _compute_reward(d: mjx.Data, cmd_vel: jax.Array,
                    action: jax.Array, last_action: jax.Array) -> jax.Array:
    vel_body = d.qvel[:3]
    ang_vel = d.qvel[3:6]

    lin_vel_err = jnp.sum(jnp.square(cmd_vel[:2] - vel_body[:2]))
    r_lin = jnp.exp(-lin_vel_err / 0.25)

    ang_vel_err = jnp.square(cmd_vel[2] - ang_vel[2])
    r_ang = jnp.exp(-ang_vel_err / 0.25)

    target_q = jnp.clip(Q_DEFAULT + action, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    r_torque = -1e-4 * jnp.sum(jnp.square(
        KP * (target_q - d.qpos[7:]) - KD * d.qvel[6:]
    ))
    r_action_rate = -0.01 * jnp.sum(jnp.square(action - last_action))
    r_z_vel = -2.0 * jnp.square(d.qvel[2])
    r_ang_vel_xy = -0.05 * jnp.sum(jnp.square(ang_vel[:2]))

    return r_lin + r_ang + r_torque + r_action_rate + r_z_vel + r_ang_vel_xy


def _is_terminated(d: mjx.Data, step: jax.Array) -> jax.Array:
    fallen = d.qpos[2] < 0.18
    timeout = step >= MAX_STEPS
    return jnp.logical_or(fallen, timeout)


def _sample_cmd(rng: jax.Array) -> jax.Array:
    rng_vx, rng_vy, rng_yaw = jax.random.split(rng, 3)
    vx = jax.random.uniform(rng_vx, minval=-1.0, maxval=1.5)
    vy = jax.random.uniform(rng_vy, minval=-0.5, maxval=0.5)
    yaw = jax.random.uniform(rng_yaw, minval=-1.0, maxval=1.0)
    return jnp.array([vx, vy, yaw])


def _rotate_vec_by_quat_inv(v: jax.Array, q: jax.Array) -> jax.Array:
    """Rotate vector v by the inverse of unit quaternion q (w,x,y,z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    qc = jnp.array([w, -x, -y, -z])
    t = 2.0 * jnp.cross(qc[1:], v)
    return v + qc[0] * t + jnp.cross(qc[1:], t)


def _build_default_qpos(mj_model: mujoco.MjModel) -> jax.Array:
    """Build full qpos vector matching the XML home keyframe."""
    import numpy as np
    qpos = np.zeros(mj_model.nq)
    qpos[2] = STANDING_HEIGHT
    qpos[3] = 1.0          # quaternion w=1 (identity)
    qpos[7:] = Q_DEFAULT
    return jnp.array(qpos)


def _find_xml() -> str:
    candidates = [
        Path.home() / "mujoco_menagerie/unitree_go2/go2_mjx.xml",
        Path("/usr/share/mujoco_menagerie/unitree_go2/go2_mjx.xml"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "go2_mjx.xml not found. Run:\n"
        "  git clone --depth=1 --filter=blob:none --sparse "
        "https://github.com/google-deepmind/mujoco_menagerie ~/mujoco_menagerie\n"
        "  cd ~/mujoco_menagerie && git sparse-checkout set unitree_go2"
    )
