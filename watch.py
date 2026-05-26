"""
Watch a saved policy checkpoint in the MuJoCo interactive viewer.
Run in a second terminal while training is going (loads latest checkpoint).

Usage:
    .venv/bin/python3 watch.py
    .venv/bin/python3 watch.py --checkpoint checkpoints/ckpt_00050.pkl
"""
import argparse
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from envs.go2_joystick import (
    Q_DEFAULT, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    KP, KD, N_SUBSTEPS, SIM_HZ, CTRL_HZ, STANDING_HEIGHT,
    _find_xml, _rotate_vec_by_quat_inv, _sample_cmd,
)
from policy import ActorCritic

CTRL_DT = 1.0 / CTRL_HZ


def load_latest_checkpoint(checkpoint_dir: str) -> dict | None:
    ckpts = sorted(Path(checkpoint_dir).glob("*.pkl"))
    if not ckpts:
        return None
    with open(ckpts[-1], "rb") as f:
        return pickle.load(f)


def get_obs(data: mujoco.MjData, cmd_vel: np.ndarray,
            last_action: np.ndarray) -> jax.Array:
    quat = data.qpos[3:7]
    gravity_body = _rotate_vec_by_quat_inv(
        jnp.array([0.0, 0.0, -1.0]),
        jnp.array(quat),
    )
    ang_vel = jnp.array(data.qvel[3:6])
    q_joint = jnp.array(data.qpos[7:]) - Q_DEFAULT
    dq_joint = jnp.array(data.qvel[6:])
    return jnp.concatenate([
        ang_vel, gravity_body,
        jnp.array(cmd_vel),
        q_joint, dq_joint,
        jnp.array(last_action),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()

    # Load checkpoint
    if args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            ckpt = pickle.load(f)
    else:
        ckpt = load_latest_checkpoint(args.checkpoint_dir)

    if ckpt is None:
        print("No checkpoint found. Running with random policy.")
        model_nn = ActorCritic()
        params = model_nn.init(jax.random.PRNGKey(0), jnp.zeros(45))
    else:
        params = ckpt["params"] if isinstance(ckpt, dict) else ckpt
        step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
        print(f"Loaded checkpoint (step={step})")

    model_nn = ActorCritic()

    @jax.jit
    def infer(params, obs):
        action_delta, _, _ = model_nn.apply(params, obs)
        return jnp.clip(Q_DEFAULT + action_delta, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

    # MuJoCo model + viewer
    xml_path = _find_xml()
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.timestep = 1.0 / SIM_HZ
    mj_data = mujoco.MjData(mj_model)

    # Reset to standing pose
    mj_data.qpos[2] = STANDING_HEIGHT
    mj_data.qpos[3] = 1.0
    mj_data.qpos[7:] = np.array(Q_DEFAULT)
    mujoco.mj_forward(mj_model, mj_data)

    cmd_vel = np.array([args.vx, args.vy, args.yaw])
    last_action = np.zeros(12)

    print(f"Command: vx={args.vx}  vy={args.vy}  yaw={args.yaw}")
    print("Viewer controls: mouse to rotate, scroll to zoom, Esc to quit")

    # Warm up JIT
    dummy_obs = jnp.zeros(45)
    _ = infer(params, dummy_obs)
    print("Ready.")

    step_count = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        while viewer.is_running():
            t_start = time.monotonic()

            # Policy inference every N_SUBSTEPS
            if step_count % N_SUBSTEPS == 0:
                obs = get_obs(mj_data, cmd_vel, last_action)
                target_q = infer(params, obs)
                last_action = np.array(target_q) - np.array(Q_DEFAULT)

            # PD torques
            q_joint = mj_data.qpos[7:]
            dq_joint = mj_data.qvel[6:]
            torques = KP * (np.array(target_q) - q_joint) - KD * dq_joint
            mj_data.ctrl[:] = np.clip(torques, -23.7, 23.7)

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()

            step_count += 1

            # Keep real-time
            elapsed = time.monotonic() - t_start
            sleep_t = (1.0 / SIM_HZ) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
