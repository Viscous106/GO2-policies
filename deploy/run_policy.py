"""
Real-robot deployment loop for Go2 locomotion policy.

Run on a machine connected to the Go2 via Ethernet (same subnet).
Usage:
    python deploy/run_policy.py --checkpoint checkpoints/policy_final.pkl \
                                --iface eth0 [--vx 0.5] [--vy 0.0] [--yaw 0.0]

Safety:
    - Press Ctrl-C to stop and zero all motor commands.
    - Policy runs at 50 Hz; the SDK enforces its own safety limits.
    - Always have your hand on the emergency stop before running.
"""
from __future__ import annotations

import argparse
import pickle
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    import unitree_sdk2py.core.channel as ch
    from unitree_sdk2py.go2.low_level.go2_low_level_commander import Go2LowLevelCommander
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    print("[WARNING] unitree_sdk2py not found — running in dry-run mode")

from envs.go2_joystick import (
    Q_DEFAULT, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
    KP, KD, CTRL_HZ, MODEL_TO_SDK, SDK_TO_MODEL,
)
from policy import ActorCritic

CTRL_DT = 1.0 / CTRL_HZ

# SDK motor index 0..11:  FR, FL, RR, RL (each: hip, thigh, calf)
MOTOR_NAMES_SDK = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]


class PolicyRunner:
    def __init__(self, params, cmd_vel: np.ndarray, iface: str):
        self.params = params
        self.model = ActorCritic()
        self.cmd_vel = jnp.array(cmd_vel, dtype=jnp.float32)
        self.last_action = jnp.array(Q_DEFAULT)
        self.iface = iface
        self._running = True
        self._low_state: LowState_ | None = None

        if HAS_SDK:
            ch.ChannelFactoryInitialize(0, iface)
            self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._pub.Init()
            self._sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._sub.Init(self._state_callback, 10)

        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _state_callback(self, msg: LowState_):
        self._low_state = msg

    def _stop(self, *_):
        self._running = False

    def _get_obs(self) -> jax.Array | None:
        if not HAS_SDK or self._low_state is None:
            return None
        s = self._low_state

        # Read motor states — SDK order (FR,FL,RR,RL) → model order (FL,FR,RL,RR)
        q_sdk = jnp.array([s.motor_state[i].q for i in range(12)])
        dq_sdk = jnp.array([s.motor_state[i].dq for i in range(12)])
        q = q_sdk[SDK_TO_MODEL]
        dq = dq_sdk[SDK_TO_MODEL]

        # IMU
        ang_vel = jnp.array(s.imu_state.gyroscope)     # (3,)
        quat = jnp.array(s.imu_state.quaternion)        # (w,x,y,z) or (x,y,z,w)?
        # unitree SDK returns (w,x,y,z)
        proj_gravity = _rotate_vec_by_quat_inv(
            jnp.array([0.0, 0.0, -1.0]), quat
        )

        obs = jnp.concatenate([
            ang_vel,
            proj_gravity,
            self.cmd_vel,
            q - Q_DEFAULT,
            dq,
            self.last_action,
        ])
        return obs

    def _send_cmd(self, target_q: jax.Array):
        """target_q is in model order (FL,FR,RL,RR); permute to SDK order."""
        if not HAS_SDK:
            return
        target_q_sdk = target_q[MODEL_TO_SDK]
        cmd = LowCmd_()
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = float(target_q_sdk[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = KP
            cmd.motor_cmd[i].kd = KD
            cmd.motor_cmd[i].tau = 0.0
        self._pub.Write(cmd)

    def _send_zero(self):
        """Send soft-hold to default pose in SDK motor order."""
        if not HAS_SDK:
            return
        q_default_sdk = Q_DEFAULT[MODEL_TO_SDK]
        cmd = LowCmd_()
        for i in range(12):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = float(q_default_sdk[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 5.0   # very soft gains for settling
            cmd.motor_cmd[i].kd = 1.0
            cmd.motor_cmd[i].tau = 0.0
        self._pub.Write(cmd)

    def run(self):
        print(f"Policy running at {CTRL_HZ} Hz. Ctrl-C to stop.")
        if not HAS_SDK:
            print("(dry-run: no robot connected)")

        @jax.jit
        def _infer(params, obs):
            # Policy outputs deltas from Q_DEFAULT
            action_delta, _, _ = ActorCritic().apply(params, obs)
            target_q = jnp.clip(Q_DEFAULT + action_delta,
                                 JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            return target_q, action_delta

        # Warm up JIT
        dummy_obs = jnp.zeros(45)
        _ = _infer(self.params, dummy_obs)
        print("JIT warm-up done.")

        while self._running:
            t_start = time.monotonic()

            obs = self._get_obs()
            if obs is None:
                target_q = Q_DEFAULT
            else:
                target_q, action_delta = _infer(self.params, obs)
                self.last_action = action_delta

            self._send_cmd(target_q)

            elapsed = time.monotonic() - t_start
            sleep_time = CTRL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[WARNING] Loop over-ran by {-sleep_time*1000:.1f} ms")

        print("Stopping — sending safe pose...")
        self._send_zero()
        time.sleep(0.5)
        print("Done.")


def _rotate_vec_by_quat_inv(v: jax.Array, q: jax.Array) -> jax.Array:
    w, x, y, z = q[0], q[1], q[2], q[3]
    qc = jnp.array([w, -x, -y, -z])
    t = 2.0 * jnp.cross(qc[1:], v)
    return v + qc[0] * t + jnp.cross(qc[1:], t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pkl checkpoint")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS")
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rate command (rad/s)")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if isinstance(ckpt, dict) else ckpt
    print(f"Loaded policy (step={ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'})")

    cmd_vel = np.array([args.vx, args.vy, args.yaw])
    runner = PolicyRunner(params, cmd_vel, args.iface)
    runner.run()


if __name__ == "__main__":
    main()
