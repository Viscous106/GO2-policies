# Go2 Sim-to-Real — Project Context

## Goal
Train a locomotion policy for the Unitree Go2 quadruped in MuJoCo/MJX and deploy it to the real robot via `unitree_sdk2_python`.

## Stack
- **Simulator**: MuJoCo + MJX (JAX-based GPU-parallel)
- **Robot model**: `mujoco_menagerie/unitree_go2/go2_mjx.xml` (12 DoF)
- **Training base**: MuJoCo Playground Go1 joystick env (Go2 is drop-in — same 12 DoF)
- **RL**: Brax PPO (JAX)
- **Deployment**: `unitree_sdk2_python` LowCmd over DDS/Ethernet at 50 Hz

## Robot: Unitree Go2
- 12 actuators: 3 per leg × 4 legs (hip_abduction, hip_pitch, knee)
- Joint order (must match policy output): `FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf`
- Force limits: ±23.7 N·m

## Sim PD Gains (go2_mjx.xml)
- kp = 50, kd = 0.5 (use these on real robot too to avoid gain mismatch)

## Joint Limits
| Joint      | Range (rad)         |
|------------|---------------------|
| Hip abduct | -1.047 to  1.047    |
| Hip pitch  | -1.571 to  3.491    |
| Knee       | -2.723 to -0.838    |

## Observation Space (45-dim)
```
ang_vel (3) | projected_gravity (3) | cmd_vel (3) | 
q - q_default (12) | dq (12) | last_action (12)
```

## Action Space
- 12 target joint positions, clipped to joint limits
- Policy runs at 50 Hz

## Domain Randomization (per episode)
| Parameter        | Range                  |
|------------------|------------------------|
| Floor friction   | uniform [0.4, 1.0]     |
| Link masses      | × uniform [0.9, 1.1]   |
| Torso mass offset| uniform [-1.0, 1.0] kg |
| Armature         | × uniform [1.0, 1.05]  |
| Init joint pos   | + uniform [-0.05, 0.05] rad |

## Real Robot Deployment Snippet
```python
import unitree_sdk2py.core.channel as ch
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

ch.ChannelFactoryInitialize(0, "eth0")  # adjust interface

# 50 Hz control loop
cmd = LowCmd_()
for i in range(12):
    cmd.motor_cmd[i].mode = 0x01
    cmd.motor_cmd[i].q   = float(target_q[i])
    cmd.motor_cmd[i].dq  = 0.0
    cmd.motor_cmd[i].kp  = 50.0
    cmd.motor_cmd[i].kd  = 0.5
    cmd.motor_cmd[i].tau = 0.0
publisher.Write(cmd)
```

## Key Repos
- https://github.com/google-deepmind/mujoco
- https://github.com/google-deepmind/mujoco_menagerie  (model: unitree_go2/)
- https://github.com/google-deepmind/mujoco_playground  (base env: go1/ → adapt for go2)
- https://github.com/unitreerobotics/unitree_sdk2_python

## Project Directory Layout (target)
```
go2_sim2real/
├── CONTEXT.md          ← this file
├── envs/
│   └── go2_joystick.py ← MJX env adapted from Go1 playground
├── train.py            ← Brax PPO training script
├── randomize.py        ← domain randomization
├── deploy/
│   └── run_policy.py   ← real robot deployment loop
└── checkpoints/        ← saved policy weights
```

## Current Status
- [ ] Clone menagerie + verify go2_mjx.xml loads
- [ ] Adapt Go1 playground env → Go2 (swap XML, update constants)
- [ ] Add domain randomization
- [ ] Train PPO policy
- [ ] Export policy (ONNX or pickle)
- [ ] Deploy on real Go2 via LowCmd

## Notes
- MJX Go2 model uses sphere collision approximations (MJX limitation) — fine for training
- Verify joint ordering between MuJoCo actuator list and SDK motor indices before first hardware run
- Go2 SDK communicates over DDS; robot must be on same subnet, check with `ping 192.168.123.161`
