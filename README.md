# Go2 Sim-to-Real Locomotion

PPO locomotion policy for the Unitree Go2 trained in MuJoCo MJX (JAX).  
Trains at ~8000 fps on RTX 4080 Super with 4096 parallel envs.

---

## Fresh Ubuntu setup (start here)

### 1. Check your GPU driver
```bash
nvidia-smi   # should show your GPU and driver version
```
If `nvidia-smi` is missing, install the driver first:
```bash
sudo apt install -y nvidia-driver-535   # or whichever is latest
sudo reboot
```

### 2. Install system prerequisites
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### 3. Clone the repo
```bash
git clone https://github.com/Yash-Virulkar/go2_sim2real   # replace with your actual URL
cd go2_sim2real
```

### 4. Run setup
```bash
bash setup.sh
```

This does everything in one shot:
- Creates `.venv/` with Python
- Installs `mujoco`, `mujoco-mjx`, `jax[cuda12]` (includes cuDNN — no extra apt install needed), `flax`, `optax`
- Clones the Go2 MuJoCo model to `../mujoco_menagerie/` (next to the repo)
- Verifies the model loads correctly

Expected output at the end:
```
Go2 model loaded: nq=19, nu=12, nv=18
Setup complete.
```

### 5. Verify GPU is working
```bash
.venv/bin/python3 -c "import jax; print(jax.devices())"
# should print: [CudaDevice(id=0)]
```

If it prints `[CpuDevice(id=0)]` instead, your CUDA driver is not visible to JAX:
```bash
.venv/bin/pip install "jax[cuda12]" --upgrade
```

---

## Watch a trained policy in the MuJoCo viewer

```bash
# Watch the final trained policy (vx=0.5 m/s forward)
.venv/bin/python3 watch.py --checkpoint checkpoints/policy_final.pkl --vx 0.5

# Try different velocity commands
.venv/bin/python3 watch.py --checkpoint checkpoints/policy_final.pkl --vx 1.0
.venv/bin/python3 watch.py --checkpoint checkpoints/policy_final.pkl --vx 0.3 --yaw 0.5

# Watch a mid-training checkpoint (iter 150)
.venv/bin/python3 watch.py --checkpoint checkpoints/ckpt_00150.pkl --vx 0.5
```

Viewer controls: mouse to rotate, scroll to zoom, Esc to quit.

> **Note:** The current policy was trained to 50M steps (mean_rew ≈ -0.63).
> It should stay upright and show early walking behaviour but won't walk perfectly yet.
> If it looks bad, continue training (see below).

---

## Continue training (recommended on 4080 Super)

```bash
# Another 50M steps — takes ~1.5 hrs on 4080 Super, should push reward toward +1.0
.venv/bin/python3 train.py --n_envs 4096 --steps 50_000_000
```

Watch reward in the output. Target: `mean_rew > 1.0` = robot is tracking velocity.

To resume from the existing checkpoint instead of starting fresh, the current
`train.py` always starts from random init. If you want to resume, load the pkl:

```bash
# Resume is not yet wired into train.py — retrain from scratch on the faster GPU
.venv/bin/python3 train.py --n_envs 4096 --steps 100_000_000
```

---

## Train from scratch

```bash
.venv/bin/python3 train.py \
  --n_envs 4096 \
  --steps 50_000_000 \
  --checkpoint_dir checkpoints/
```

Checkpoints save every 50 iterations to `checkpoints/ckpt_XXXXX.pkl`.  
Final policy saves to `checkpoints/policy_final.pkl`.

---

## Deploy to real robot (after policy is good in sim)

Connect the Go2 via Ethernet, then:

```bash
pip install unitree_sdk2py

python deploy/run_policy.py \
  --checkpoint checkpoints/policy_final.pkl \
  --iface eth0 \
  --vx 0.0   # start with zero velocity to check stance
```

**Always start with `--vx 0.0` to verify stance before commanding movement.**  
Press Ctrl-C to stop — sends robot to safe standing pose.

---

## Project structure

```
go2_sim2real/
├── envs/go2_joystick.py   # MJX env — 45-dim obs, 12-dim action, 4096 parallel
├── policy.py              # ActorCritic 512→256→128 ELU
├── train.py               # PPO training loop (JAX/Flax)
├── randomize.py           # Domain randomisation (friction, mass, armature)
├── watch.py               # MuJoCo viewer for trained policy
├── export_policy.py       # ONNX export (optional)
├── deploy/run_policy.py   # Real robot 50 Hz deployment
├── setup.sh               # One-shot setup
└── checkpoints/
    ├── policy_final.pkl   # Trained policy (50M steps)
    └── ckpt_00150.pkl     # Mid-training checkpoint
```

## Key constants

| Parameter | Value | Notes |
|---|---|---|
| Control freq | 50 Hz | 10 physics substeps at 500 Hz |
| Observation | 45-dim | ang_vel, gravity, cmd, q, dq, last_action |
| Action | 12-dim | joint position deltas from default |
| kp / kd | 50 / 0.5 | PD gains, matches real robot |
| Default pose | [0, 0.9, -1.8] × 4 | From go2_mjx.xml keyframe |
| Joint order (model) | FL, FR, RL, RR | Permuted to FR, FL, RR, RL for SDK |
