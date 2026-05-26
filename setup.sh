#!/usr/bin/env bash
# One-shot setup for the Go2 sim-to-real project.
# Run once from the project root: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "=== Creating venv ==="
python3 -m venv .venv --system-site-packages

echo "=== Installing Python dependencies ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install mujoco mujoco-mjx flax optax onnx onnxruntime

# GPU: requires cuDNN. On Arch Linux: sudo pacman -S cudnn
# Then: .venv/bin/pip install "jax[cuda12]"
# Without cuDNN, training runs on CPU (slower but functional).
.venv/bin/pip install "jax[cuda12_local]" 2>/dev/null || .venv/bin/pip install jax

echo "=== Cloning mujoco_menagerie (Go2 model) ==="
if [ ! -d "$HOME/mujoco_menagerie" ]; then
  git clone --depth=1 https://github.com/google-deepmind/mujoco_menagerie "$HOME/mujoco_menagerie"
fi

echo "=== Verifying model loads ==="
.venv/bin/python3 - <<'EOF'
import mujoco, pathlib
xml = pathlib.Path.home() / "mujoco_menagerie/unitree_go2/go2_mjx.xml"
m = mujoco.MjModel.from_xml_path(str(xml))
print(f"Go2 model loaded: nq={m.nq}, nu={m.nu}, nv={m.nv}")
EOF

echo ""
echo "Setup complete. Run training with:"
echo "  .venv/bin/python3 train.py --n_envs 4096 --steps 50000000"
