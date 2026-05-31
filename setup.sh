#!/usr/bin/env bash
# One-shot setup for the Go2 sim-to-real project.
# Run once from the project root: bash setup.sh

set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

echo "=== Creating venv ==="
python3 -m venv .venv

echo "=== Installing Python dependencies ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install mujoco mujoco-mjx flax optax onnx onnxruntime

# jax[cuda12] bundles cuDNN wheels — no system cuDNN install needed
# Falls back to CPU-only jax if CUDA is unavailable
.venv/bin/pip install "jax[cuda12]" 2>/dev/null || .venv/bin/pip install jax

echo "=== Cloning mujoco_menagerie (Go2 model) ==="
# Clone next to the project directory so it works regardless of where repo is cloned
MENAGERIE="$(dirname "$PROJECT_DIR")/mujoco_menagerie"
if [ ! -d "$MENAGERIE/unitree_go2" ]; then
  git clone --depth=1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
  cd "$MENAGERIE" && git sparse-checkout set unitree_go2 && cd "$PROJECT_DIR"
fi
echo "  -> model at $MENAGERIE/unitree_go2/go2_mjx.xml"

echo "=== Verifying model loads ==="
.venv/bin/python3 - <<EOF
import mujoco, pathlib
xml = pathlib.Path("$MENAGERIE") / "unitree_go2/go2_mjx.xml"
m = mujoco.MjModel.from_xml_path(str(xml))
print(f"Go2 model loaded: nq={m.nq}, nu={m.nu}, nv={m.nv}")
EOF

echo ""
echo "Setup complete. Run training with:"
echo "  .venv/bin/python3 train.py --n_envs 4096 --steps 50000000"
echo ""
echo "Watch the policy in the viewer:"
echo "  .venv/bin/python3 watch.py --checkpoint checkpoints/policy_final.pkl"
