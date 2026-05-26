"""
Export a trained policy checkpoint to ONNX for deployment.

Usage:
    python export_policy.py --checkpoint checkpoints/policy_final.pkl \
                            --output checkpoints/policy.onnx
"""
import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort

# pytorch-free ONNX export via jax2torch or direct numpy tracing
# We use a simpler approach: serialize params + a numpy forward pass
# via flax's apply in a thin ONNX-compatible wrapper.


def export_to_onnx(params, output_path: str):
    """Export policy to ONNX using jax2tf + tf2onnx."""
    try:
        import tensorflow as tf
        import tf2onnx
        import jax.experimental.jax2tf as jax2tf
        from policy import ActorCritic
    except ImportError:
        print("Install: pip install tensorflow tf2onnx jax[cuda12_pip]")
        raise

    model = ActorCritic()

    def predict(obs):
        action_mean, log_std, _ = model.apply(params, obs)
        return action_mean  # deterministic deployment: use mean

    tf_predict = tf.function(
        jax2tf.convert(predict, enable_xla=False),
        input_signature=[tf.TensorSpec(shape=(45,), dtype=tf.float32)],
        autograph=False,
    )

    input_spec = (tf.TensorSpec(shape=(45,), dtype=tf.float32, name="obs"),)
    model_proto, _ = tf2onnx.convert.from_function(
        tf_predict,
        input_signature=input_spec,
        opset=17,
        output_path=output_path,
    )
    print(f"Exported ONNX model to {output_path}")
    _verify_onnx(output_path)


def _verify_onnx(path: str):
    sess = ort.InferenceSession(path)
    dummy = np.zeros((45,), dtype=np.float32)
    out = sess.run(None, {"obs": dummy})
    print(f"  ONNX verification passed — output shape: {out[0].shape}")


def export_pickle_only(params, output_path: str):
    """Fallback: re-save params as a plain pickle for the deploy script."""
    with open(output_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Saved raw params to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/policy.onnx")
    parser.add_argument("--format", choices=["onnx", "pickle"], default="onnx")
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if isinstance(ckpt, dict) else ckpt
    print(f"Loaded checkpoint (step={ckpt.get('step', '?')})")

    if args.format == "onnx":
        export_to_onnx(params, args.output)
    else:
        export_pickle_only(params, args.output.replace(".onnx", ".pkl"))


if __name__ == "__main__":
    main()
