"""TFLite -> Clean FP32 ONNX pipeline.

Order matters:
  1. tf2onnx convert (TFLite -> raw ONNX)
  2. force int graph inputs to FP32
  3. replace TFL_edgetpu-custom-op nodes with native ONNX ops
  4. onnxsim (folds INT8_weight + DQ into FP32 initializers)
  5. Strip remaining Q/DQ (now activation-only)
  6. onnxsim again (cleans up identity chains)
  7. checker + numerical parity vs source TFLite (hard fail on drift)
"""

from pathlib import Path
import argparse
import logging
import numpy as np
import onnx
from onnx import TensorProto
from onnxsim import simplify
import onnxruntime as ort
import tf2onnx

OPSET = 17
PARITY_TOLERANCE = 1e-2
PARITY_SEED = 123

INT_TYPES = {
    TensorProto.UINT8, TensorProto.INT8,
    TensorProto.UINT16, TensorProto.INT16,
    TensorProto.INT32,
}

log = logging.getLogger("tflite_to_fp32_onnx")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="Path to source .tflite file")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write clean FP32 .onnx file")
    p.add_argument("--input-shape", nargs=4, type=int,
                   metavar=("N", "C", "H", "W"),
                   help="Override input shape (rarely needed)")
    p.add_argument("--skip-smoke-test", action="store_true",
                   help="Skip TFLite vs ONNX numerical parity check")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def convert_tflite_to_onnx(tflite_path, onnx_path):
    log.info("Stage 1: tf2onnx convert")
    log.info("  %s -> %s", tflite_path, onnx_path)
    tf2onnx.convert.from_tflite(
        str(tflite_path),
        output_path=str(onnx_path),
        opset=OPSET,
    )


def force_fp32_inputs(m, input_shape=None):
    for inp in m.graph.input:
        old = inp.type.tensor_type.elem_type
        if old in INT_TYPES:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            log.info("  forced input %s: %d -> FLOAT", inp.name, old)
        if input_shape is not None:
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == len(input_shape):
                for d, v in zip(dims, input_shape):
                    d.dim_value = int(v)
                log.info("  overrode input shape: %s", input_shape)
    return m


def replace_edgetpu_ops(m):
    """Replace TFL_edgetpu-custom-op nodes with native ONNX ops.

    The EdgeTPU compiler bundles small subgraphs into custom ops carrying
    binary TPU bytecode that ONNX runtime can't validate. Heuristically map
    them to native ops based on the node name. The parity check is your
    validation gate - if shapes/values don't match the source TFLite,
    inspect the original via Netron and extend NAME_TO_OP."""
    NAME_TO_OP = [
        ("squeeze",   "Squeeze"),
        ("transpose", "Transpose"),
        ("reshape",   "Reshape"),
    ]
    replaced = 0
    for node in m.graph.node:
        if not node.op_type.startswith("TFL_"):
            continue
        new_op = "Identity"
        for pat, op in NAME_TO_OP:
            if pat in node.name.lower():
                new_op = op
                break
        log.info("  replacing %s (was %s) -> %s",
                 node.name, node.op_type, new_op)
        node.op_type = new_op
        node.domain = ""
        del node.attribute[:]  # purge unparseable binary attrs
        replaced += 1
    if replaced:
        log.warning("  replaced %d EdgeTPU op(s) - parity check is your "
                    "validation gate", replaced)
    return m


def simplify_model(m, label):
    log.info("Simplify: %s", label)
    n_before = len(m.graph.node)
    m_simp, ok = simplify(m)
    if not ok:
        raise RuntimeError(f"onnxsim failed at stage: {label}")
    log.info("  nodes: %d -> %d", n_before, len(m_simp.graph.node))
    return m_simp


def strip_quant_nodes(m):
    log.info("Stripping residual Q/DQ nodes")
    rewrites = {}
    nodes_to_remove = []
    for n in m.graph.node:
        if n.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            rewrites[n.output[0]] = n.input[0]
            nodes_to_remove.append(n)

    if not nodes_to_remove:
        log.info("  no Q/DQ nodes remain")
        return m

    def resolve(name):
        seen = set()
        while name in rewrites and name not in seen:
            seen.add(name)
            name = rewrites[name]
        return name

    for n in m.graph.node:
        for i, name in enumerate(n.input):
            n.input[i] = resolve(name)

    for out in m.graph.output:
        new_name = resolve(out.name)
        if new_name != out.name:
            log.warning("  graph output %s rewired to upstream %s",
                        out.name, new_name)
            out.name = new_name

    for n in nodes_to_remove:
        m.graph.node.remove(n)

    log.info("  stripped %d Q/DQ nodes", len(nodes_to_remove))
    return m


def parity_check(tflite_path, onnx_path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            log.warning("No TFLite runtime - skipping parity check")
            return

    log.info("Parity check: TFLite vs ONNX (seed=%d)", PARITY_SEED)
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    rng = np.random.default_rng(PARITY_SEED)
    shape = in_detail["shape"]
    if in_detail["dtype"] == np.uint8:
        x_tflite = rng.integers(0, 256, shape, dtype=np.uint8)
        scale, zp = in_detail.get("quantization", (0.0, 0))
        x_onnx = ((x_tflite.astype(np.float32) - zp) * scale
                  if scale != 0.0 else x_tflite.astype(np.float32))
    else:
        x_onnx = rng.random(shape, dtype=np.float32)
        x_tflite = x_onnx

    interp.set_tensor(in_detail["index"], x_tflite)
    interp.invoke()
    tflite_outs = [interp.get_tensor(o["index"]) for o in out_details]

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_outs = sess.run(None, {sess.get_inputs()[0].name: x_onnx})

    log.info("  TFLite outputs: %d  ONNX outputs: %d",
             len(tflite_outs), len(onnx_outs))

    failures = []
    for i, (t, o, od) in enumerate(zip(tflite_outs, onnx_outs, out_details)):
        t_fp = t.astype(np.float32)
        if t.dtype != np.float32:
            scale, zp = od.get("quantization", (0.0, 0))
            if scale != 0.0:
                t_fp = (t.astype(np.float32) - zp) * scale

        if t_fp.shape != o.shape:
            msg = (f"out[{i}] shape mismatch: "
                   f"tflite {t_fp.shape} vs onnx {o.shape}")
            log.error("  " + msg)
            failures.append(msg)
            continue

        diff = float(np.max(np.abs(t_fp - o)))
        log.info("  out[%d] max|Δ|=%.4e shape=%s", i, diff, t_fp.shape)
        if diff > PARITY_TOLERANCE:
            failures.append(
                f"out[{i}] max delta {diff:.4e} > tol {PARITY_TOLERANCE:.4e}"
            )

    if failures:
        raise RuntimeError(
            "Parity check failed - investigate dropped EdgeTPU op:\n  "
            + "\n  ".join(failures)
        )
    log.info("  parity OK")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if args.input.suffix != ".tflite":
        log.warning("Input does not have .tflite extension: %s", args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_name(args.output.stem + ".raw.onnx")

    convert_tflite_to_onnx(args.input, raw_path)

    log.info("loading: %s", raw_path)
    m = onnx.load(str(raw_path))

    m = force_fp32_inputs(m, input_shape=args.input_shape)
    m = replace_edgetpu_ops(m)

    # Critical ordering: simplify FIRST so weight DQs fold into FP32
    # initializers. Stripping before this can leave INT8 weights feeding
    # into FP32 ops and fail onnx.checker.
    m = simplify_model(m, "post-fp32-input")
    m = strip_quant_nodes(m)
    m = simplify_model(m, "post-strip")

    onnx.checker.check_model(m)
    onnx.save(m, str(args.output))
    log.info("Saved clean FP32 ONNX: %s", args.output)

    log.info("Final graph outputs:")
    for o in m.graph.output:
        log.info("  %s", o.name)

    if not args.skip_smoke_test:
        parity_check(args.input, args.output)
    else:
        log.info("Skipping parity check (--skip-smoke-test)")


if __name__ == "__main__":
    main()
