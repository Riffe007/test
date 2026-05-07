"""
TFLite -> Clean FP32 ONNX pipeline.

Order matters:
  1. tf2onnx convert
  2. force int graph inputs to FP32
  3. onnxsim (folds INT8_weight + DQ into FP32 initializers)
  4. Strip remaining Q/DQ (now activation-only)
  5. onnxsim again (cleans up identity chains)
  6. checker + numerical parity vs source TFLite (hard fail on drift)
"""

from pathlib import Path
import logging
import numpy as np
import onnx
from onnx import TensorProto
from onnxsim import simplify
import onnxruntime as ort
import tf2onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("tflite_to_fp32_onnx")

ROOT = Path.home() / "Documents/projects/MetaExecuTorch"
WEIGHTS = ROOT / "model_sources/MobileNetV2/weights"
TFLITE = WEIGHTS / "model.tflite"
ONNX_RAW = WEIGHTS / "model.raw.onnx"
ONNX_SIMP1 = WEIGHTS / "model.simp1.onnx"
ONNX_STRIP = WEIGHTS / "model.strip.onnx"
ONNX_CLEAN = WEIGHTS / "model.fp32.onnx"

OPSET = 17
PARITY_TOLERANCE = 1e-2  # SSD scores are forgiving; tighten for classification
PARITY_SEED = 123

INT_TYPES = {
    TensorProto.UINT8, TensorProto.INT8,
    TensorProto.UINT16, TensorProto.INT16,
    TensorProto.INT32,
}


# --------------------------------------------------------------------------
def convert_tflite_to_onnx():
    log.info("Stage 1: tf2onnx convert")
    tf2onnx.convert.from_tflite(
        str(TFLITE),
        output_path=str(ONNX_RAW),
        opset=OPSET,
    )
    log.info("  -> %s", ONNX_RAW)


def force_fp32_inputs(m):
    for inp in m.graph.input:
        old = inp.type.tensor_type.elem_type
        if old in INT_TYPES:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            log.info("  forced input %s: %d -> FLOAT", inp.name, old)
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
    """Bypass remaining Q/DQ nodes. Run AFTER onnxsim so weight-side
    DQs are already folded into FP32 initializers; this pass should
    only touch activation Q/DQ."""
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
            log.warning(
                "  graph output %s rewired to upstream %s",
                out.name, new_name,
            )
            out.name = new_name

    for n in nodes_to_remove:
        m.graph.node.remove(n)

    log.info("  stripped %d Q/DQ nodes", len(nodes_to_remove))
    return m


def parity_check(onnx_path):
    """Compare TFLite vs cleaned ONNX on a deterministic seeded input.
    Hard-fails if any output exceeds PARITY_TOLERANCE - this catches
    silent semantic drift from the dropped TFL_edgetpu-custom-op or
    any other op tf2onnx couldn't translate."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            log.warning("No TFLite runtime - skipping parity check")
            return

    log.info("Parity check: TFLite vs ONNX (seed=%d)", PARITY_SEED)
    interp = Interpreter(model_path=str(TFLITE))
    interp.allocate_tensors()
    in_detail = interp.get_input_details()[0]
    out_details = interp.get_output_details()

    rng = np.random.default_rng(PARITY_SEED)
    shape = in_detail["shape"]
    if in_detail["dtype"] == np.uint8:
        x_tflite = rng.integers(0, 256, shape, dtype=np.uint8)
        scale, zp = in_detail.get("quantization", (0.0, 0))
        # ONNX path expects pre-dequantized FP32 since input DQ was stripped
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
        # Dequantize TFLite output if quantized
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
    convert_tflite_to_onnx()

    m = onnx.load(str(ONNX_RAW))
    m = force_fp32_inputs(m)
    onnx.save(m, str(ONNX_RAW))

    # Critical ordering: simplify FIRST so weight DQs fold into FP32
    # initializers. Stripping before this can leave INT8 weights feeding
    # into FP32 ops and fail onnx.checker.
    m = simplify_model(m, "post-fp32-input")
    onnx.save(m, str(ONNX_SIMP1))

    m = strip_quant_nodes(m)
    onnx.save(m, str(ONNX_STRIP))

    # Second simplify cleans identity chains left by the strip
    m = simplify_model(m, "post-strip")

    onnx.checker.check_model(m)
    onnx.save(m, str(ONNX_CLEAN))
    log.info("Saved clean FP32 ONNX: %s", ONNX_CLEAN)

    log.info("Final graph outputs:")
    for o in m.graph.output:
        log.info("  %s", o.name)

    parity_check(ONNX_CLEAN)


if __name__ == "__main__":
    main()
