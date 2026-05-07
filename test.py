python3 - <<'PY'
import tensorflow as tf
from pathlib import Path

src = "/tmp/ssd_mobilenet_v2_coco_2018_03_29/tflite_graph.pb"
out = Path("weights/model.tflite")
out.parent.mkdir(parents=True, exist_ok=True)

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file=src,
    input_arrays=["normalized_input_image_tensor"],
    output_arrays=[
        "TFLite_Detection_PostProcess",
        "TFLite_Detection_PostProcess:1",
        "TFLite_Detection_PostProcess:2",
        "TFLite_Detection_PostProcess:3",
    ],
    input_shapes={"normalized_input_image_tensor": [1, 300, 300, 3]},
)
converter.allow_custom_ops = True
# NO converter.optimizations -> stays FP32

tflite_model = converter.convert()
out.write_bytes(tflite_model)

# Verify it's FP32
import numpy as np
interp = tf.lite.Interpreter(model_path=str(out))
interp.allocate_tensors()
in_d = interp.get_input_details()[0]
print(f"Saved: {out} ({len(tflite_model)/1024/1024:.1f} MB)")
print(f"Input dtype: {np.dtype(in_d['dtype']).name}  shape: {in_d['shape'].tolist()}")
print(f"Quantization: {in_d.get('quantization', 'none')}")
PY
