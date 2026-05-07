cd /tmp
rm -rf ssd_mobilenet_v2_coco_2018_03_29  # FP32 tarball doesn't have what we need

[ -f ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz ] || wget -q --show-progress \
  http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
tar xzf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

# Confirm tflite_graph.pb is there
ls -lah /tmp/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb

cd ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2

python3 - <<'PY' 2>&1 | tee /tmp/convert_output.log
import tensorflow as tf
import numpy as np
from pathlib import Path

src = "/tmp/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb"
out = Path("weights/model.tflite")
assert Path(src).exists(), f"Missing: {src}"
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
converter.inference_type = tf.float32          # explicit FP32
converter.inference_input_type = tf.float32    # explicit FP32 input
# NO converter.optimizations -> FakeQuant nodes are stripped, weights stay FP32

tflite_model = converter.convert()
out.write_bytes(tflite_model)

interp = tf.lite.Interpreter(model_path=str(out))
interp.allocate_tensors()
in_d = interp.get_input_details()[0]
print(f"\nSaved:        {out}  ({len(tflite_model)/1024/1024:.1f} MB)")
print(f"Input dtype:  {np.dtype(in_d['dtype']).name}")
print(f"Input shape:  {in_d['shape'].tolist()}")
print(f"Quant params: {in_d['quantization']}")
PY

ls -lah weights/
