# FROM: ~/Documents/projects/MetaExecuTorch/executorch-toolkit

# =========================================
# REPLACE config_mobile_net_v2_ssd.json
# =========================================

cat > export/configs/vision/config_mobile_net_v2_ssd.json << 'EOF'
{
  "__comment__": "COCO evaluation config for MobileNetV2 SSD with PyTorch + TFLite + ExecuTorch comparison",

  "model": {
    "model_path": "~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth",
    "model_name": "mobile_net_v2_ssd",
    "input_shape": [1, 3, 300, 300],

    "use_pretrained": true,
    "normalize": true,
    "normalize_mean": [0.498, 0.498, 0.498],
    "normalize_std": [0.502, 0.502, 0.502],

    "num_classes": 21,

    "source_path": "~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src/pytorch",
    "model_sources_repo_path": "~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/src"
  },

  "tflite_parity": {
    "__comment__": "Optional parity validation only (NOT evaluation baseline)",
    "enabled": false,
    "tflite_source_path": "~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mobile_net_v2_ssd.tflite",
    "sample_image_path": "../dataset/samples/voc2012/parity_sample.jpg",
    "report_path": "output/models/mobile_net_v2_ssd/tflite_parity_report.json",
    "strict": true,
    "max_abs_tolerance": 0.01,
    "cosine_tolerance": 0.999
  },

  "export": {
    "enabled": true,
    "is_base": true,
    "output_dir": "~/Documents/projects/MetaExecuTorch/executorch-toolkit/output/models/mobile_net_v2_ssd",
    "output_name": "mobile_net_v2_ssd_executorch",
    "dump_ptd": true,
    "backend": "xnnpack",
    "strict_export": false,
    "save_metadata": true,
    "benchmark": true,
    "benchmark_iterations": 50,
    "use_recipes": false,
    "log_level": "INFO",
    "enable_profiling": true,
    "etdump_save_debug_buffer": false,
    "etdump_debug_buffer_size": 200000000,
    "calibration_dir": "../dataset/samples/voc2012",
    "save_pytorch_model": true
  },

  "quantization": [
    {
      "enabled": true,
      "output_name": "mobile_net_v2_ssd_executorch_8a8w_pt",
      "mode": "static",
      "weight_dtype": "int8",
      "activation_dtype": "int8",
      "per_channel": true,
      "calibration_samples": 25,
      "calibration_dir": "../dataset/samples/voc2012"
    },
    {
      "enabled": true,
      "output_name": "mobile_net_v2_ssd_executorch_8a8w_pc",
      "mode": "static",
      "weight_dtype": "int8",
      "activation_dtype": "int8",
      "per_channel": true,
      "calibration_samples": 25,
      "calibration_dir": "../dataset/samples/voc2012"
    },
    {
      "enabled": true,
      "output_name": "mobile_net_v2_ssd_executorch_8da4w",
      "mode": "dynamic",
      "weight_dtype": "int4",
      "activation_dtype": "int8",
      "per_channel": true,
      "calibration_samples": 25,
      "calibration_dir": "../dataset/samples/voc2012"
    }
  ],

  "evaluation": {
    "enabled": true,
    "task_type": "detection",
    "primary_metric": "mAP_0.5_0.95",

    "dataset": {
      "data_path": "~/Documents/projects/MetaExecuTorch/dataset/coco_val2017/val2017",
      "gt_coco_json": "~/Documents/projects/MetaExecuTorch/dataset/coco_val2017/annotations/instances_val2017.json",
      "split": "val",
      "batch_size": 1,
      "num_workers": 4,
      "max_samples": null,
      "class_subset": null,

      "class_names": [
        "BACKGROUND",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
      ]
    },

    "decoder": {
      "image_size": 300,
      "score_threshold": 0.01,
      "nms_iou_threshold": 0.45,
      "max_detections_per_image": 100,
      "candidate_size": 200,
      "center_variance": 0.1,
      "size_variance": 0.2
    },

    "paths": {
      "results_dir": "~/Documents/projects/MetaExecuTorch/output/results/mobile_net_v2_ssd_coco"
    },

    "output": {
      "save_per_image": true
    },

    "report": {
      "save_predictions": true,
      "save_consolidated_json": true,
      "consolidated_json_name": "mobile_net_v2_ssd_coco_evaluation.json",
      "generate_html": true,
      "html_name": "MobileNetV2_SSD_Lite_COCO_Evaluation_Report.html",
      "include_per_image_table": true,
      "include_metric_interpretations": true,
      "include_system_info": true
    },

    "backends": {
      "include_tflite_baseline": true,
      "include_pytorch_baseline": true,

      "tflite_baseline": {
        "name": "tflite_baseline",
        "model_path": "~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mobile_net_v2_ssd.tflite",
        "score_threshold": 0.01
      },

      "executorch_models": [
        {
          "name": "executorch_fp32",
          "pte_path": "~/Documents/projects/MetaExecuTorch/executorch-toolkit/output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch.pte",
          "is_baseline": true
        },
        {
          "name": "executorch_int8_8a8w_pt",
          "pte_path": "~/Documents/projects/MetaExecuTorch/executorch-toolkit/output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8a8w_pt.pte"
        },
        {
          "name": "executorch_int8_8a8w_pc",
          "pte_path": "~/Documents/projects/MetaExecuTorch/executorch-toolkit/output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8a8w_pc.pte"
        },
        {
          "name": "executorch_int4_8da4w",
          "pte_path": "~/Documents/projects/MetaExecuTorch/executorch-toolkit/output/models/mobile_net_v2_ssd/mobile_net_v2_ssd_executorch_8da4w.pte"
        }
      ]
    }
  }
}
EOF

# =========================================
# RUN EVALUATION
# =========================================

python export/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
