# Gemma 3-1B-PT ExecuTorch Export Summary

## Status

Export and baseline evaluation completed successfully.

## Model

- Model: Gemma 3-1B-PT
- Source: Local Hugging Face snapshot
- Local path: `/home/timothy_riffe/Documents/projects/llm/models/gemma-3-1b-pt`
- Model type: Base pretrained causal language model, not instruction-tuned

## ExecuTorch Artifact

- Output directory: `output/models/gemma_3_1b_pt_optimum`
- Artifact type: `.pte`
- Runtime target: Portable CPU FP32
- Artifact size: `3814.83 MB`
- XNNPACK: Deferred due installed ExecuTorch API mismatch

## Export Method

- Loaded model with `AutoModelForCausalLM`
- Disabled KV cache with `model.config.use_cache = False`
- Exported simplified forward pass using `torch.export.export`
- Used `strict=False`
- Removed unsupported `partitioners` argument
- Lowered to ExecuTorch Edge program
- Emitted portable CPU `.pte`

## PyTorch Baseline Evaluation

- Script: `evaluation/gemma_3_1b_pt/evaluate_pytorch.py`
- Runtime: PyTorch FP32
- Prompt: `The reason the sky appears blue is`
- Max new tokens: `200`
- Sampling enabled: `true`
- Temperature: `0.7`
- Top-p: `0.95`
- Top-k: `50`
- Repetition penalty: `1.15`
- No repeat n-gram size: `4`
- Seed: `42`
- KV cache: disabled
- Latency: `99.0753 seconds`

## Issues Resolved

- Fixed malformed JSON config.
- Corrected local model path by adding leading `/`.
- Verified tokenizer/model loading from local snapshot.
- Disabled `DynamicCache` / KV cache for `torch.export`.
- Used `strict=False` to support export.
- Removed unsupported `partitioners` keyword from `to_edge_transform_and_lower`.
- Improved PyTorch baseline generation to avoid prompt repetition.

## Deliverables Completed

- Local model load test.
- PyTorch inference baseline.
- Portable FP32 `.pte` export.
- Artifact smoke test.
- Artifact metadata report.
- Export summary report.

## Deferred / Next Work

- ExecuTorch runtime inference validation.
- XNNPACK partitioner integration.
- INT8 / INT4 quantization.
- ETDump / Inspector profiling.
- Comparative latency table: PyTorch FP32 vs ExecuTorch portable FP32 vs quantized/XNNPACK variants.
