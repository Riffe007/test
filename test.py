import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import EdgeCompileConfig

from torchao.quantization import quantize_, Int8WeightOnlyConfig


def main():
    config_path = Path("export/configs/llm/config_gemma_3_1b_pt_optimum.json")

    with config_path.open() as f:
        cfg = json.load(f)

    model_id = cfg["model_id"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).eval()

    # Required for torch.export compatibility.
    model.config.use_cache = False

    print("Applying TorchAO INT8 weight-only quantization...")

    quantize_(
        model,
        Int8WeightOnlyConfig(),
    )

    prompt = "The reason the sky appears blue is"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("Exporting INT8 quantized forward pass...")

    ep = torch.export.export(
        model,
        args=(input_ids, attention_mask),
        strict=False,
    )

    print("Lowering INT8 model to Edge + XNNPACK...")

    edge = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=True,
        ),
    )

    print("Creating INT8 XNNPACK .pte file...")

    et_program = edge.to_executorch()

    output_file = output_dir / "gemma_3_1b_pt_xnnpack_int8_weight_only.pte"

    with open(output_file, "wb") as f:
        f.write(et_program.buffer)

    print(f"SUCCESS: {output_file}")


if __name__ == "__main__":
    main()
