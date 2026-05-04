import torch

from transformers import AutoModelForCausalLM

from torchao.quantization import quantize_, Int4WeightOnlyConfig

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

from executorch.exir import to_edge_transform_and_lower


MODEL_ID = "/home/timothy_riffe/Documents/projects/llm/models/gemma-3-1b-pt"

OUTPUT_FILE = (
    "output/models/gemma_3_1b_pt_optimum/"
    "gemma_3_1b_pt_xnnpack_int4_weight_only.pte"
)


def main():
    print(f"Loading model from: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        local_files_only=True,
    )

    model.eval()
    model.config.use_cache = False

    print("Applying INT4 weight-only quantization...")

    quantize_(
        model,
        Int4WeightOnlyConfig(),
    )

    example_inputs = (
        torch.randint(
            0,
            100,
            (1, 16),
            dtype=torch.long,
        ),
    )

    print("Exporting simplified forward pass...")

    exported_program = torch.export.export(
        model,
        example_inputs,
        strict=False,
    )

    print("Lowering to Edge + XNNPACK...")

    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    print("Creating INT4 XNNPACK .pte file...")

    with open(OUTPUT_FILE, "wb") as f:
        edge_program.to_executorch().write_to_file(f)

    print(f"SUCCESS: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
