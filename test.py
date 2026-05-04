from pathlib import Path

files = [
    "output/models/gemma_3_1b_pt_optimum/gemma_3_1b_pt_fp32.pte",
    "output/models/gemma_3_1b_pt_optimum/gemma_3_1b_pt_xnnpack_fp32.pte",
    "output/models/gemma_3_1b_pt_optimum/gemma_3_1b_pt_xnnpack_int8_weight_only.pte",
]

print("\nArtifact Comparison Report")
print("=" * 60)

for f in files:
    path = Path(f)

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{path.name}")
        print(f"  Size: {size_mb:.2f} MB")
        print("-" * 60)
    else:
        print(f"{f} -> NOT FOUND")
