from pathlib import Path

MODEL_DIR = Path("output/models/gemma_3_1b_pt_optimum")

pte_files = list(MODEL_DIR.glob("*.pte"))

if not pte_files:
    raise FileNotFoundError(f"No .pte files found in {MODEL_DIR}")

print("ExecuTorch Runtime Validation")
print("-" * 40)

for file in pte_files:
    size_mb = round(file.stat().st_size / (1024 * 1024), 2)

    print(f"Artifact Found: {file.name}")
    print(f"Path: {file}")
    print(f"Size: {size_mb} MB")

    if size_mb < 100:
        print("WARNING: suspiciously small artifact")
    else:
        print("PASS: artifact size looks valid")

print("-" * 40)
print("Portable .pte validation complete.")
print("Next step: runtime execution on target backend.")
