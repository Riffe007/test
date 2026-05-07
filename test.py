cd ~/Documents/projects/MetaExecuTorch

cat > dataset/scripts/download/coco.py <<'PY'
#!/usr/bin/env python3
"""
Download and prepare the COCO 2017 validation dataset for object detection.

Usage:
    python dataset/scripts/download/coco.py \
        --output-dir dataset/samples/coco/ \
        --max-samples 100

Idempotent: re-running skips files already present.
Layout produced:
    <output-dir>/
        annotations/instances_val2017.json
        val2017/<image_id>.jpg   (first --max-samples images, sorted by id)
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/{filename}"


def download(output_dir: Path, max_samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "val2017"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotations (extract just instances_val2017.json from the trainval zip)
    annotations_file = annotations_dir / "instances_val2017.json"
    if annotations_file.exists():
        print(f"Annotations cached: {annotations_file}")
    else:
        zip_path = output_dir / "annotations_trainval2017.zip"
        print(f"Downloading annotations zip (~241 MB) from {COCO_ANNOTATIONS_URL} ...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        print("Extracting instances_val2017.json ...")
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.endswith("instances_val2017.json"):
                    z.extract(member, output_dir)
                    break
        zip_path.unlink()
        print(f"  -> {annotations_file}")

    # 2. Sample images (deterministic order: sorted by COCO image id)
    with open(annotations_file) as f:
        coco = json.load(f)
    images = sorted(coco["images"], key=lambda x: x["id"])[:max_samples]

    n_new, n_cached = 0, 0
    for i, meta in enumerate(images):
        filename = meta["file_name"]
        out_path = images_dir / filename
        if out_path.exists():
            n_cached += 1
            continue
        urllib.request.urlretrieve(COCO_IMAGE_URL.format(filename=filename), out_path)
        n_new += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] {filename}")

    print(f"\nDone. {n_new} new, {n_cached} cached. {len(images)} samples in {images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("dataset/samples/coco"))
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    download(args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
PY

chmod +x dataset/scripts/download/coco.py

# Run with defaults — output to dataset/samples/coco/, 100 samples
python3 dataset/scripts/download/coco.py
