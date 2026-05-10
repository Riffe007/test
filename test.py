#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch a deterministic subset of COCO val2017 from HuggingFace mirrors.

This script exists because cocodataset.org HTTPS is blocked in some corporate
network environments while HuggingFace remains reachable. It downloads:

    1. The canonical ``instances_val2017.json`` annotations file from the
       ``merve/coco`` dataset (a verbatim mirror of the original COCO release).
    2. ``num_images`` validation images from the ``rafaelpadilla/coco2017``
       dataset, written to disk in the canonical zero-padded 12-digit naming
       convention (e.g. ``000000000139.jpg``) so downstream tools that expect
       the standard cocodataset.org layout work without modification.

The resulting on-disk layout matches what an unpacked ``val2017.zip`` plus
``annotations_trainval2017.zip`` would produce::

    <out-dir>/
        annotations/instances_val2017.json
        val2017/000000000139.jpg
        val2017/000000000285.jpg
        ...

The script is idempotent: re-running it with the same arguments skips files
that already exist on disk, so a Ctrl+C mid-download is recoverable by simply
re-invoking the same command.

Author: Persistent Systems
Date:   May 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNOTATIONS_URL: Final[str] = (
    "https://huggingface.co/datasets/merve/coco/resolve/main/"
    "annotations/instances_val2017.json"
)
"""Canonical instances_val2017.json hosted on HuggingFace (LFS-backed, ~20 MB).

This is a verbatim copy of the file that cocodataset.org serves under
``annotations_trainval2017.zip``; the byte content is identical.
"""

IMAGES_DATASET_REPO: Final[str] = "rafaelpadilla/coco2017"
"""HuggingFace datasets repo that bundles COCO 2017 val images as parquet shards.

The dataset's row schema is ``{"image": PIL.Image, "image_id": int, "objects": ...}``.
``image_id`` matches the canonical COCO image_id used in the annotations file.
"""

IMAGES_SPLIT: Final[str] = "val"
"""Split to pull from ``rafaelpadilla/coco2017``: 5000 val2017 images."""

EXPECTED_ANNOTATIONS_MIN_BYTES: Final[int] = 5_000_000
"""Floor for the annotations file size; anything smaller is a corrupted download.

The real file is ~20 MB; we use a generous 5 MB floor to detect HTML error
pages, partial transfers, or empty stubs.
"""

PROGRESS_LOG_INTERVAL: Final[int] = 50
"""Log a progress line every N images saved (keeps logs readable for 1000+ pulls)."""

JPEG_QUALITY: Final[int] = 95
"""JPEG quality for re-encoding. The source images are already lossy JPEGs;
re-encoding at 95 is visually indistinguishable while keeping files small.
"""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: Final[logging.Logger] = logging.getLogger("fetch_coco_val_subset")


def _setup_logging(verbosity: int) -> None:
    """Configure root logging to match the toolkit's evaluator format."""
    level = logging.DEBUG if verbosity >= 2 else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Annotations: single-file download
# ---------------------------------------------------------------------------


def _annotations_already_valid(path: Path) -> bool:
    """Return ``True`` iff *path* exists and looks like a real annotations file."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < EXPECTED_ANNOTATIONS_MIN_BYTES:
        logger.warning(
            "Existing annotations file at %s is suspiciously small (%d bytes); "
            "will redownload",
            path,
            size,
        )
        return False
    return True


def download_annotations(out_path: Path) -> None:
    """Fetch ``instances_val2017.json`` to *out_path*, skipping if already present.

    Raises:
        RuntimeError: if the download fails or produces a file below the
            ``EXPECTED_ANNOTATIONS_MIN_BYTES`` floor.
    """
    if _annotations_already_valid(out_path):
        size_mb = out_path.stat().st_size / (1024 ** 2)
        logger.info("Annotations already present (%.1f MB), skipping: %s", size_mb, out_path)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading annotations: %s", ANNOTATIONS_URL)

    # Imported lazily so the script's --help works even without `requests` installed.
    import requests

    try:
        with requests.get(ANNOTATIONS_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(out_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to download annotations from {ANNOTATIONS_URL}") from exc

    final_size = out_path.stat().st_size
    if final_size < EXPECTED_ANNOTATIONS_MIN_BYTES:
        raise RuntimeError(
            f"Downloaded annotations file is too small ({final_size} bytes); "
            f"expected >= {EXPECTED_ANNOTATIONS_MIN_BYTES}. The mirror may have "
            f"served an error page or the transfer was truncated."
        )

    logger.info("Wrote annotations: %s (%.1f MB)", out_path, final_size / (1024 ** 2))


# ---------------------------------------------------------------------------
# Images: streaming HF dataset, write to canonical naming
# ---------------------------------------------------------------------------


def _save_pil_image(image: object, dest: Path) -> None:
    """Save a PIL image to *dest* as JPEG, converting non-RGB modes."""
    # Local import to keep the global dependency surface explicit.
    from PIL import Image  # type: ignore[import-not-found]

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)!r}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    tmp = dest.with_suffix(dest.suffix + ".part")
    image.save(tmp, format="JPEG", quality=JPEG_QUALITY)
    tmp.replace(dest)


def download_images(out_dir: Path, num_images: int) -> int:
    """Stream up to *num_images* val images from HF and save with canonical naming.

    Returns the number of images successfully written (including ones that
    already existed from a prior partial run).
    """
    if num_images <= 0:
        raise ValueError(f"num_images must be positive, got {num_images}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so missing-dep errors are localized and informative.
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install with:\n"
            "    pip install datasets"
        ) from exc

    logger.info(
        "Streaming %s split=%s (target %d images, output: %s)",
        IMAGES_DATASET_REPO,
        IMAGES_SPLIT,
        num_images,
        out_dir,
    )
    dataset = load_dataset(IMAGES_DATASET_REPO, split=IMAGES_SPLIT, streaming=True)

    saved = 0
    skipped = 0
    for row in dataset:
        if saved + skipped >= num_images:
            break

        try:
            image_id = int(row["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed row (image_id missing/invalid): %s", exc)
            continue

        dest = out_dir / f"{image_id:012d}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            saved += 1  # count toward target either way; the file is on disk
            continue

        try:
            _save_pil_image(row["image"], dest)
        except Exception as exc:
            logger.warning("Failed to save image_id=%d: %s", image_id, exc)
            continue

        saved += 1
        if saved % PROGRESS_LOG_INTERVAL == 0:
            logger.info("Progress: %d/%d images on disk", saved, num_images)

    logger.info(
        "Done: %d total on disk (%d newly downloaded, %d already present)",
        saved,
        saved - skipped,
        skipped,
    )
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a subset of COCO val2017 images and the canonical "
            "instances_val2017.json from HuggingFace mirrors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of val2017 images to download.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("~/coco2017").expanduser(),
        help="Root output directory (will create annotations/ and val2017/ underneath).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    out_dir: Path = args.out_dir.resolve()
    annotations_path = out_dir / "annotations" / "instances_val2017.json"
    images_dir = out_dir / "val2017"

    logger.info("=" * 70)
    logger.info("COCO val2017 subset acquisition")
    logger.info("=" * 70)
    logger.info("  Output root:    %s", out_dir)
    logger.info("  Annotations:    %s", annotations_path)
    logger.info("  Images dir:     %s", images_dir)
    logger.info("  Target count:   %d", args.num_images)
    logger.info("=" * 70)

    download_annotations(annotations_path)
    saved = download_images(images_dir, args.num_images)

    logger.info("=" * 70)
    logger.info("Acquisition complete:")
    logger.info("  Annotations:    %s", annotations_path)
    logger.info("  Images:         %s (%d files)", images_dir, saved)
    logger.info("=" * 70)
    logger.info("Next: run the evaluator with these paths:")
    logger.info(
        "    python evaluation/mobilenetv2/evaluate.py \\"
        "\n        --data-path %s \\"
        "\n        --coco-annotations %s \\"
        "\n        --max-samples %d \\"
        "\n        ...",
        images_dir,
        annotations_path,
        saved,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
