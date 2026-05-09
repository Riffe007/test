# 1. Sanity check: maybe COCO is already on the system from another project
echo "=== Existing COCO on disk? ===" ; find /home /mnt 2>/dev/null \( -name "instances_val2017.json" -o -name "val2017" -type d \) 2>/dev/null | head -20

# 2. Fix the empty .pth (delete the 0-byte stub, redownload from GitHub)
echo "=== Re-fetching .pth from GitHub mirror ===" ; rm -f ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth ; wget -q --show-progress -O ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth https://github.com/fa0311/pytorch-ssd-archive-model/releases/download/v0.0.1/mb2-ssd-lite-mp-0_686.pth ; ls -la ~/Documents/projects/MetaExecuTorch/model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth

# 3. Probe whether HF *file-download* path works (tiny 43-byte file)
echo "=== HF file-download probe ===" ; curl -fLs --max-time 30 -o /tmp/hf_probe.txt https://huggingface.co/datasets/merve/coco/resolve/main/annotations/.gitattributes && wc -c /tmp/hf_probe.txt && head -1 /tmp/hf_probe.txt && echo "HF download: WORKS" || echo "HF download: BLOCKED"
