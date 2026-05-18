cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit

python - <<'PY'
from pathlib import Path
import re

path = Path("evaluation/mobilenetv2/evaluate.py")
text = path.read_text()

old = '''def load_executorch_model(pte_path: Path) -> Any:
    """Load a ``.pte`` ExecuTorch program for inference.

    # ── VERIFY ── ExecuTorch runtime API. This is the most version-sensitive
    # spot in the file. The pybindings loader below is common; replace it with
    # whatever your toolkit uses if it differs.
    """
    from executorch.extension.pybindings.portable_lib import (  # type: ignore
        _load_for_executorch,
    )

    program = _load_for_executorch(str(pte_path))
    logger.info("ExecuTorch program loaded: %s", Path(pte_path).name)
    return program
'''

new = '''def load_executorch_model(pte_path: Path) -> Any:
    """Load a .pte ExecuTorch program, including external .ptd data when present."""
    from executorch.extension.pybindings.portable_lib import (  # type: ignore
        _load_for_executorch,
    )

    pte_path = Path(pte_path).expanduser().resolve()

    ptd_candidates = [
        pte_path.with_suffix(".ptd"),
        pte_path.parent / f"{pte_path.stem}.ptd",
        pte_path.parent / f"{pte_path.stem}_data.ptd",
        pte_path.parent / f"{pte_path.stem}.pte.ptd",
    ]

    ptd_path = next((p for p in ptd_candidates if p.is_file()), None)

    if ptd_path is not None:
        logger.info("ExecuTorch external data found: %s", ptd_path.name)

        load_attempts = [
            lambda: _load_for_executorch(str(pte_path), str(ptd_path)),
            lambda: _load_for_executorch(str(pte_path), external_data_path=str(ptd_path)),
            lambda: _load_for_executorch(str(pte_path), external_constants_path=str(ptd_path)),
            lambda: _load_for_executorch(str(pte_path), data_path=str(ptd_path)),
            lambda: _load_for_executorch(str(pte_path), ptd_path=str(ptd_path)),
        ]

        last_error = None
        for attempt in load_attempts:
            try:
                program = attempt()
                logger.info(
                    "ExecuTorch program loaded with external data: %s + %s",
                    pte_path.name,
                    ptd_path.name,
                )
                return program
            except TypeError as exc:
                last_error = exc
                continue

        logger.warning(
            "ExecuTorch loader did not accept .ptd args (%s). Falling back to .pte only.",
            last_error,
        )

    program = _load_for_executorch(str(pte_path))
    logger.info("ExecuTorch program loaded: %s", pte_path.name)
    return program
'''

if old not in text:
    text = re.sub(
        r'def load_executorch_model\(pte_path: Path\) -> Any:\n.*?\n\n# =============================================================================\n# SSD decoding',
        new + '\n\n# =============================================================================\n# SSD decoding',
        text,
        flags=re.S,
    )
else:
    text = text.replace(old, new)

path.write_text(text)
print("Patched:", path)
PY

python ./evaluation/mobilenetv2/evaluate.py \
  --config export/configs/vision/config_mobile_net_v2_ssd.json
