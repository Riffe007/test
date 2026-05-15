@staticmethod
def load_mobile_net_v2_ssd_model(
    model_path: str,
    source_path: Optional[str] = None,
    num_classes: int = 21,
) -> nn.Module:
    """Load qfgaohao MobileNetV2-SSD-Lite from a pickled SSD ``nn.Module`` checkpoint.

    Why this exists
    ---------------
    The qfgaohao .pth is a pickled ``nn.Module`` — not a state_dict. The base
    loader path uses ``torch.load(weights_only=True)`` (a hard fail on pickled
    classes) and ``load_state_dict(strict=False)`` (silent random-init on key
    mismatch). Net effect: a shape-correct model with garbage weights and
    mAP@0.5 ≈ 0.087 vs. standalone-verified ≈ 0.686.

    Contract
    --------
    Returns a bare ``nn.Module`` in ``eval()`` mode. Predictor wrapping
    (``create_mobilenetv2_ssd_lite_predictor``: decode + NMS) is the eval
    harness's job; the export pipeline only traces ``forward()``.

    Args
    ----
    model_path:    Path to the qfgaohao .pth checkpoint.
    source_path:   Directory containing the qfgaohao ``vision/`` package
                   (i.e. the root of the pytorch-ssd source tree).
                   Defaults to ``<project_root>/model_sources/MobileNetV2/src/pytorch/pytorch-ssd``.
    num_classes:   Detector class count including background. 21 for VOC.

    Security note
    -------------
    ``weights_only=False`` deserializes arbitrary pickled Python objects. Only
    point this at checkpoints from trusted sources under ``model_sources/``.

    Raises
    ------
    FileNotFoundError  weights or source tree missing.
    ImportError        qfgaohao ``vision.ssd.mobilenet_v2_ssd_lite`` unimportable.
    TypeError          checkpoint deserializes to neither ``nn.Module`` nor ``dict``.
    RuntimeError       state_dict has any missing/unexpected keys (no silent re-init).
    """
    # ── Resolve checkpoint ──────────────────────────────────────────────────
    weights = Path(model_path).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"MobileNetV2-SSD weights not found: {weights}")

    # ── Resolve qfgaohao source tree ────────────────────────────────────────
    if source_path:
        repo = Path(source_path).expanduser().resolve()
    else:
        # model_loader.py = executorch-toolkit/export/vision/model_loader.py
        # parents[3] = project root (sibling of executorch-toolkit/).
        project_root = Path(__file__).resolve().parents[3]
        repo = (
            project_root
            / "model_sources" / "MobileNetV2" / "src" / "pytorch" / "pytorch-ssd"
        ).resolve()

    sentinel = repo / "vision" / "ssd" / "mobilenet_v2_ssd_lite.py"
    if not sentinel.is_file():
        raise FileNotFoundError(
            f"qfgaohao pytorch-ssd source not found (expected {sentinel}). "
            f"Set `source_path` in the config to override."
        )

    print(f"Loading MobileNetV2-SSD-Lite: weights={weights} "
          f"num_classes={num_classes} source={repo}")

    # ── Import qfgaohao module under a scoped sys.path entry ────────────────
    repo_str = str(repo)
    inserted = repo_str not in sys.path
    if inserted:
        sys.path.insert(0, repo_str)
    try:
        try:
            from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
        except ImportError as e:
            raise ImportError(
                f"Cannot import qfgaohao MobileNetV2-SSD-Lite from {repo}. "
                f"Ensure vision/ssd/mobilenet_v2_ssd_lite.py exists.\nError: {e}"
            ) from e

        # ── Deserialize checkpoint (mirrors load_mobile_net_v1_model.load_checkpoint) ──
        try:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch builds without weights_only kwarg.
            checkpoint = torch.load(weights, map_location="cpu")

        # ── Normalize to a state_dict ───────────────────────────────────────
        if isinstance(checkpoint, nn.Module):
            state_dict = checkpoint.state_dict()
            source_kind = "pickled nn.Module"
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            source_kind = "dict checkpoint"
        else:
            raise TypeError(
                f"Unexpected checkpoint type {type(checkpoint).__name__} at "
                f"{weights}; expected nn.Module or dict."
            )

        # ── Build architecture and load weights strictly ────────────────────
        model = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    finally:
        if inserted:
            try:
                sys.path.remove(repo_str)
            except ValueError:
                pass  # Removed by something else; benign.

    # `strict=True` raises before this point on any mismatch, but PyTorch has
    # historically wobbled on this contract — guard explicitly so a future
    # downgrade to NamedTuple-only behaviour can't reintroduce the silent fail.
    if missing or unexpected:
        raise RuntimeError(
            f"MobileNetV2-SSD state_dict mismatch from {source_kind}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}. "
            f"First missing: {list(missing)[:3]} | "
            f"First unexpected: {list(unexpected)[:3]}"
        )

    model.eval()
    print(f"  ✓ MobileNetV2-SSD load OK: {len(state_dict)} tensors "
          f"(0 missing, 0 unexpected)")
    return model
