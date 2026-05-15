# MobileNetV2-SSD-Lite (qfgaohao port, pickled nn.Module .pth)
elif (
    "mobile_net_v2_ssd" in config.model_path.lower()
    or "mobilenet_v2_ssd" in config.model_path.lower()
) and config.model_path.endswith(".pth"):
    return cls.load_mobile_net_v2_ssd_model(
        config.model_path,
        source_path=getattr(config, "source_path", None),
        num_classes=getattr(config, "num_classes", 21),
    )
