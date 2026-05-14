cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
sed -n '380,420p' export/vision/model_loader.py


checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
if isinstance(checkpoint, torch.nn.Module):
    state_dict = checkpoint.state_dict()
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint
missing, unexpected = model.load_state_dict(state_dict, strict=True)  # strict=True!
