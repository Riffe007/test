cd ~/Documents/projects/MetaExecuTorch/executorch-toolkit
sed -n '380,420p' export/vision/model_loader.py


cp export/vision/model_loader.py export/vision/model_loader.py.bak.$(date +%s)
   # apply patch

cp export/vision/model_loader.py export/vision/model_loader.py.bak.$(date +%s)
   # apply patch


python -c "
   from export.vision.model_loader import <load_fn>
   m = <load_fn>('<config_or_path>')
   print(type(m).__name__, sum(p.numel() for p in m.parameters()))
   "


python -m export.vision.pytorch_to_executorch_vision \
     export/configs/vision/config_mobile_net_v2_ssd.json --generate-report
     
   

checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
if isinstance(checkpoint, torch.nn.Module):
    state_dict = checkpoint.state_dict()
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint
missing, unexpected = model.load_state_dict(state_dict, strict=True)  # strict=True!
