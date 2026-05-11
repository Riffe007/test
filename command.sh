cd ~/Documents/projects/MetaExecuTorch && python -c "
import sys, types, torch
sys.path.insert(0, 'model_sources/MobileNetV2/src/pytorch/pytorch-ssd')
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
m = create_mobilenetv2_ssd_lite(num_classes=21, is_test=True)
m.load_state_dict(torch.load(
    'model_sources/MobileNetV2/weights/mb2-ssd-lite-mp-0_686.pth',
    map_location='cpu', weights_only=False), strict=True)
m.eval()
m.config = types.SimpleNamespace(
    center_variance=m.config.center_variance,
    size_variance=m.config.size_variance,
    priors=m.config.priors,
)
torch.save(m, 'model_sources/MobileNetV2/weights/mobile_net_v2_ssd.pth')
print('done')
"
