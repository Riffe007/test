cd ~/Documents/projects/MetaExecuTorch

find . -type d -iname "*voc2012*" 2>/dev/null
find . -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -path "*voc2012*" 2>/dev/null | head -10
find . -name "voc-model-labels.txt" 2>/dev/null
find . -name "mobile_net_v2_ssd.pth" 2>/dev/null
