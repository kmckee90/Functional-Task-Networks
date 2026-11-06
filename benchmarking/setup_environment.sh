#!/usr/bin/env bash
set -euo pipefail
source /home/dev/workspace/remote_env.sh
mkdir -p "$BM_ROOT/envs" "$BM_ROOT/manifests"
# Run once on bm1; all three pods use the same image and shared environment.
exec 9>"$BM_ROOT/envs/setup.lock"
flock 9
if [[ ! -x "$BM_ROOT/envs/main/bin/python" ]]; then
    python -m venv --system-site-packages "$BM_ROOT/envs/main"
fi
"$BM_ROOT/envs/main/bin/python" -m pip install 'timm==0.9.8' numpy scipy pandas \
    scikit-learn matplotlib imageio tqdm pillow pyyaml requests \
    'huggingface_hub==0.25.2' hydra-core omegaconf einops tensorboard torchmetrics \
    kornia wandb gdown python-dotenv open_clip_torch ftfy onedrivedownloader \
    optuna pytz pytorch-ignite 'ray[tune]' tensorboardx tabulate json2html \
    more_itertools plotly pydot pycocotools visdom sacred pymongo
"$BM_ROOT/envs/main/bin/python" -m pip freeze > "$BM_ROOT/manifests/environment.txt"
"$BM_ROOT/envs/main/bin/python" -c 'import torch, torchvision, timm; print("torch",torch.__version__,"torchvision",torchvision.__version__,"timm",timm.__version__); assert torch.cuda.device_count() == 4'
