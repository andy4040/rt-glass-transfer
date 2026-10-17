#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/experiments/hf-cache
export PIP_NO_CACHE_DIR=1
python3 -m venv /workspace/experiments/download-env
/workspace/experiments/download-env/bin/python -m pip install huggingface_hub==0.25.0
/workspace/experiments/download-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('nexuslrf/diffusion_renderer-inverse-svd',
    revision='54243c27ad6299e43159410188403f7136b2dcba',
    local_dir='/workspace/experiments/diffusion-renderer/checkpoints/diffusion_renderer-inverse-svd')
snapshot_download('stabilityai/stable-video-diffusion-img2vid',
    revision='9cf024d5bfa8f56622af86c884f26a52f6676f2e',
    allow_patterns=['image_encoder/config.json','image_encoder/model.safetensors','feature_extractor/*'])
PY
touch /workspace/experiments/download_priors_5090.complete
