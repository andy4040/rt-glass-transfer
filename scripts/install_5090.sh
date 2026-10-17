#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/venv/main/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=12.0
export PIP_NO_CACHE_DIR=1
cd /workspace/experiments/rt-glass-transfer
apt-get update
apt-get install -y --no-install-recommends colmap libegl1 libgl1 libglib2.0-0 cmake
python -m pip install 'setuptools<81' wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy==1.26.4' 'opencv-python==4.11.0.86' plyfile scipy pillow pyyaml lpips pytest imageio imageio-ffmpeg kornia matplotlib open3d scikit-image scikit-learn torchmetrics==1.4.3 mediapy tqdm tensorboard tensorboardX einops braceexpand omegaconf ffmpegcv OpenEXR 'diffusers==0.32.0' 'huggingface_hub==0.25.0' 'accelerate==1.3.0' 'transformers==4.47.1' 'peft==0.7.0' 'datasets==2.19.0'
python scripts/patch_cuda_headers.py
python -m pip install --no-build-isolation vendor/RT-Splatting/submodules/simple-knn vendor/RT-Splatting/submodules/diff-surfel-anych
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
python -c 'import torch, diff_surfel_anych, simple_knn._C, nvdiffrast.torch; print(torch.__version__, torch.cuda.get_device_name(), torch.ones(16, device="cuda").sum().item(), flush=True)'
python -m pip freeze > installed_requirements_5090.txt
touch /workspace/experiments/install_5090.complete
