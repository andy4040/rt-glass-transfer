#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6
python3.10 -m venv "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --upgrade pip wheel 'setuptools<81'
# Runtime adaptation: installed toolkit is CUDA12.1, unlike upstream torch2.0.1 default.
"$PY" -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
"$PY" -m pip install 'numpy<2' imageio imageio-ffmpeg kornia matplotlib ninja open3d opencv-python plyfile scikit-image scikit-learn scipy torchmetrics==1.4.3 mediapy tqdm tensorboard lpips pytest
if [ ! -d vendor/RT-Splatting/.git ] && [ ! -f vendor/RT-Splatting/.git ]; then
  git clone https://github.com/sjj118/RT-Splatting.git vendor/RT-Splatting
fi
git -C vendor/RT-Splatting checkout 3f45b3cac4be04db9f3092234666b695991b268a
git -C vendor/RT-Splatting submodule update --init --recursive
"$PY" -m pip install --no-build-isolation vendor/RT-Splatting/submodules/simple-knn vendor/RT-Splatting/submodules/diff-surfel-anych
"$PY" -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3'
"$PY" -c 'import torch, diff_surfel_anych, simple_knn._C, nvdiffrast.torch; print(torch.__version__, "IMPORTS_OK")'
"$PY" -m pip freeze > installed_requirements.txt
touch install.complete
