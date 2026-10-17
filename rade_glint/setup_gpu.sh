#!/usr/bin/env bash
set -euo pipefail
cd /workspace/experiments/rade_glint_20261018
exec > >(tee -a setup.log) 2>&1
export PATH=/venv/main/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda PIP_NO_CACHE_DIR=1 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
date -u
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
mkdir -p vendor
if [ ! -d vendor/IGS ]; then git clone --recursive https://github.com/yjb6/IGS.git vendor/IGS; fi
git -C vendor/IGS checkout f4b1880c5230834e276c8f8793ffaac364210a37
git -C vendor/IGS submodule update --init --recursive
if [ ! -d vendor/simple-knn ]; then git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git vendor/simple-knn; fi
python build_extensions.py
python -m unittest discover -s tests -v
python gpu_smoke.py | tee gpu_smoke.log
touch gpu_setup.complete
