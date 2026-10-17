#!/usr/bin/env bash
set -euo pipefail
cd /workspace/experiments/rade_glint_20261018
exec > >(tee -a prepare_data.log) 2>&1
export PATH=/venv/main/bin:$PATH QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends colmap
if [ ! -d coffee_frame0 ]; then tar -xzf coffee_frame0.tar.gz --exclude='coffee_frame0/cam00'; fi
python prepare_n3dv.py --colmap-only --output coffee_frame0
touch triangulation.complete
