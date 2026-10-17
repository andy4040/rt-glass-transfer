#!/usr/bin/env bash
set -euo pipefail
python3 -m venv /workspace/experiments/preparation-env
PY=/workspace/experiments/preparation-env/bin/python
export PIP_NO_CACHE_DIR=1 QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$PY" -m pip install numpy==1.26.4 opencv-python==4.11.0.86 scipy plyfile
ROOT=/workspace/experiments/glint_n3dv_5090
"$PY" "$ROOT/prepare_n3dv.py" --colmap-only --output "$ROOT/coffee_frame0"
"$PY" "$ROOT/prepare_static.py" --root "$ROOT"
touch /workspace/experiments/prepare_cpu_5090.complete
