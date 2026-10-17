RTX 5090 restart, 2026-10-17
==========================

The old GPU host is inaccessible and no teacher checkpoint or complete mask
backup exists. A local coffee_frame0.tar.gz preserves all 18 synchronized frame-0
images and the supplied calibration. This is a NEW reproduction, not an exact
resume or a claim of identical teacher masks to the 3090 experiment.

The paired comparison still runs RaDe-initialized RT first and original SfM RT
second. Both use ONE newly generated GLINT mask set, 17 training cameras at
676 x 507, the same dedicated view RNG, and the same 61,000-update RT schedule.
cam00 RGB is excluded from triangulation, priors, all training and selection.
The existing frozen cam00 evaluation annotations are reused only by evaluation.

Recovery stages
---------------

1. Restore the archived images and calibration. CPU COLMAP triangulates training
   cameras only and verifies that supplied extrinsics/intrinsics did not change.
2. Generate official DiffusionRenderer normal/depth/diffuse-albedo/basecolor priors
   on training images only. Check normal-axis conventions against training SfM
   geometry before packaging priors; fail if evidence is ambiguous.
3. Build pinned GLINT a20591b20ef3365c230e3c95f749e65c498c68c4,
   gsplat 83a0dcd3f850cccdc7b92b50fd6a568e3260eae1 and its OptiX dependency.
   Require renderer/backward checks and a separate 2-update transport smoke run.
4. Train a NEW 60,000-update teacher from scratch, no added plane constraint:
   Interface warmup 5,000; Transmission/Reflection start 5,000; Interface geometry
   freeze 40,000. This differs from the lost early teacher's 1,000/3,000/31,000
   schedule; it follows the later requested warmup configuration. Record the new
   checkpoint SHA256 and never assert equality with the old teacher.
5. Export shared masks, train RaDe for 30,000 updates, then run the two 61,000
   update RT arms. See INITIALIZATION_COMPARISON.rst for transfer limitations.

Runtime
-------

Python 3.12, torch 2.7.1+cu128, torchvision 0.22.1, CUDA toolkit 12.8,
TORCH_CUDA_ARCH_LIST=12.0. Shared environment: /venv/main. NumPy 1.26.4 is a
compatibility adaptation for the combined legacy RT/RaDe/prior environment;
GLINT metadata requests NumPy >=2, so GLINT installs with --no-deps. CUDA and
transport checks gate training; dependency installation alone is not validation.
No renderer/loss substitution is made to bypass an unsupported CUDA extension.
CUDA 12.8 compilation required explicit standard headers: cfloat for simple-knn
FLT_MAX, and cstdint for RT/RaDe rasterizer integer types. The idempotent
scripts/patch_cuda_headers.py applies only these declarations; algorithms and
losses are unchanged. The RT CUDA extensions and basic 5090 tensor operation
passed installation; full transport checks remain a separate pipeline gate.

Server root: /workspace/experiments. Training data/teacher:
glint_n3dv_5090. RaDe source: igs-source/submodules/RaDe-GS, copied from the same
local IGS source as the previous experiment (f4b1880c5230834e276c8f8793ffaac364210a37).
Input preparation helpers are preserved in scripts/static_preparation; copy them
to the GLINT experiment root and provide 2dgs/scene/colmap_loader.py from RaDe.

Managed commands::

    bash scripts/install_5090.sh
    bash scripts/download_priors_5090.sh
    bash scripts/prepare_cpu_5090.sh
    /venv/main/bin/python -u scripts/rebuild_5090.py

These run as separate Supervisor programs, experiment-install-5090,
experiment-priors-download-5090, experiment-prepare-cpu-5090 and
experiment-rebuild-5090, with autorestart=false.
The rebuild waits for successful installation and weight download. SSH
disconnection does not stop them. Failed stages retain logs
and partial training is not silently restarted. A container destruction still
loses the non-volume workspace; checkpoint/mask backup is necessary.
CPU preparation has its own small environment so it can run while the much
larger CUDA runtime is downloading. GPU training remains sequential.

Status and logs::

    supervisorctl status experiment-install-5090 experiment-rebuild-5090
    cat runs/rebuild_5090_20261017/status.json
    cat runs/rade_vs_sfm_5090_20261017/status.json

Record elapsed time for the newly required teacher/preprocessing separately from
RaDe and RT times. Previous server timing is not an RTX 5090 estimate. No new
PSNR or training completion is claimed before actual runs finish.

Recovered input archive SHA256:
20cee90b121d8cee966aaf8989aff80c760cb7c49d2d368b6262ee933b6302ae.
RaDe/preparation source transfer archive SHA256:
e4edfb3e6a8956d5f1b4bba4d3cc5bc7eca23c99542b2c56bbc55dff6d9a6f74.
Inverse weights revision: 54243c27ad6299e43159410188403f7136b2dcba.
SVD encoder revision: 9cf024d5bfa8f56622af86c884f26a52f6676f2e.
