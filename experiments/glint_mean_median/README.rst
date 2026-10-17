GLINT: expected-depth + median-depth normal consistency
======================================================

This folder preserves the model code used by the GLINT-only experiment started
on 2026-10-17. It does not train RaDe-GS or RT-Splatting. Final quality results
were not yet available when this code was published.

Change
------

For both Interface and the directly rendered Transmission set::

    L_normal = 0.4 * L(normal_rendered, normal_from_expected_depth)
             + 0.6 * L(normal_rendered, normal_from_median_depth)

The two normals are computed independently. Depth maps are NOT mixed before
normal estimation. The outer normal loss coefficient remains 0.1, and the
existing inverse-depth weighting, prior losses and stage schedules are unchanged.
Transport, pseudo-mask guidance and depth-prior supervision still use the
original expected-depth surface. No plane loss or depth-distortion loss is added.

The mixture is recorded as ``GlintLossConfig.normal_consistency_median_weight``
(default 0.6); this experiment does not expose a YAML/CLI override for it.
The assumption that expected depth is used follows the unchanged transport
``depth_ratio=0.0`` setting. This ablation tests a hypothesis, not a guaranteed
solution for fragmented blinds or incorrect glass depth.

Source and provenance
---------------------

* Official GLINT: https://github.com/youngju-na/GLINT
* Upstream snapshot: a20591b20ef3365c230e3c95f749e65c498c68c4
* ``GLINT/`` contains the upstream source snapshot with only renderer.py and
  trainer.py modified. Original license, notices and citations are retained.
* ``changes.patch`` is the exact model patch used by the live experiment.
* ``provenance.json`` records the byte hashes of the running model files.
* No dataset, pretrained weights, trained checkpoints, server credentials or
  instance-deletion automation are included.

Protocol
--------

N3DV coffee_martini frame 0; 17 training cameras at 676 x 507. cam00 is held out
from training, prior generation and hyperparameter selection. Same train-only
SfM point cloud, reflection seed cloud and precomputed DiffusionRenderer priors
as the original GLINT baseline; seed 0. Fresh initialization, no resume.

* 0--4,999: Interface-only warmup.
* From step 5,000: enable Transmission and Reflection.
* At step 40,000: freeze Interface geometry, as in the baseline.
* Total: 60,000 iterations; checkpoints every 10,000; images every 5,000.

Run with an existing GLINT-compatible CUDA environment
----------------------------------------------------

From the repository root::

    python experiments/glint_mean_median/train.py \
        --train-root /path/to/prepared/data/train \
        --output /path/to/new/glint_mean_median_run

Add ``--print-only`` to inspect the generated command without launching.
Inputs must already include GLINT-format camera calibration, same-time images,
initial point clouds and DiffusionRenderer priors. They are not supplied here.
The launcher selects this snapshot via PYTHONPATH and refuses an existing
output folder. It never invokes the older paired-experiment queue.

Fresh environment
-----------------

The actual run used Linux, RTX 5090, PyTorch 2.7.1+cu128 and CUDA toolkit 12.8.
Do not use the upstream README's older CUDA example for a 5090. Install the
compatible PyTorch build and follow GLINT/README.md for Python dependencies.

This is a vendored snapshot, not a recursively cloned upstream repository.
Populate the OptiX tracer explicitly before building it::

    git clone https://github.com/xbillowy/diff-surfel-tracing.git \
        experiments/glint_mean_median/GLINT/third_party/diff-surfel-tracing
    git -C experiments/glint_mean_median/GLINT/third_party/diff-surfel-tracing \
        checkout ef6f24be9174eee037469f7e221cd72f6c0222cb
    git -C experiments/glint_mean_median/GLINT/third_party/diff-surfel-tracing \
        submodule update --init --recursive

Pin youngju-na/gsplat to 83a0dcd3f850cccdc7b92b50fd6a568e3260eae1.
The live 5090 environment also corrected two CUDA event calls in
``gsplat/cuda/csrc/Utils.cpp``: use ``cudaEventCreateWithFlags`` instead of
``cudaEventCreate`` for calls passing ``cudaEventDisableTiming`` (the
``events[device_id]`` and ``merge_event`` variables). Build with
``BUILD_2DGS=1 NUM_CHANNELS=3,4,6,7 BUILD_EXPERIMENTAL=0`` and
``TORCH_CUDA_ARCH_LIST=12.0``. This compatibility change is independent of the
mean/median experiment and was already present in the baseline environment.

Then install this GLINT folder without replacing the pinned dependencies and
build/verify the tracer::

    cd experiments/glint_mean_median/GLINT
    pip install --no-deps -e .
    python -m glint.install_optix
    python -m glint.verify_installation

Validation
----------

``validation/verify_gradients.py`` tests CUDA rasterization, not just algebra:
unchanged forward outputs against the original renderer, finite nonzero median
depth gradients to positions, and median-normal gradients to geometry.
Run it in the CUDA environment::

    python experiments/glint_mean_median/validation/verify_gradients.py

Recorded results are in ``gradient_checks.json``. A separate two-step full
Interface/Transmission/Reflection smoke run passed before the live training
started. Smoke states were not reused. PSNR should be measured using full GLINT
OptiX transport on cam00; a browser Gaussian viewer is not that renderer.
