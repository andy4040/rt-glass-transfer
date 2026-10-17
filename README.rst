RT glass-mask transfer on synchronized N3DV
==========================================

Current execution: the 2026-10-17 RTX 5090 restart compares RaDe versus SfM
initialization with shared GLINT masks (RaDe first). See RESTART_5090.rst and
INITIALIZATION_COMPARISON.rst. The mask-source comparison described below is the
earlier design and is not an additional pair of jobs in the current queue.

This experiment compares two sources of transparent-region supervision for the
official RT-Splatting representation and renderer. It uses coffee_martini frame
0, the same 17 training cameras at 676 x 507, and cam00 exclusively for evaluation.
It is a static multi-view experiment, not a 4D extension or an official N3DV benchmark.

Method A: GLINT teacher masks
----------------------------

Wait for the current no-plane-loss GLINT run to complete 60,000 updates. Rasterize
its learned conditional material transparency on TRAIN views only. Pixels with
coverage > .7 and transparency >= .8 are reliable positives; <= .2 are negatives.
The rest are unknown. The teacher has learned using geometric separation and
DiffusionRenderer priors, but the exported field is not ground-truth glass geometry.

Advantages: reuse existing geometry/material optimization; shared 3D field provides
cross-view structure. Limitations: teacher errors may be inherited; account for
GLINT training time as well as RT training time. Better RGB does not prove correct
glass depth. No GLINT Gaussian positions or radiance weights initialize RT.

Method B: DiffusionRenderer-only material heuristic
-------------------------------------------------

Use the already-generated TRAIN diffuse-albedo and basecolor priors, with no GLINT
checkpoint or GLINT depth. Fixed positives: mean albedo < .05 and basecolor > .25.
Negatives: albedo > .10, or albedo < .05 and basecolor < .20. Other pixels are unknown.
These thresholds are declared before evaluation, derived from the GLINT material
cue defaults with uncertainty margins. They are not selected using cam00.

Advantages: no GLINT optimization needed; inexpensive reusable preprocessing.
Limitations: this is explicitly a heuristic baseline, not a learned glass detector.
Dark materials/metals and prediction errors can be confused with transparency.
Its 0/.5/1 score is not a calibrated probability. A learned segmentation head would
be a different method requiring training labels and separate validation.

What this comparison can establish
----------------------------------

Both routes use the same prior generation, training RGB, camera parameters, point
cloud, renderer, seed, dedicated view-sampling RNG and 61,000-step RT configuration.
Initial tensor checksums are compared. Later densification and CUDA operations can
diverge; bitwise identical optimization is not promised. Mask coverage is reported
because different numbers of supervised pixels also affect the outcome.

The official RT paper already uses SAM2 masks. Therefore this experiment tests
alternative physically informed pseudo supervision, not the novelty of automating
manual labels. A SAM2-mask baseline is still needed for a claim against the published
mask-generation method. Two methods alone cannot isolate every source of benefit.

Explicit modifications to upstream
----------------------------------

* Pinned RT upstream is a submodule: 3f45b3cac4be04db9f3092234666b695991b268a.
* Geometry/optical opacity factorization, Sph-Mip shading, rasterizer, rendering
  composition, original RGB/geometric losses and specular gradient gating remain.
* All THREE mask-related losses ignore unknown pixels; empty selections give finite
  zero loss. The same adapter is used for both methods. This is not unmodified RT
  training and no local plane constraint is added.
* A dedicated view RNG prevents densification RNG changes from changing camera order.
* cam00 never enters the training Scene. Train without --eval to avoid upstream's
  default every-eighth-camera holdout. The independent evaluator reads cam00 after
  training. It reports crop-normalized PSNR, SSIM and LPIPS, not upstream's
  zero-padded window PSNR.
* N3DV warmup is explicitly 1,000 steps, normal loss starts at 700, total 61,000.
  These are a fixed adaptation drawn from upstream examples, not scene-tuned values.
* The runtime uses Python 3.10 / torch2.3.1 cu121 to match the installed CUDA12.1
  toolkit, instead of upstream's torch2.0.1 requirement. CUDA imports and a deferred
  three-step renderer/backward smoke run gate production training.

Current experiment (2026-10-15)
------------------------------

The active experiment now compares **RaDe-GS geometry initialization first**
against **original SfM initialization second**, with one identical GLINT mask set.
The previous GLINT-versus-Diffusion mask queue was stopped by user request.
See INITIALIZATION_COMPARISON.rst for conversion details, fixed settings and commands.
The earlier mask comparison below is retained as a separate, inactive protocol.

Execution
---------

Clone recursively and install in a separate environment::

  git clone --recurse-submodules https://github.com/andy4040/rt-glass-transfer.git
  cd rt-glass-transfer
  bash scripts/install.sh

With existing GLINT input/prior artifacts, queue the comparison::

  nohup .venv/bin/python -u scripts/queue_experiments.py \
    --glint-base /path/to/glint_experiment \
    --glint-run /path/to/glint_experiment/glint_paper60k_20261015 \
    > queue.log 2>&1 < /dev/null &

The queue requires successful status, the completed-training log marker, and the
stable final checkpoint before any dependent GPU rendering/training. It stops on
failure rather than silently falling back to an intermediate teacher. Each route
exports masks, validates train-only calibration, runs a tiny isolated smoke test,
trains RT, and evaluates cam00. Method B starts after method A completes.

Progress: runtime/status.json and outputs/<method>/status.json. Masks and overlays:
masks/<method>/. Final metrics and full/window/layer renders: outputs/<method>/evaluation/.
Independent evaluation can also target any saved PLY/light/encoding triplet.

Upstream capture/restore does not contain all learned shading/material state;
this experiment does not claim optimizer-exact resume support. Preserve the saved
PLY plus light_mlp.pt and dir_encoding.pt for rendering. Interrupted production
training must be handled explicitly rather than reusing incomplete checkpoints.

No data, priors, model weights, credentials or output artifacts are committed.
Existing GLINT runs and input files are never overwritten.

Sources and licensing
---------------------

RT-Splatting: https://github.com/sjj118/RT-Splatting
RT paper: https://arxiv.org/abs/2605.18263
GLINT: https://github.com/youngju-na/GLINT
DiffusionRenderer: https://github.com/nv-tlabs/diffusion-renderer

Upstream RT carries both LICENSE and LICENSE.md, including noncommercial research
and evaluation restrictions inherited from dependencies. Both remain preserved in
the submodule; this wrapper does not relicense upstream code or grant rights to
third-party data/model weights. Attribution remains in the generated in-memory
training source. Use is subject to all applicable upstream terms.
