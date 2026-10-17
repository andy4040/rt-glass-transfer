Code audit of the RTX 5090 paired experiment
===========================================

Scope and timing
----------------

Audited the local adaptation and the actual 5090 sources while the new GLINT
teacher was running. No training was stopped, no cam00 image was used for this
audit, and no training hyperparameter or GLINT model state was changed. RaDe and
both RT arms had not started. The changes apply to their upcoming execution.

Concrete defects and corrections
-------------------------------

1. Unjustified IGS scene crop in the RaDe arm.

   The queue selected ``--loader N3D``. The installed IGS RaDe ``train.py`` lines
   196-199 delete all Gaussian centers with world z < 4.5 at densification. This
   is independent of mask/material/visibility. The recovered initial cloud has
   3,482 points, z range [2.4972057, 166.859726]; 3 points satisfy that crop.
   This count does not establish the crop's eventual effect after densification.

   The queue now uses the generic COLMAP loader, retaining all 17 supplied train
   cameras (eval=false). A pre-import guard rejects IGS dataset-specific crop
   loaders and an unintended second train/eval split. The RaDe source is kept
   intact and its real file hashes are recorded. This fixes a comparison
   confound; it does not prove or promise improved glass geometry.

2. Missing geometry counted as transparent glass in evaluation.

   RT's composited material ``surface_opacity`` is alpha weighted. Thus the old
   ``1 - surface_opacity`` can be 1 for an empty ray. A synthetic all-glass
   reference with an empty rendering obtained raw-proxy IoU=1. An opaque surface
   with coverage=0.4 similarly yielded raw transparency=0.6.

   Preserve and label the raw proxy, and add conditional material transparency
   ``1 - surface_opacity / surface_alpha`` with fixed coverage >= 0.7 and material
   threshold >= 0.5. All originally annotated pixels stay in the denominator;
   uncovered reference glass counts as a false negative. Save coverage/no-hit
   statistics and maps. The upstream RT training losses remain unchanged.

3. Stale teacher masks, copied masks and completion artifacts could be reused.

   Existing manifests were accepted by existence; input/protocol hashes could
   be overwritten on rerun. Validation checked exported masks, not necessarily
   the copied labels read by RT. Completion markers were not bound to outputs.

   Check the teacher SHA, compare actual packaged labels, preserve immutable
   protocol/input records, and bind completions to source/config/input/output
   hashes. Stop on a mismatch or unverified legacy marker. Partial output
   directories are preserved even if the crash preceded the first status file.

4. RT optimizer checkpoint resume was accepted despite incomplete serialization.

   The pinned upstream capture/restore omits material/shading state needed for
   exact continuation. The wrapper now rejects ``--start_checkpoint`` including
   equals/abbreviated forms before importing CUDA. Saved PLY plus shading-module
   artifacts remain usable for rendering, not optimizer-exact resume.

5. Silent source-adaptation failures and ignored config changes.

   Unchecked string replacements could silently fail after an upstream change,
   restoring the upstream view RNG or reporting path. Pin the normalized trainer
   source SHA and check that every replacement has exactly one target. Reject
   unsupported seed/split values rather than recording values the wrapper does
   not apply. Record actual critical Python source hashes, including guards.

6. RaDe and RaDe-initialized RT shared one training log/command name.

   Both stages used ``rade_train``, which appended dissimilar logs and overwrote
   the recorded command. The RaDe pretraining stage now uses ``rade_pretrain``;
   RT retains ``rade_train``. The integration test checks distinct stage names.

Validation and provenance
-------------------------

CPU regressions run on the server in a separate staging directory with
``CUDA_VISIBLE_DEVICES=``. They cover camera exclusion, mask validity/losses,
geometry transfer under 40 arbitrary rotations, metric denominators, empty-ray
transparency, stale/corrupt records, early checkpoint refusal, source drift,
disk preflight and orchestration. Final result: 50 passed, 6 subtests passed in
2.41 seconds on the actual server CPU, including seven mocked orchestration
integration cases. The command was::

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /venv/main/bin/python -m pytest tests -q

A 3-GiB free-space preflight is a floor, not an
estimate of all future checkpoint storage. Free space at inspection was 14 GiB.

The live source commits inspected were:

* GLINT: a20591b20ef3365c230e3c95f749e65c498c68c4; clean.
* gsplat: 83a0dcd3f850cccdc7b92b50fd6a568e3260eae1; two CUDA event API fixes.
* RT: 3f45b3cac4be04db9f3092234666b695991b268a; standard CUDA header additions.
* IGS RaDe train.py SHA256:
  d518c157b6a24b18b8a6e6ab23ee1f1c2af42f176d8ef71277c74f437058c705.

The header additions, event API correction and restored GLM links are build
compatibility changes. Earlier GLINT CUDA forward/backward and disposable full
transport checks passed. The upcoming RT arms still require their queued GPU
smoke tests and actual training; CPU success is not proof of all CUDA paths or
numerical stability over 61,000 iterations.

Remaining scientific and operational limits
-------------------------------------------

* No cam00 optimizer/prior/triangulation input leakage was found in the inspected
  paths. Preparation necessarily reads cam00 to resize/copy it into isolated
  evaluation storage. Scores never select training settings.
* The fixed ROI covers the rear window, excluding the bright right window/blinds.
  Its scores cannot verify the user's right-window staircase-depth complaint.
  No depth ground truth is available. Do not change the primary ROI after scores.
* RaDe ellipsoids become 2D RT surfels via a lossy tangent-plane initialization.
  Quaternion conventions, handedness, axes, shapes and filter correction passed
  the checked cases; near-isotropic ellipsoids still lack a unique surface normal.
* The treatment changes Gaussian count, scales, occupancy and pretraining cost,
  not just center locations. Report these differences alongside RGB metrics.
* Upstream RT/RaDe PLY saves occur before the labeled iteration's optimizer step;
  the final loop skips that step. Iteration N does not mean N parameter updates.
  Both RT arms keep the same upstream behavior.
* Full CUDA binary/environment hashing and mid-stage disk reservation are not
  implemented. Dependency-install markers in the parent rebuild pipeline are
  not a universal environment recovery system. Do not reinterpret an arbitrary
  modified environment or partial training as an exact resume.
