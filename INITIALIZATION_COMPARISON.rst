RaDe geometry initialization versus original SfM initialization
=============================================================

This experiment supersedes the previously queued GLINT-versus-Diffusion mask
comparison at the user's request on 2026-10-15. The GLINT paper-schedule training
and the old dependent queue were stopped. Existing outputs remain untouched.

Order and fixed protocol
------------------------

1. Generate one train-only mask set using the previously completed GLINT
   step_059999.pt (SHA256 450796e4fe03ea3699b733f9adb97d7f5594c8f68161d975307f46378b4421e7).
2. Train the IGS-bundled RaDe-GS for 30,000 steps, including its native depth-normal
   regularizer starting at step 10,000. No added plane loss or mask-conditioned
   loss is applied to RaDe. No compression is performed.
3. Initialize RT from RaDe geometry, run an isolated three-step smoke test,
   then train for 61,000 iterations and evaluate saved snapshots.
4. Initialize RT from the original SfM points, smoke-test, train and evaluate
   using the exact same masks, RGB, camera calibration, RT options and view seed.

Scene: coffee_martini, frame 0, resolution 676x507, 17 training cameras.
cam00 is absent from training, initialization, mask generation and prior generation.
Its RGB is read only by the independent evaluator. No test-score parameter selection.

Geometry transfer
-----------------

RT uses 2D surfels while RaDe uses 3D ellipsoids. We preserve centers, choose the
two largest principal axes as tangents, and orient the shortest axis as the normal.
RaDe's filter_3D correction is baked into scales and opacity, with corrected
opacity assigned to RT occupancy. This is a documented, lossy initialization
adapter, NOT an exact rendering-equivalent conversion.

No RaDe SH appearance or GLINT geometry is transferred. Each transferred center
receives RGB from its nearest original SfM point; higher SH coefficients start
at zero, as in baseline. Both arms initialize RT material opacity/transmissivity
at 0.5. Both light MLP and directional encoding initial states must match by hash.
Geometry is trainable throughout the normal RT schedule, not treated as ground truth.

The treatment includes differences in Gaussian count, scale, orientation, occupancy
and preprocessing compute. It is NOT a matched-count or matched-total-compute
ablation. Report RaDe preprocessing time separately and include it in total cost.
Reconstructed thin-axis normals may be ambiguous for near-isotropic Gaussians.
Neither correct glass masks nor high RGB PSNR establish correct 3D glass depth.

Commands and records
--------------------

On the configured GPU server::

  cd /opt/experiments/rt_glass_transfer
  nohup .venv/bin/python -u scripts/queue_initialization_comparison.py \
    > initialization_queue.log 2>&1 < /dev/null &

The paths and stage commands are recorded in runs/rade_vs_sfm_20261015/.
protocol.json records source revisions, fixed teacher identity and limitations.
shared_input_hashes.json fingerprints images, calibration, points and masks.
status.json records the current stage. rade/status.json, rt_rade/status.json and
sfm/status.json record iteration, Gaussian count, wall time and peak allocated GPU
memory. Evaluation is under each RT output's evaluation/<iteration>/ directory.
All 5k, 10k, 15k, 20k, 30k, 40k, 50k and 61k snapshots are evaluated after each
arm's training completes. Scores never feed back into training or configuration.

The queue survives SSH disconnects. It stops on errors and preserves partial
outputs. It does not claim optimizer-exact RT resume support. Do not blindly
restart partial runs; select an explicit recovery or new output directory.
