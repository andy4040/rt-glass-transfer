Joint RaDe Interface / Transmission prototype
============================================

Status
------
Research implementation, not a demonstrated improvement on N3DV. CPU unit
tests (13), a supervised synthetic optimization (250 steps), and the real
RaDe CUDA smoke test on RTX 3080 pass. Dataset integration is still pending. The synthetic test
has known depth targets and does not demonstrate automatic glass detection.

Source and architecture
-----------------------
Official IGS: https://github.com/yjb6/IGS.git
Commit: f4b1880c5230834e276c8f8793ffaac364210a37

The vendor checkout is unchanged. Two original RaDe GaussianModel instances
live in a single jointly trained model and checkpoint. Interface carries
separate material transmission/specular logits. Geometric opacity remains
the original RaDe coverage parameter. Material parameters inherit exact
clone/split parent values and follow pruning; their Adam states are saved.

Both primary camera passes use the original RaDe CUDA renderer. Separate
Interface and Transmission direct depths, plus DiffusionRenderer diffuse
albedo/basecolor, provide detached pseudo-targets with an unknown region.
These targets are heuristics and can be wrong; they are not calibrated glass
probabilities. No separately trained GLINT model or mask teacher is used.

Secondary transmission/reflection currently use a differentiable PyTorch
anisotropic 3D Gaussian reference tracer. It is NOT the original RaDe
rasterizer, GLINT's OptiX tracer, or a faithful GLINT reproduction. Reflection
queries the same Transmission field. Transmission is a straight thin-sheet
approximation, without refraction or thickness. Secondary origins and normals
are detached from Interface geometry. Origin-support rejection is approximate.
The reference tracer is O(number of rays * number of Gaussians), with chunked
checkpointing; full-scene throughput and memory remain to be measured.

Training
--------
Default: 30,000 iterations, Interface preparation for 6,000 iterations,
joint training and prior guidance from 6,001. Original RaDe normal consistency
uses expected/median depth (0.4/0.6), beginning at its default 10,000 schedule.
This is a new experiment schedule, not the original IGS static/compression
pipeline or GLINT paper schedule. No planar loss or compression is added.
Final transport RGB uses sampled patches; direct Transmission RGB and geometry
use complete training views. Coefficients have not been benchmarked/tuned.

Inputs must be train-only flat camNN.png images, centered PINHOLE text COLMAP
calibration and initial points, plus verified per-camera NPZ priors.
cam00 is rejected. Priors must use matching image hashes, frame and geometry;
prepare_priors.py converts existing DiffusionRenderer outputs, not inference.
It requires an explicit training-only normal-coordinate verification report.
Generic COLMAP loading avoids the original N3D loader's world-z pruning rule.

Commands (from this directory)::

  python -m unittest discover -s tests -v
  python preflight.py
  python build_extensions.py
  python gpu_smoke.py
  python train.py --source DATA --priors PRIORS --output runs/smoke --config configs/smoke.json
  python train.py --source DATA --priors PRIORS --output runs/reference --config configs/reference_30k.json

Use --resume with the joint checkpoint and exactly matching configuration,
calibration and prior provenance. Geometry, material optimizers, RNG and
remaining shuffled views are restored. PLY alone cannot restore this model.

GPU deployment 2026-10-18
-------------------------
Replacement server reports RTX 3080 10GB, CUDA toolkit 13.0, Python 3.12.
setup_gpu.sh installs pinned torch 2.9.1 / torchvision 0.24.1 cu130, fetches
official sources, builds staged copies, and executes CPU and CUDA smoke tests.
Supervisor job: rade-glint-setup. Remote directory:
/workspace/experiments/rade_glint_20261018. Logs: setup.log, gpu_smoke.log.
Successful completion writes gpu_setup.complete; this does NOT mean dataset
training has started. No training data or prior weights were present initially.
No instance destruction is scheduled.

GPU smoke passed on 2026-10-18 KST: original CUDA forward/backward, material
feature gradients isolated from geometry, transmission gradients, and checkpoint
render parity. Peak allocated memory was 25,084,416 bytes for ONLY nine synthetic
Gaussians at 32x32; this cannot estimate full-scene memory or throughput.
Recovered local coffee_frame0.tar.gz supplies synchronized images/calibration;
Fresh train-only triangulation completed. This is a validation snapshot, not
a live training-status report.

Automatic prior and training pipeline
------------------------------------
run_pipeline.py runs under the rade-glint-pipeline Supervisor service configured
by pipeline.conf. It requires gpu_setup.complete and triangulation.complete.
The stages are dependency installation, train-only input preparation, pinned
weight downloads, DiffusionRenderer inference, normal-coordinate verification,
prior packaging, a three-step dataset smoke test, and fresh 30,000-step training.
Every successful stage writes a completion marker in pipeline/. A failure
records the error in pipeline/status.json and prevents later stages, including
training. Partial training is never silently overwritten or restarted.

prepare_joint_inputs.py creates centered PINHOLE inputs at 676x507 from fixed
calibration and orders the 17 training views by camera proximity for prior
inference. All images are frame zero; cam00 RGB is excluded. No GLINT teacher
is trained. verify_prior_coordinates.py checks normal axis signs using only
training-camera SfM tracks and planar neighborhoods before packaging priors.

infer_priors_lowmem.py runs the pinned official DiffusionRenderer entrypoint
with CPU model offload, UNet forward chunking, one-image VAE encoding and
decode_chunk_size=1. The 24-frame input is padded by repeating the last training
view; inference uses 512x384, 20 steps, seed 0, and four passes: normal, depth,
diffuse albedo and basecolor. A one-frame temporal decode changes context and
is not claimed bit-identical to the default decoder. Source hash and memory
adaptations are recorded in prior_runtime.json. These changes target RTX 3080
10GB, but full-scene performance and quality still require validation.

Supervisor keeps the pipeline alive after SSH disconnect. It does not create
or destroy GPU instances. Model weights, inputs, priors, logs, status and
training checkpoints are excluded from Git; this repository is a code snapshot,
not an experiment-data backup.
