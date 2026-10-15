"""Export train-only pseudo masks, with a separate validity mask for unknown pixels.

GLINT is a learned material-transparency teacher. DiffusionRenderer is NOT a
glass classifier: the ``diffusion`` method below is an explicitly fixed material
heuristic using its diffuse-albedo and basecolor priors, with no GLINT geometry.
No cam00 RGB, prior, render or mask is opened by either exporter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings

import numpy as np
from PIL import Image


def check_train_names(names):
    names = list(names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Training camera names must be nonempty and unique")
    if any(not re.fullmatch(r"cam\d+", name) for name in names):
        raise ValueError("Expected explicit N3DV camXX training camera names")
    if "cam00" in names:
        raise ValueError("cam00 is evaluation-only; refusing mask/prior generation")
    return names


def binary_mask(probability, threshold=0.5):
    p = np.asarray(probability)
    if p.ndim != 2:
        raise ValueError("Probability must have shape H,W")
    if not 0 < threshold < 1:
        raise ValueError("Binary threshold must lie strictly inside (0,1)")
    finite = np.isfinite(p)
    if np.any(finite & ((p < 0) | (p > 1))):
        raise ValueError("Finite probability values must lie in [0,1]")
    return (finite & (p >= threshold)).astype(np.uint8) * 255


def glint_validity(probability, coverage, low=0.2, high=0.8, min_coverage=0.7):
    p, alpha = np.asarray(probability), np.asarray(coverage)
    if p.ndim != 2 or p.shape != alpha.shape:
        raise ValueError("Probability and coverage must have the same H,W shape")
    if not 0 <= low < high <= 1 or not 0 <= min_coverage <= 1:
        raise ValueError("Invalid confidence thresholds")
    return (np.isfinite(p) & np.isfinite(alpha) & (alpha > min_coverage)
            & ((p <= low) | (p >= high)))


def diffusion_labels(albedo, basecolor, albedo_threshold=0.05,
                     basecolor_threshold=0.25, negative_albedo=0.10,
                     negative_basecolor=0.20):
    """Fixed prior-only rule. Values .5 are unknown, NOT calibrated probabilities."""
    a, b = np.asarray(albedo), np.asarray(basecolor)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("Both priors must have matching H,W,3 shapes")
    if not 0 <= albedo_threshold < negative_albedo <= 1:
        raise ValueError("Albedo thresholds must define a nonempty dead band")
    if not 0 <= negative_basecolor < basecolor_threshold <= 1:
        raise ValueError("Basecolor thresholds must define a nonempty dead band")
    finite = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    a, b = a.mean(-1), b.mean(-1)
    positive = finite & (a < albedo_threshold) & (b > basecolor_threshold)
    negative = finite & ((a > negative_albedo)
                         | ((a < albedo_threshold) & (b < negative_basecolor)))
    values = np.full(a.shape, 0.5, dtype=np.float32)
    values[positive], values[negative] = 1.0, 0.0
    return values, positive | negative


def read_train_cameras(root):
    """CPU-only intrinsics inventory. Read train intrinsics, never evaluation files."""
    import cv2
    fs = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(root / "intri.yml")
    try:
        node = fs.getNode("names")
        names = check_train_names([node.at(i).string() for i in range(node.size())])
        records = {}
        for name in names:
            w, h = (int(fs.getNode(prefix + name).real()) for prefix in ("W_", "H_"))
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid image geometry for {name}")
            with Image.open(root / "images" / name / "000000.png") as image:
                if image.size != (w, h):
                    raise ValueError(f"RGB/calibration shape mismatch for {name}")
            records[name] = (w, h)
        return records
    finally:
        fs.release()


def load_prior(path, size):
    """Match GLINT uint8 resize before normalization; priors may be 512x384."""
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if (image.shape[1], image.shape[0]) != size:
        interpolation = (cv2.INTER_AREA if size[0] <= image.shape[1]
                         and size[1] <= image.shape[0] else cv2.INTER_LINEAR)
        image = cv2.resize(image, size, interpolation=interpolation)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_maps(destination, name, probability, valid, coverage=None, rgb_path=None):
    mask = binary_mask(probability)
    if valid.shape != mask.shape or valid.dtype != np.bool_:
        raise ValueError("Validity must be boolean with the exact probability shape")
    if not valid.any():
        raise ValueError(f"{name} has no valid pseudo-label pixels; refusing empty supervision")
    # Unknown pixels are encoded zero for display/storage; mask_valid determines
    # whether either positive or negative losses may use them.
    mask[~valid] = 0
    Image.fromarray(mask, mode="L").save(destination / "transparent_masks" / f"{name}.png")
    Image.fromarray(valid.astype(np.uint8) * 255, mode="L").save(destination / "mask_valid" / f"{name}.png")
    np.save(destination / "probabilities" / f"{name}.npy", probability.astype(np.float32))
    if coverage is not None:
        np.save(destination / "coverage" / f"{name}.npy", coverage.astype(np.float32))
    positive = (mask > 0) & valid
    if rgb_path is not None:
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if rgb.shape[:2] != mask.shape:
            raise ValueError("Overlay RGB and pseudo-label shape mismatch")
        overlay = rgb.copy()
        overlay[positive] = .60 * rgb[positive] + .40 * np.array([255, 40, 20])
        overlay[~valid] = .75 * rgb[~valid] + .25 * np.array([70, 70, 255])
        Image.fromarray(np.rint(overlay).clip(0, 255).astype(np.uint8)).save(
            destination / "overlays" / f"{name}.png")
    if not positive.any():
        warnings.warn(f"{name}: no confident positive pixels; only confident negatives supervise")
    return {"camera": name, "width": int(mask.shape[1]), "height": int(mask.shape[0]),
            "valid_pixels": int(valid.sum()), "positive_pixels": int(positive.sum()),
            "negative_pixels": int((valid & ~positive).sum()),
            "unknown_pixels": int((~valid).sum()),
            "valid_fraction": float(valid.mean()), "positive_fraction": float(positive.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["glint", "diffusion"], required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glint-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    cameras = read_train_cameras(args.train_root)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.method == "glint" and (args.glint_root is None or args.checkpoint is None):
        parser.error("--method glint requires --glint-root and --checkpoint")
    manifest = {"schema_version": 1, "method": args.method, "frame": 0,
                "train_cameras": list(cameras), "held_out_camera": "cam00",
                "held_out_used": False, "binary_threshold": 0.5,
                "validity_required_for_all_mask_losses": True,
                "output_geometry": "Exact training RGB pixels; no crop or camera transform",
                "overlay_legend": "red=confident transparent; blue=unknown; unchanged=confident opaque",
                "settings_selected_on_cam00": False, "cameras": []}
    for name in ("transparent_masks", "mask_valid", "probabilities", "coverage", "overlays"):
        (args.output / name).mkdir(parents=True)
    if args.method == "glint":
        import torch
        sys.path.insert(0, str(args.glint_root.resolve()))
        from glint.camera import load_easyvolcap_cameras
        from glint.checkpoint import load_glint_checkpoint
        from glint.transport import render_glint_transport
        model = load_glint_checkpoint(args.checkpoint, device="cuda")
        glint_cameras = load_easyvolcap_cameras(args.train_root)
        check_train_names(glint_cameras)
        if set(glint_cameras) != set(cameras):
            raise ValueError("Camera inventory changed while preparing masks")
        manifest.update(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                        source="GLINT learned material_trans_map, conditioned on Interface hit",
                        uncertainty={"coverage_gt": 0.7, "negative_le": 0.2, "positive_ge": 0.8},
                        render_stage="interface (material field is identical across stages)")
        revision = subprocess.run(["git", "-C", str(args.glint_root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False)
        manifest["glint_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        with torch.no_grad():
            for name, (w, h) in cameras.items():
                output = render_glint_transport(glint_cameras[name].to("cuda"), model,
                                                None, stage="interface")
                probability = output.material_trans_map.reshape(h, w).float().cpu().numpy()
                coverage = output.acc_map.reshape(h, w).float().cpu().numpy()
                valid = glint_validity(probability, coverage)
                row = save_maps(args.output, name, probability, valid, coverage,
                                args.train_root / "images" / name / "000000.png")
                manifest["cameras"].append(row)
                print(json.dumps(row), flush=True)
                del output
    else:
        manifest.update(source="DiffusionRenderer material-prior heuristic; not a learned glass classifier",
                        probability_semantics="1 positive, 0 negative, .5 unknown heuristic score; not calibrated probability",
                        uncertainty={"positive": "mean diffuse_albedo < .05 AND mean basecolor > .25",
                                     "negative": "mean diffuse_albedo > .10 OR (mean diffuse_albedo < .05 AND mean basecolor < .20)"},
                        prior_resize="OpenCV INTER_LINEAR when upsampling; INTER_AREA when downsampling; uint8 then /255",
                        uses_glint_depth=False, uses_glint_checkpoint=False)
        for name, size in cameras.items():
            paths = {kind: args.train_root / "diffrens" / kind / name / "000000.png"
                     for kind in ("diffuse_albedo", "basecolor")}
            albedo, basecolor = (load_prior(paths[kind], size) for kind in paths)
            probability, valid = diffusion_labels(albedo, basecolor)
            row = save_maps(args.output, name, probability, valid,
                            rgb_path=args.train_root / "images" / name / "000000.png")
            row["prior_sha256"] = {kind: sha256(path) for kind, path in paths.items()}
            manifest["cameras"].append(row)
            print(json.dumps(row), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
