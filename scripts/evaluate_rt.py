#!/usr/bin/env python3
"""Independent cam00 evaluation of an official RT-Splatting saved model.

This executable never uses Scene or the training dataset loader. It renders a
camera with a dummy image, then opens the evaluation photograph for scoring.
No evaluation image, mask or metric is returned to the training process.

Requires the RT CUDA environment plus opencv-python, scikit-image and lpips.
Only load the PLY/PT triplet produced by your own trusted training process:
upstream saves its shading modules as Python pickle objects, not state_dicts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import os
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    """RGB [0,1] PSNR, averaged over exactly the supplied pixels/channels."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("PSNR requires equal, nonempty arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite image values")
    return float(-10.0 * np.log10(max(float(np.mean((prediction - target) ** 2)), 1e-12)))


def crop(image: np.ndarray, xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = xyxy
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError(f"Invalid exclusive-end ROI {xyxy} for {image.shape}")
    return image[y0:y1, x0:x1]


def roi_psnr(prediction: np.ndarray, target: np.ndarray, xyxy: tuple[int, int, int, int]) -> float:
    """Crop, then average; NEVER zero-pad the ROI into the whole image."""
    return psnr(crop(prediction, xyxy), crop(target, xyxy))


def scaled_roi(annotation: dict, width: int, height: int) -> tuple[int, int, int, int]:
    if annotation.get("camera") != "cam00" or not annotation.get("frozen_before_training", False):
        raise ValueError("A frozen cam00 evaluation annotation is required")
    rw, rh = annotation["annotation_reference_size"]
    return tuple(round(float(v) * s) for v, s in zip(
        annotation["window_xyxy"], (width / rw, height / rh, width / rw, height / rh)))


def glass_annotation(annotation: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Same polygon rasterization/boundary exclusion as GLINT score_results.py."""
    import cv2
    roi = scaled_roi(annotation, width, height)
    rw, rh = annotation["annotation_reference_size"]
    scale = np.array([width / rw, height / rh])
    glass = np.zeros((height, width), dtype=np.uint8)
    for polygon in annotation["glass_positive_polygons"]:
        cv2.fillPoly(glass, [np.round(np.asarray(polygon) * scale).astype(np.int32)], 1)
    for polygon in annotation.get("opaque_exclusion_polygons", []):
        cv2.fillPoly(glass, [np.round(np.asarray(polygon) * scale).astype(np.int32)], 0)
    radius = int(annotation.get("boundary_ignore_radius_at_eval_resolution", 0))
    if radius < 0:
        raise ValueError("Boundary ignore radius must not be negative")
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    boundary = cv2.dilate(glass, kernel) != cv2.erode(glass, kernel)
    valid = np.zeros_like(glass, dtype=bool)
    x0, y0, x1, y1 = roi
    # Validate the rectangle before assigning, rather than silently clipping.
    crop(glass, roi)
    valid[y0:y1, x0:x1] = True
    valid &= ~boundary
    return glass.astype(bool), valid


def localization_metrics(probability: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict:
    """Fixed 0.5 optical-opacity-complement test, only in the annotated ROI."""
    if probability.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Detection map, reference and scoring mask must have equal shapes")
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite detection probabilities")
    prediction = probability >= 0.5
    target, valid = target.astype(bool), valid.astype(bool)
    tp = int((prediction & target & valid).sum())
    fp = int((prediction & ~target & valid).sum())
    fn = int((~prediction & target & valid).sum())
    tn = int((~prediction & ~target & valid).sum())
    return {"threshold": 0.5, "iou": tp / max(tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "scored_pixels": int(valid.sum()), "positive_reference_pixels": int((target & valid).sum())}


def read_namespace(path: Path) -> dict:
    """Parse upstream's Namespace(...) cfg_args without executing its contents."""
    expression = ast.parse(path.read_text(encoding="utf-8"), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("cfg_args must contain Namespace(keyword=value, ...)")
    if expression.func.id != "Namespace" or expression.args or any(k.arg is None for k in expression.keywords):
        raise ValueError("Unsupported cfg_args expression")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords}


def read_cam00(data_root: Path) -> dict:
    import cv2
    intri = cv2.FileStorage(str(data_root / "intri.yml"), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(str(data_root / "extri.yml"), cv2.FILE_STORAGE_READ)
    try:
        if not intri.isOpened() or not extri.isOpened():
            raise FileNotFoundError(f"Missing intri.yml/extri.yml under {data_root}")
        def matrix(storage, name, optional=False):
            node = storage.getNode(name)
            if node.empty():
                if optional:
                    return None
                raise KeyError(name)
            return node.mat()
        K = matrix(intri, "K_cam00").astype(np.float64)
        width = int(intri.getNode("W_cam00").real())
        height = int(intri.getNode("H_cam00").real())
        R = matrix(extri, "Rot_cam00", optional=True)
        if R is None:
            vector = matrix(extri, "R_cam00")
            R = vector if vector.shape == (3, 3) else cv2.Rodrigues(vector)[0]
        T = matrix(extri, "T_cam00").reshape(3)
        distortion = matrix(intri, "D_cam00", optional=True)
        if distortion is None:
            distortion = matrix(intri, "dist_cam00", optional=True)
        if distortion is not None and not np.allclose(distortion, 0.0, atol=1e-8):
            raise ValueError("Evaluation images must be undistorted before RT rendering")
        if width <= 0 or height <= 0 or K.shape != (3, 3):
            raise ValueError("Invalid cam00 dimensions or K")
        if not np.allclose(K[2], [0, 0, 1], atol=1e-7) or abs(K[0, 1]) > 1e-7:
            raise ValueError("RT's camera adapter requires a conventional zero-skew K")
        if not np.allclose([K[0, 2], K[1, 2]], [width / 2, height / 2], atol=1e-3, rtol=0):
            raise ValueError("Principal point is not image-centered; use the same explicit camera adaptation as training")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
            raise ValueError("Invalid world-to-camera rotation")
        return {"K": K, "R": R, "T": T, "width": width, "height": height}
    finally:
        intri.release()
        extri.release()


def select_iteration(model_dir: Path, iteration: int) -> tuple[int, Path]:
    if iteration < 0:
        candidates = []
        for path in (model_dir / "point_cloud").glob("iteration_*"):
            try:
                step = int(path.name.split("_")[-1])
            except ValueError:
                continue
            if all((path / name).is_file() for name in ("point_cloud.ply", "light_mlp.pt", "dir_encoding.pt")):
                candidates.append(step)
        if not candidates:
            raise FileNotFoundError("No complete saved PLY plus shading-module triplet")
        iteration = max(candidates)
    folder = model_dir / "point_cloud" / f"iteration_{iteration}"
    for name in ("point_cloud.ply", "light_mlp.pt", "dir_encoding.pt"):
        if not (folder / name).is_file():
            raise FileNotFoundError(folder / name)
    return iteration, folder


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_saved_gaussians(model, folder: Path, torch) -> None:
    # Upstream <= torch2.0 uses pickle module files. Keep the compatibility
    # adjustment confined to the two named artifacts in this evaluation process.
    original_load = torch.load
    expected = {(folder / name).resolve() for name in ("light_mlp.pt", "dir_encoding.pt")}
    def load_trusted_module(path, *args, **kwargs):
        if Path(path).resolve() not in expected:
            raise ValueError(f"Unexpected pickle path requested: {path}")
        kwargs.update(weights_only=False, map_location="cuda")
        return original_load(path, *args, **kwargs)
    try:
        torch.load = load_trusted_module
        model.load_ply(str(folder / "point_cloud.ply"))
    finally:
        torch.load = original_load
    model.light_mlp.eval()
    model.dir_encoding.eval()


def image_metrics(prediction: np.ndarray, target: np.ndarray, lpips_model, torch, perceptual: bool = True) -> dict:
    # Match the existing GLINT/2DGS score_results.py protocol, including its
    # zero-padding. skimage's cropped valid-window mean would differ at borders.
    def tensor(image):
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0).cuda()
    with torch.no_grad():
        a, b = tensor(prediction), tensor(target)
        x = torch.arange(11, dtype=torch.float32, device=a.device) - 5
        g = torch.exp(-x * x / (2 * 1.5 ** 2))
        g /= g.sum()
        kernel = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
        conv = lambda value: torch.nn.functional.conv2d(value, kernel, padding=5, groups=3)
        ma, mb = conv(a), conv(b)
        va, vb = conv(a * a) - ma * ma, conv(b * b) - mb * mb
        covariance = conv(a * b) - ma * mb
        ssim = ((2 * ma * mb + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
                / ((ma * ma + mb * mb + 0.01 ** 2) * (va + vb + 0.03 ** 2)))
        metrics = {"psnr": psnr(prediction, target), "ssim": float(ssim.mean().item())}
        if perceptual:
            if min(prediction.shape[:2]) < 32:
                raise ValueError("LPIPS comparison requires image/crop sides >= 32 pixels")
            value = float(lpips_model(a, b, normalize=True).item())
            metrics.update(lpips=value, **{f"lpips_{lpips_model.pnet_type}": value})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rt-root", type=Path, default=Path(__file__).resolve().parents[1] / "vendor/RT-Splatting")
    parser.add_argument("--model-dir", "--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=-1, help="Saved PLY iteration; -1 selects latest complete triplet")
    parser.add_argument("--glint-eval-root", "--eval-root", type=Path, required=True, help="Read-only GLINT data/eval root")
    parser.add_argument("--evaluation-regions", "--regions", type=Path, required=True)
    parser.add_argument("--frame", default="000000", help="Frame stem, default same-time frame 0")
    parser.add_argument("--output-dir", "--output", type=Path, required=True)
    parser.add_argument("--lpips-net", choices=("alex", "vgg"), default="alex")
    parser.add_argument("--init-until-iter", type=int, default=0, help="Use the same RT initialization cutoff as training")
    args = parser.parse_args()
    if not args.frame.isdigit():
        parser.error("--frame must be a numeric image stem")
    # Import GPU dependencies only when evaluating, so CPU metric tests stay light.
    sys.path.insert(0, str(args.rt_root.resolve()))
    import torch
    import lpips
    from PIL import Image
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel

    if not torch.cuda.is_available():
        raise RuntimeError("Official RT-Splatting evaluation needs CUDA and its compiled extensions")
    print("Evaluation only: cam00, official final_rendering, no training-mask generation.", flush=True)
    print("LPIPS may download pretrained weights on first use; outputs remain evaluation-only.", flush=True)
    model_dir = args.model_dir.resolve()
    iteration, folder = select_iteration(model_dir, args.iteration)
    config = read_namespace(model_dir / "cfg_args")
    base_parser = argparse.ArgumentParser(add_help=False)
    model_params = ModelParams(base_parser)
    pipeline_params = PipelineParams(base_parser)
    defaults = vars(base_parser.parse_args([]))
    defaults.update(config)
    namespace = SimpleNamespace(**defaults)
    model = GaussianModel(namespace.sh_degree, model_params.extract(namespace))
    load_saved_gaussians(model, folder, torch)
    model.active_sh_degree = min(iteration // 1000, model.max_sh_degree)
    pipe = pipeline_params.extract(namespace)
    pipe.init_stage = iteration < args.init_until_iter
    calibration = read_cam00(args.glint_eval_root)
    K, R, T = (calibration[key] for key in ("K", "R", "T"))
    width, height = calibration["width"], calibration["height"]
    camera = Camera(colmap_id=0, R=R.T.astype(np.float32), T=T.astype(np.float32),
                    FoVx=2 * math.atan(width / (2 * K[0, 0])), FoVy=2 * math.atan(height / (2 * K[1, 1])),
                    image=torch.zeros((3, height, width)), gt_alpha_mask=None,
                    gt_transparent_mask=torch.zeros((1, height, width), dtype=torch.bool),
                    image_name="cam00", uid=0, data_device="cuda")
    background = torch.tensor([1., 1., 1.] if namespace.white_background else [0., 0., 0.], device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        result = render(camera, model, pipe, background)
        predicted = result["final_rendering"].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    torch.cuda.synchronize()
    render_seconds = time.perf_counter() - start

    # Ground truth is opened strictly AFTER rendering, and is never attached to Camera.
    gt_path = args.glint_eval_root / "images" / "cam00" / f"{args.frame}.png"
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64) / 255.0
    if gt.shape != predicted.shape:
        raise ValueError(f"GT/render shape mismatch {gt.shape} != {predicted.shape}; no silent resizing")
    annotation = json.loads(args.evaluation_regions.read_text(encoding="utf-8"))
    if int(annotation.get("frame", -1)) != int(args.frame):
        raise ValueError("Frozen evaluation annotations are for another frame")
    roi = scaled_roi(annotation, width, height)
    perceptual = lpips.LPIPS(net=args.lpips_net, verbose=False).cuda().eval()
    metrics = {"full": image_metrics(predicted, gt, perceptual, torch),
               "window": image_metrics(crop(predicted, roi), crop(gt, roi), perceptual, torch)}
    # Optical opacity and material transmissivity are different RT attributes.
    # A fixed threshold of 1 - surface_opacity is the semantic glass proxy.
    glass_probability = np.clip(1 - result["surface_opacity"].detach().float().cpu().numpy().squeeze(), 0, 1)
    localization = None
    glass_reference = glass_valid = None
    if "glass_positive_polygons" in annotation:
        glass_reference, glass_valid = glass_annotation(annotation, width, height)
        localization = localization_metrics(glass_probability, glass_reference, glass_valid)
        localization.update(source="1 - surface_opacity", scope="fixed annotated window only; ignored boundaries excluded",
                            annotation_kind="manual visible-glass segmentation, not exact optical material or depth ground truth",
                            boundary_ignore_radius=int(annotation.get("boundary_ignore_radius_at_eval_resolution", 0)))
        mask = glass_reference & glass_valid
        metrics["visible_glass_mask_psnr"] = psnr(predicted[mask], gt[mask]) if mask.any() else None
    import cv2
    behind_details = {}
    detail_regions = {}
    rw, rh = annotation["annotation_reference_size"]
    for region in annotation.get("behind_glass_detail_crops", []):
        xyxy = tuple(round(float(v) * s) for v, s in zip(region["xyxy"], (width / rw, height / rh, width / rw, height / rh)))
        pred_crop, gt_crop = crop(predicted, xyxy), crop(gt, xyxy)
        detail = image_metrics(pred_crop, gt_crop, perceptual, torch, perceptual=False)
        def gradient(image):
            image = image.astype(np.float32)
            return np.stack([cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)])
        detail["gradient_mae"] = float(np.abs(gradient(pred_crop) - gradient(gt_crop)).mean())
        detail["xyxy"] = xyxy
        behind_details[region["name"]] = detail
        detail_regions[region["name"]] = xyxy
    output = args.output_dir.resolve()
    if output == args.glint_eval_root.resolve() or args.glint_eval_root.resolve() in output.parents:
        raise ValueError("Evaluation output must not modify the source evaluation dataset")
    output.mkdir(parents=True, exist_ok=True)
    def save_rgb(name, array):
        Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(output / name)
    save_rgb("render.png", predicted)
    save_rgb("ground_truth.png", gt)
    save_rgb("window_render.png", crop(predicted, roi))
    save_rgb("window_ground_truth.png", crop(gt, roi))
    save_rgb("comparison_gt_render.png", np.concatenate((gt, predicted), axis=1))
    save_rgb("window_comparison_gt_render.png", np.concatenate((crop(gt, roi), crop(predicted, roi)), axis=1))
    for name, xyxy in detail_regions.items():
        # Annotation names become file stems, never arbitrary paths.
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        save_rgb(f"detail_{safe_name}_gt_render.png", np.concatenate((crop(gt, xyxy), crop(predicted, xyxy)), axis=1))
    Image.fromarray(np.rint(glass_probability * 255).astype(np.uint8)).save(output / "glass_probability_opacity_complement.png")
    np.save(output / "glass_probability_opacity_complement.npy", glass_probability)
    if glass_reference is not None:
        for name, value in (("glass_prediction", glass_probability >= 0.5),
                            ("glass_reference", glass_reference), ("glass_scoring_valid", glass_valid)):
            np.save(output / f"{name}.npy", value)
            Image.fromarray(crop(value, roi).astype(np.uint8) * 255).save(output / f"{name}_window.png")
        annotation_overlay = gt.copy()
        annotation_overlay[glass_reference] = annotation_overlay[glass_reference] * 0.6 + [0, 0.4, 0]
        x0, y0, x1, y1 = roi
        ignored = np.zeros_like(glass_valid)
        ignored[y0:y1, x0:x1] = ~glass_valid[y0:y1, x0:x1]
        annotation_overlay[ignored] = [1, 1, 0]
        save_rgb("glass_annotation_window.png", crop(annotation_overlay, roi))
    np.save(output / "render_float.npy", predicted)
    np.save(output / "absolute_error.npy", np.abs(predicted.astype(np.float64) - gt).astype(np.float32))
    for key in ("surface_opacity", "transmissivity", "surface_alpha", "surface_depth", "volume_depth"):
        array = result[key].detach().float().cpu().numpy().squeeze()
        np.save(output / f"{key}.npy", array)
        if key in ("surface_opacity", "transmissivity", "surface_alpha"):
            Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(output / f"{key}.png")
    for key in ("final_spec", "final_tran", "final_scat"):
        save_rgb(f"{key}.png", result[key].detach().float().permute(1, 2, 0).cpu().numpy())
    record = {"camera": "cam00", "frame": int(args.frame), "iteration": iteration,
              "width": width, "height": height, "window_xyxy": roi, "metrics": metrics,
              "glass_localization": localization, "behind_glass_details": behind_details,
              "render_seconds_synchronized_including_readback": render_seconds,
              "gaussian_count": int(model.get_xyz.shape[0]), "render_key": "final_rendering",
              "model_dir": str(model_dir), "artifacts": {p.name: sha256(p) for p in folder.iterdir() if p.name in ("point_cloud.ply", "light_mlp.pt", "dir_encoding.pt")},
              "annotation_sha256": sha256(args.evaluation_regions), "ground_truth_sha256": sha256(gt_path),
              "intrinsics": K.tolist(), "world_to_camera_R": R.tolist(), "world_to_camera_T": T.tolist(),
              "metric_protocol": {"psnr": "float64 MSE on full image or cropped ROI, no outside zeros; maximum 120 dB",
                                  "ssim": "RGB mean, 11x11 Gaussian sigma=1.5, zero padding, float32; same as GLINT score_results.py",
                                  "lpips": f"lpips {args.lpips_net}, normalize=True ([0,1] to [-1,1]), crop before evaluation",
                                  "glass_localization": "1-surface_opacity >=0.5; fixed window and manual polygons; boundaries excluded; not material transmissivity",
                                  "behind_glass_details": "fixed crops: PSNR, zero-padded SSIM, RGB 3x3 Sobel gradient MAE; no LPIPS on small crops"},
              "caveats": ["No 3D geometry ground truth; image scores do not establish correct glass depth.",
                          "Independent metrics intentionally avoid upstream zero-padded window PSNR.",
                          "This script does not generate pseudo masks or feed evaluation results into training."]}
    (output / "metrics.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
