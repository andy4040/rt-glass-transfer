"""Package existing train-only calibrated N3DV images for RT-Splatting.

Training source is the previous data/2dgs_train directory. Only its RGB images
and sparse COLMAP calibration/points are used. No source masks are copied.
Evaluation metadata is written separately; no cam00 RGB or dummy mask enters
the training dataset. Run official RT training WITHOUT --eval and WITH -r 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

try:
    from .export_masks import check_train_names
    from .experiment_provenance import require_copied_masks, require_teacher
except ImportError:
    from export_masks import check_train_names
    from experiment_provenance import require_copied_masks, require_teacher


def read_colmap_text(sparse):
    cameras = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split()
        identifier, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
        params = list(map(float, p[4:]))
        if model != "PINHOLE" or len(params) != 4 or w <= 0 or h <= 0:
            raise ValueError("Expected calibrated PINHOLE cameras")
        fx, fy, cx, cy = params
        if fx <= 0 or fy <= 0 or not np.isfinite(params).all():
            raise ValueError("Invalid intrinsics")
        if not np.allclose([cx, cy], [w / 2, h / 2], rtol=0, atol=1e-5):
            raise ValueError("RT-Splatting assumes a centered principal point")
        if identifier in cameras:
            raise ValueError("Duplicate COLMAP camera id")
        cameras[identifier] = {"id": identifier, "width": w, "height": h, "params": params}
    images, rows = [], iter((sparse / "images.txt").read_text().splitlines())
    for line in rows:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split()
        if len(p) != 10:
            raise ValueError("Invalid COLMAP image header")
        name, camera_id = p[9], int(p[8])
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError("Expected flat camXX.png COLMAP image names")
        if camera_id not in cameras:
            raise ValueError("Image references missing camera calibration")
        pose = np.array(list(map(float, p[1:8])))
        if not np.isfinite(pose).all() or abs(np.linalg.norm(pose[:4]) - 1) > 1e-4:
            raise ValueError("Invalid COLMAP world-to-camera pose")
        images.append({"id": int(p[0]), "name": name, "camera": Path(name).stem,
                       "camera_id": camera_id, "qvec_wxyz": pose[:4].tolist(), "T": pose[4:].tolist()})
        if next(rows, None) is None:
            raise ValueError("COLMAP image header is missing its observations line")
    check_train_names([row["camera"] for row in images])
    if len({r["id"] for r in images}) != len(images):
        raise ValueError("Duplicate COLMAP image id")
    if {row["camera_id"] for row in images} != set(cameras):
        raise ValueError("Unreferenced camera metadata in training source")
    return cameras, images


def validate_inputs(source, masks, *, expected_teacher_sha256=None, require_packaged_masks=False):
    sparse = source / "sparse" / "0"
    cameras, images = read_colmap_text(sparse)
    names = {row["camera"] for row in images}
    actual_images = {p.stem for p in (source / "images").glob("*.png")}
    if actual_images != names:
        raise ValueError("Training RGB inventory differs from calibration (possible held-out leakage)")
    for directory in ("transparent_masks", "mask_valid"):
        actual = {p.stem for p in (masks / directory).glob("*.png")}
        if actual != names:
            raise ValueError(f"{directory} inventory differs from training cameras")
    manifest = json.loads((masks / "manifest.json").read_text())
    if expected_teacher_sha256 is not None:
        require_teacher(manifest, expected_teacher_sha256)
    if set(manifest["train_cameras"]) != names or manifest.get("held_out_used") is not False:
        raise ValueError("Mask provenance does not establish train-only generation")
    for row in images:
        camera = cameras[row["camera_id"]]
        size = camera["width"], camera["height"]
        with Image.open(source / "images" / row["name"]) as image:
            if image.size != size or image.mode != "RGB":
                raise ValueError(f"RGB geometry/mode mismatch: {row['name']}")
        values = {}
        for directory in ("transparent_masks", "mask_valid"):
            with Image.open(masks / directory / row["name"]) as image:
                if image.size != size or image.mode != "L":
                    raise ValueError(f"Mask geometry/mode mismatch: {row['name']}")
                values[directory] = np.asarray(image)
                if not np.isin(values[directory], [0, 255]).all():
                    raise ValueError("Masks must be strictly binary uint8, not soft probabilities")
        if not (values["mask_valid"] > 0).any():
            raise ValueError(f"No valid supervision: {row['name']}")
    for name in ("points3D.ply", "points3D.txt"):
        if not (sparse / name).is_file():
            raise FileNotFoundError(sparse / name)
    if require_packaged_masks:
        require_copied_masks(source, masks)
    return cameras, images, manifest


def link_or_copy(source, target, mode):
    if mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def export_eval_metadata(root, output):
    """Only cam00 calibration is read; no held-out image/prior/mask is generated."""
    import cv2
    intri = cv2.FileStorage(str(root / "intri.yml"), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(str(root / "extri.yml"), cv2.FILE_STORAGE_READ)
    if not intri.isOpened() or not extri.isOpened():
        intri.release(); extri.release()
        raise FileNotFoundError("Evaluation calibration is missing")
    try:
        names = intri.getNode("names")
        if [names.at(i).string() for i in range(names.size())] != ["cam00"]:
            raise ValueError("Expected an isolated cam00 evaluation root")
        K = intri.getNode("K_cam00").mat()
        R = extri.getNode("Rot_cam00").mat()
        if R is None:
            R = cv2.Rodrigues(extri.getNode("R_cam00").mat())[0]
        T = extri.getNode("T_cam00").mat()
        w, h = int(intri.getNode("W_cam00").real()), int(intri.getNode("H_cam00").real())
        if not np.allclose([K[0, 2], K[1, 2]], [w / 2, h / 2]):
            raise ValueError("RT evaluation assumes centered principal point")
        record = {"name": "cam00", "width": w, "height": h, "K": K.tolist(),
                  "R_world_to_camera": R.tolist(), "T_world_to_camera": T.reshape(3).tolist(),
                  "convention": "OpenCV/COLMAP right/down/forward; RT Camera R must receive R.T",
                  "rgb_path": str((root / "images" / "cam00" / "000000.png").resolve()),
                  "rgb_read": False, "mask_generated": False, "split": "evaluation_only"}
        output.mkdir(parents=True, exist_ok=False)
        (output / "camera.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finally:
        intri.release(); extri.release()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path, help="Existing train-only data/2dgs_train")
    p.add_argument("--masks", required=True, type=Path, help="export_masks.py output")
    p.add_argument("--output", required=True, type=Path, help="New RT training dataset")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    p.add_argument("--eval-root", type=Path)
    p.add_argument("--eval-output", type=Path, help="Separate sibling directory, not inside training output")
    args = p.parse_args(argv)
    if bool(args.eval_root) != bool(args.eval_output):
        p.error("--eval-root and --eval-output must be supplied together")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if args.eval_output and (args.eval_output.resolve() == args.output.resolve()
                            or args.output.resolve() in args.eval_output.resolve().parents):
        raise ValueError("Evaluation metadata must be outside the training dataset")
    cameras, images, mask_manifest = validate_inputs(args.source, args.masks)
    for directory in ("images", "transparent_masks", "mask_valid", "sparse/0"):
        (args.output / directory).mkdir(parents=True)
    for row in images:
        link_or_copy(args.source / "images" / row["name"], args.output / "images" / row["name"], args.mode)
        for directory in ("transparent_masks", "mask_valid"):
            # Copy labels rather than linking: teacher data is immutable per run.
            shutil.copy2(args.masks / directory / row["name"], args.output / directory / row["name"])
    for name in ("cameras.txt", "images.txt", "points3D.txt", "points3D.ply"):
        shutil.copy2(args.source / "sparse" / "0" / name, args.output / "sparse" / "0" / name)
    record = {"schema_version": 1, "source": str(args.source.resolve()), "frame": 0,
              "train_cameras": [r["camera"] for r in images], "test_camera": "cam00",
              "test_rgb_in_training_dataset": False, "test_mask_generated": False,
              "rt_resolution_argument": 1, "rt_eval_argument": False,
              "mask_method": mask_manifest["method"], "mask_manifest": mask_manifest,
              "camera_convention": "Unchanged COLMAP world-to-camera poses, no recenter or scale change",
              "cameras": list(cameras.values()), "images": images}
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.eval_root:
        export_eval_metadata(args.eval_root, args.eval_output)
    print(json.dumps({key: record[key] for key in ("train_cameras", "mask_method", "test_rgb_in_training_dataset")}, indent=2))
    return record


if __name__ == "__main__":
    main()
