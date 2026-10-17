"""Prepare calibrated Neural 3D Video data without modifying source files.

Pose rows follow the lexically sorted *existing* camXX video names, as in the
upstream N3DV loader. A missing numeric camera ID does not consume a pose row.
This convention is recorded, but cannot be independently inferred from an NPY.
"""

import argparse
from contextlib import closing
import hashlib
import io
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np


CAMERA_VIDEO = re.compile(r"cam\d+\.mp4$")
MANIFEST_NAME = "preparation_manifest.json"


def camera_pose_mapping(names, poses):
    names = sorted(names)
    if len(set(names)) != len(names) or not names:
        raise ValueError("Camera names must be unique and nonempty")
    if any(not CAMERA_VIDEO.fullmatch(name) for name in names):
        raise ValueError("Expected camXX.mp4 camera names")
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(names), 17):
        raise ValueError("Expected poses_bounds shape (%d, 17), got %s" % (len(names), poses.shape))
    if not np.isfinite(poses).all():
        raise ValueError("poses_bounds contains non-finite values")
    hwf = poses[:, :15].reshape(-1, 3, 5)[:, :, 4]
    if np.any(hwf <= 0) or np.any(poses[:, 15] <= 0) or np.any(poses[:, 16] <= poses[:, 15]):
        raise ValueError("Invalid camera dimensions, focal length, or near/far bounds")
    if not np.allclose(hwf[:, :2], np.round(hwf[:, :2])):
        raise ValueError("Camera height and width must be integers")
    return [(name[:-4], row) for row, name in enumerate(names)]


def resize_poses(poses, downscale):
    """Use exact uniform scaling; reject rounding that would require fx != fy."""
    if not isinstance(downscale, int) or downscale < 1:
        raise ValueError("downscale must be a positive integer")
    result = np.asarray(poses, dtype=np.float64).copy()
    hwf = result[:, :15].reshape(-1, 3, 5)[:, :, 4]
    if np.any(np.remainder(hwf[:, :2], downscale) != 0):
        raise ValueError("Height and width must be divisible by downscale to preserve LLFF's single focal length")
    hwf /= downscale
    return result


def extract_frames(video_path, output_dir, frame_count, downscale, expected_hwf, reuse_directory=None):
    import cv2

    if frame_count < 1:
        raise ValueError("frames must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video: %s" % video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if (height, width) != tuple(int(value) for value in expected_hwf[:2]):
            raise ValueError("Video dimensions do not match supplied poses: %s" % video_path)
        if not math.isfinite(fps) or fps <= 0 or reported_count < frame_count:
            raise ValueError("Invalid fps or video shorter than requested %d frames: %s" % (frame_count, video_path))
        resized_width, resized_height = width // downscale, height // downscale
        expected_names = ["%06d.png" % index for index in range(frame_count)]
        reuse = (reuse_directory is not None and Path(reuse_directory).is_dir()
                 and sorted(p.name for p in Path(reuse_directory).iterdir()) == expected_names)
        for frame_index in range(frame_count):
            if reuse:
                cached = Path(reuse_directory) / ("%06d.png" % frame_index)
                frame = cv2.imread(str(cached), cv2.IMREAD_COLOR)
                if frame is None or frame.shape != (resized_height, resized_width, 3):
                    raise ValueError("Invalid reusable frame: %s" % cached)
                shutil.copy2(cached, output_dir / cached.name)
                continue
            success, frame = capture.read()
            if not success or frame is None:
                raise ValueError("Failed decoding frame %d in %s" % (frame_index, video_path))
            if frame.shape[:2] != (height, width):
                raise ValueError("Video dimensions changed at frame %d" % frame_index)
            if downscale != 1:
                frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            target = output_dir / ("%06d.png" % frame_index)
            if not cv2.imwrite(str(target), frame):
                raise OSError("Failed writing frame: %s" % target)
        expected = ["%06d.png" % index for index in range(frame_count)]
        if sorted(path.name for path in output_dir.iterdir()) != expected:
            raise ValueError("Extracted frame sequence is not complete and contiguous")
        return {"width": width, "height": height, "fps": fps,
                "reported_frame_count": reported_count, "decoded_frame_count": frame_count,
                "output_width": resized_width, "output_height": resized_height,
                "reused_verified_frames": reuse}
    finally:
        capture.release()


def _source_description(source):
    """Read only expected archive members; never extract archive paths verbatim."""
    if source.is_dir():
        pose_path = source / "poses_bounds.npy"
        if not pose_path.is_file():
            raise ValueError("Input directory must directly contain poses_bounds.npy and camXX.mp4")
        names = sorted(path.name for path in source.iterdir() if path.is_file() and CAMERA_VIDEO.fullmatch(path.name))
        return pose_path.read_bytes(), names, None
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("Input must be a video directory or a ZIP archive")
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        pose_members = [item for item in members if PurePosixPath(item.filename).name == "poses_bounds.npy"]
        if len(pose_members) != 1:
            raise ValueError("ZIP must contain exactly one poses_bounds.npy")
        parent = PurePosixPath(pose_members[0].filename).parent
        video_members = [item for item in members
                         if PurePosixPath(item.filename).parent == parent
                         and CAMERA_VIDEO.fullmatch(PurePosixPath(item.filename).name)]
        names = sorted(PurePosixPath(item.filename).name for item in video_members)
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate camera members")
        member_map = {PurePosixPath(item.filename).name: item.filename for item in video_members}
        return archive.read(pose_members[0]), names, member_map


def prepare_dataset(source, output, frames=300, downscale=2, colmap=False, colmap_binary="colmap", use_gpu=False, reuse_staging=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory: %s" % output)
    if source == output or (source.is_dir() and source in output.parents):
        raise ValueError("Output must be outside the input dataset")
    if frames < 1:
        raise ValueError("frames must be positive")
    pose_bytes, video_names, zip_members = _source_description(source)
    original_poses = np.load(io.BytesIO(pose_bytes), allow_pickle=False)
    mapping = camera_pose_mapping(video_names, original_poses)
    scaled_poses = resize_poses(original_poses, downscale)
    if reuse_staging is not None:
        reuse_staging = Path(reuse_staging).resolve()
        if (reuse_staging / 'poses_bounds_original.npy').read_bytes() != pose_bytes:
            raise ValueError('Reusable staging calibration differs from source')
        if not np.array_equal(np.load(reuse_staging / 'poses_bounds.npy'), scaled_poses):
            raise ValueError('Reusable staging resolution differs from requested resolution')
    if "cam00" not in [name for name, _ in mapping]:
        raise ValueError("Expected held-out camera cam00 for the standard N3DV split")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".%s.prepare-" % output.name, dir=str(output.parent)))
    try:
        # Keep provenance of the original calibration; the loader reads resized calibration.
        (stage / "poses_bounds_original.npy").write_bytes(pose_bytes)
        np.save(stage / "poses_bounds.npy", scaled_poses)
        original_hwf = original_poses[:, :15].reshape(-1, 3, 5)[:, :, 4]
        output_hwf = scaled_poses[:, :15].reshape(-1, 3, 5)[:, :, 4]
        manifest = {"schema_version": 1, "source_name": source.name,
                    "poses_sha256": hashlib.sha256(pose_bytes).hexdigest(),
                    "pose_order_convention": "lexically sorted existing camXX.mp4 filenames; not numeric ID indexing",
                    "pose_alignment_independently_verified": False,
                    "frames": frames, "downscale": downscale, "test_camera": "cam00",
                    "camera_order": [name for name, _ in mapping], "cameras": []}
        for camera, row in mapping:
            filename = camera + ".mp4"
            video = source / filename
            if zip_members is not None:
                video = stage / "_source_video.mp4"
                with zipfile.ZipFile(source) as archive, archive.open(zip_members[filename]) as src, video.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            metadata = extract_frames(video, stage / camera, frames, downscale, original_hwf[row],
                                      reuse_directory=reuse_staging / camera if reuse_staging else None)
            if zip_members is not None:
                video.unlink()
            manifest["cameras"].append({"camera": camera, "pose_row": row, "source_video": filename,
                                         "split": "test" if camera == "cam00" else "train",
                                         "original_hwf": original_hwf[row].tolist(),
                                         "output_hwf": output_hwf[row].tolist(), "video": metadata})
            print("Prepared %s: %d frames" % (camera, frames), flush=True)
        videos = [entry["video"] for entry in manifest["cameras"]]
        if any(abs(video["fps"] - videos[0]["fps"]) > 1e-3 for video in videos):
            raise ValueError("Camera videos have inconsistent frame rates")
        if len({video["reported_frame_count"] for video in videos}) != 1:
            raise ValueError("Camera videos have inconsistent reported frame counts")
        (stage / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if colmap:
            prepare_colmap(stage, colmap_binary, use_gpu)
        if output.exists():
            raise FileExistsError("Output was created while preparation was running: %s" % output)
        stage.rename(output)
        return output
    except Exception as error:
        raise RuntimeError("Preparation failed; original files unchanged. Partial staging directory retained at %s: %s" % (stage, error)) from error


def _calibration_records(poses, camera_order):
    # Same LLFF down/right/back -> COLMAP right/down/forward conversion as
    # dataset_utils.etc_utils.posetow2c_matrcs, with no recentering or scale change.
    project = str(Path(__file__).resolve().parents[1])
    if project not in sys.path:
        sys.path.insert(0, project)
    from dataset_utils.etc_utils import posetow2c_matrcs, rotmat2qvec

    matrices = posetow2c_matrcs(poses[:, :15].reshape(-1, 3, 5).transpose(1, 2, 0))
    hwfs = poses[:, :15].reshape(-1, 3, 5)[:, :, 4]
    records = []
    for index, (camera, matrix, hwf) in enumerate(zip(camera_order, matrices, hwfs)):
        h, w, focal = hwf
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("Invalid calibrated rotation for %s" % camera)
        records.append({"id": index + 1, "camera": camera, "name": camera + ".png",
                        "width": int(w), "height": int(h), "params": [float(focal), float(focal), float(w / 2), float(h / 2)],
                        "qvec": rotmat2qvec(rotation).tolist(), "tvec": matrix[:3, 3].tolist()})
    return records


def _camera_line(record):
    return "%d PINHOLE %d %d %s\n" % (record["id"], record["width"], record["height"], " ".join("%.17g" % number for number in record["params"]))


def _image_line(record):
    numbers = record["qvec"] + record["tvec"]
    return "%d %s %d %s\n" % (record["id"], " ".join("%.17g" % number for number in numbers), record["id"], record["name"])


def _write_manual_model(path, records):
    path.mkdir()
    (path / "cameras.txt").write_text("".join(_camera_line(record) for record in records), encoding="utf-8")
    (path / "images.txt").write_text("".join(_image_line(record) + "\n" for record in records), encoding="utf-8")
    (path / "points3D.txt").write_text("", encoding="utf-8")


def _seed_database(path, records):
    # COLMAP fills its remaining feature/match tables when feature_extractor runs.
    with closing(sqlite3.connect(str(path))) as database:
        database.executescript("""
            CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
                params BLOB, prior_focal_length INTEGER NOT NULL);
            CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL,
                prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
                prior_tx REAL, prior_ty REAL, prior_tz REAL,
                CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
            CREATE UNIQUE INDEX index_name ON images(name);
        """)
        for record in records:
            database.execute("INSERT INTO cameras VALUES (?, 1, ?, ?, ?, 1)",
                             (record["id"], record["width"], record["height"], np.asarray(record["params"], dtype=np.float64).tobytes()))
            database.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (record["id"], record["name"], record["id"], *record["qvec"], *record["tvec"]))
        database.commit()


def _verify_fixed_calibration(model, records):
    expected = {record["id"]: record for record in records}
    seen = set()
    for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        record = expected[int(parts[0])]
        if parts[1] != "PINHOLE" or tuple(map(int, parts[2:4])) != (record["width"], record["height"]):
            raise ValueError("COLMAP changed the calibrated camera model or image dimensions")
        if not np.allclose(np.asarray(parts[4:], dtype=float), record["params"], atol=1e-6, rtol=1e-9):
            raise ValueError("COLMAP changed supplied intrinsics; refusing incompatible initialization")
        seen.add(int(parts[0]))
    if seen != set(expected):
        raise ValueError("COLMAP output is missing calibrated training cameras")
    lines = iter((model / "images.txt").read_text(encoding="utf-8").splitlines())
    seen = set()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        record = expected[int(parts[0])]
        qvec = np.asarray(parts[1:5], dtype=float)
        if np.dot(qvec, record["qvec"]) < 0:
            qvec = -qvec
        if (not np.allclose(qvec, record["qvec"], atol=1e-6, rtol=1e-9)
                or not np.allclose(np.asarray(parts[5:8], dtype=float), record["tvec"], atol=1e-6, rtol=1e-9)
                or int(parts[8]) != record["id"] or parts[9] != record["name"]):
            raise ValueError("COLMAP changed supplied camera poses; refusing incompatible initialization")
        seen.add(int(parts[0]))
        next(lines, None)  # The following (possibly empty) line contains feature tracks.
    if seen != set(expected):
        raise ValueError("COLMAP output is missing calibrated training images")
    points = [line for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not points:
        raise ValueError("COLMAP triangulated no points; inspect camera mapping and feature matches")
    return len(points)


def prepare_colmap(dataset, colmap_binary="colmap", use_gpu=False):
    """Triangulate training-camera frame 0 while keeping cam00 RGB held out."""
    import cv2

    dataset = Path(dataset).resolve()
    target = dataset / "colmap_0"
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing COLMAP data: %s" % target)
    manifest = json.loads((dataset / MANIFEST_NAME).read_text(encoding="utf-8"))
    poses = np.load(dataset / "poses_bounds.npy", allow_pickle=False)
    order = manifest["camera_order"]
    mapping = camera_pose_mapping([name + ".mp4" for name in order], poses)
    if [name for name, _ in mapping] != order or manifest["test_camera"] != "cam00":
        raise ValueError("Manifest camera order or split is inconsistent")
    records = _calibration_records(poses, order)
    train_records = [record for record in records if record["camera"] != "cam00"]
    test_records = [record for record in records if record["camera"] == "cam00"]
    if len(train_records) < 2 or len(test_records) != 1:
        raise ValueError("Need cam00 plus at least two training cameras")
    for record in train_records:
        frame = cv2.imread(str(dataset / record["camera"] / "000000.png"), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (record["height"], record["width"]):
            raise ValueError("First-frame image is missing or inconsistent with poses: %s" % record["camera"])
    binary = shutil.which(colmap_binary)
    if binary is None:
        raise FileNotFoundError("COLMAP executable not found: %s" % colmap_binary)
    help_result = subprocess.run([binary, "point_triangulator", "-h"], check=True, capture_output=True, text=True)
    triangulator_help = help_result.stdout + help_result.stderr
    fixed_flags = []
    if "Mapper.ba_refine_focal_length" in triangulator_help:
        for name in ("focal_length", "principal_point", "extra_params"):
            fixed_flags += ["--Mapper.ba_refine_" + name, "0"]
    elif "refine_intrinsics" in triangulator_help:
        fixed_flags += ["--refine_intrinsics", "0"]
    else:
        raise RuntimeError("This COLMAP point_triangulator exposes no supported fixed-intrinsics option")
    if "Mapper.fix_existing_images" in triangulator_help:
        fixed_flags += ["--Mapper.fix_existing_images", "1"]
    stage = Path(tempfile.mkdtemp(prefix=".colmap_0.prepare-", dir=str(dataset)))
    commands = []
    try:
        image_dir = stage / "input"
        image_dir.mkdir()
        for record in train_records:
            shutil.copy2(dataset / record["camera"] / "000000.png", image_dir / record["name"])
        manual = stage / "manual"
        _write_manual_model(manual, train_records)
        database = stage / "input.db"
        _seed_database(database, train_records)
        sparse = stage / "triangulated"
        sparse.mkdir()
        export = stage / "sparse" / "0"
        export.mkdir(parents=True)
        commands = [
            [binary, "feature_extractor", "--database_path", str(database), "--image_path", str(image_dir), "--SiftExtraction.use_gpu", str(int(use_gpu))],
            [binary, "exhaustive_matcher", "--database_path", str(database), "--SiftMatching.use_gpu", str(int(use_gpu))],
            [binary, "point_triangulator", "--database_path", str(database), "--image_path", str(image_dir), "--input_path", str(manual), "--output_path", str(sparse)] + fixed_flags,
            [binary, "model_converter", "--input_path", str(sparse), "--output_path", str(export), "--output_type", "TXT"],
        ]
        for command in commands:
            print("Running COLMAP %s" % command[1], flush=True)
            subprocess.run(command, check=True)
        point_count = _verify_fixed_calibration(export, train_records)
        # Add the supplied test calibration, without its RGB contributing points.
        with (export / "cameras.txt").open("a", encoding="utf-8") as handle:
            for record in test_records:
                handle.write(_camera_line(record))
        with (export / "images.txt").open("a", encoding="utf-8") as handle:
            for record in test_records:
                handle.write(_image_line(record) + "\n")
        (stage / "initialization_manifest.json").write_text(json.dumps({
            "frame": 0, "train_cameras": [record["camera"] for record in train_records],
            "held_out_cameras": ["cam00"], "held_out_rgb_used": False,
            "supplied_calibration_verified": True, "point_count": point_count,
            "commands": commands}, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError("COLMAP target appeared during preparation: %s" % target)
        stage.rename(target)
        return target
    except Exception as error:
        raise RuntimeError("COLMAP preparation failed; staging retained at %s: %s" % (stage, error)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Raw N3DV ZIP or directory")
    parser.add_argument("--output", required=True, type=Path, help="New dataset directory, or prepared dataset with --colmap-only")
    parser.add_argument("--frames", "--max-frames", type=int, default=300, help="Extract exactly this many frames from index 0 (default: 300)")
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--colmap", action="store_true", help="Also triangulate first-frame training-camera points")
    parser.add_argument("--colmap-only", action="store_true", help="Add colmap_0 to a prepared dataset; original videos are not needed")
    parser.add_argument("--colmap-binary", default="colmap")
    parser.add_argument("--colmap-gpu", action="store_true", help="Enable GPU SIFT; default is CPU")
    parser.add_argument("--reuse-staging", type=Path, help="Reuse verified complete camera directories from this task's interrupted extraction; originals are preserved")
    args = parser.parse_args(argv)
    if args.colmap_only:
        if args.input is not None or args.colmap:
            parser.error("--colmap-only uses only --output, not --input or --colmap")
        result = prepare_colmap(args.output, args.colmap_binary, args.colmap_gpu)
    else:
        if args.input is None:
            parser.error("--input is required unless --colmap-only is used")
        result = prepare_dataset(args.input, args.output, args.frames, args.downscale, args.colmap, args.colmap_binary, args.colmap_gpu, args.reuse_staging)
    print("Prepared: %s" % result)


if __name__ == "__main__":
    main()
