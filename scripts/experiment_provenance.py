"""Fail-closed provenance guards for a sequential experiment directory.

These checks do not select parameters or change model execution. Existing
artifacts are never adopted merely because their names happen to match.
"""
import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def immutable_json(path, value):
    """Create a record once; stop instead of relabeling an existing experiment."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Immutable experiment record differs: {path}; preserve this run and review explicitly')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require_teacher(manifest, expected_sha256):
    if manifest.get('method') != 'glint' or manifest.get('checkpoint_sha256') != expected_sha256:
        raise ValueError('Mask teacher does not match this experiment checkpoint')


def require_copied_masks(data, masks):
    """Validate the files RT will read, not only the original export directory."""
    data, masks = Path(data), Path(masks)
    for directory in ('transparent_masks', 'mask_valid'):
        actual = {p.name: p for p in (data / directory).glob('*.png')}
        reference = {p.name: p for p in (masks / directory).glob('*.png')}
        if not reference or actual.keys() != reference.keys():
            raise ValueError(f'Packaged {directory} inventory differs from shared masks')
        for name, path in actual.items():
            if file_sha256(path) != file_sha256(reference[name]):
                raise ValueError(f'Packaged mask differs from shared teacher export: {directory}/{name}')


def artifact_hashes(paths):
    hashes = {}
    for path in map(Path, paths):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'Missing or empty completion artifact: {path}')
        hashes[str(path)] = file_sha256(path)
    if not hashes:
        raise ValueError('A completion record must name its output artifacts')
    return hashes


def completed(path, expected_signature, artifacts):
    path = Path(path)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f'Unverified legacy or malformed completion marker: {path}') from error
    if not isinstance(record, dict) or record.get('signature') != expected_signature:
        raise RuntimeError(f'Completion belongs to a different or unverified experiment: {path}')
    if record.get('artifacts') != artifact_hashes(artifacts):
        raise RuntimeError(f'Completed output changed: {path}')
    return True


def record_completion(path, expected_signature, artifacts):
    immutable_json(path, {'schema_version': 1, 'signature': expected_signature,
                          'artifacts': artifact_hashes(artifacts)})


def require_free_disk(path, minimum_bytes):
    """A preflight floor, not an estimate of all future checkpoint storage."""
    if minimum_bytes <= 0:
        raise ValueError('Minimum free disk space must be positive')
    free = shutil.disk_usage(path).free
    if free < minimum_bytes:
        raise RuntimeError(f'Insufficient free disk space before training: {free} bytes < {minimum_bytes} bytes; preserve outputs and free space explicitly')
    return free


def require_supported_config(config):
    """Do not record configurable seeds/splits that the wrappers do not apply."""
    fixed = {'seed': 0, 'view_sequence_seed': 20261015, 'held_out_camera': 'cam00',
             'scene': 'coffee_martini', 'frame': 0, 'resolution': [676, 507]}
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f'Unsupported experiment config {key}={config.get(key)!r}; implemented value is {expected!r}')
