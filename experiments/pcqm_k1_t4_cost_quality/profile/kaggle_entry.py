"""Pinned native-T4 profile entry; no Kaggle API or publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Parent preparation replaces these exact markers and retains the entry digest.
EXPECTED_PROFILE_MANIFEST_SHA256 = None
EXPECTED_SETUP_SHA256 = None
EXPECTED_PROFILE_ARCHIVE_SHA256 = None
EXPECTED_UNPACK_SHA256 = None
FORMAT = "molgap-k1-native-t4-profile-payload-v1"

# Execute the complete, separately pinned stdlib extractor in an isolated child.
# It supplies all archive/path handling; do not duplicate its extraction loop.
UNPACK_COMMAND = """
import hashlib, json, pathlib, sys, types
path = pathlib.Path(sys.argv[1])
code = path.read_bytes()
if hashlib.sha256(code).hexdigest() != sys.argv[2]:
    raise ValueError('Frozen extractor changed before execution')
module = types.ModuleType('_frozen_profile_unpack')
module.__file__ = str(path)
exec(compile(code, module.__file__, 'exec'), module.__dict__)
root = pathlib.Path(sys.argv[4])
module._unpack(pathlib.Path(sys.argv[3]), root)
manifest_path = root / 'payload_manifest.json'
if module._file_sha(manifest_path) != sys.argv[5]:
    raise ValueError('Outer/inner profile manifest differs')
manifest = json.loads(manifest_path.read_text())
for name, expected in manifest['files'].items():
    if module._file_sha(module._regular(module._under(root, name))) != expected:
        raise ValueError('Extracted profile payload bytes differ: ' + name)
"""


def main():
    allocation_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/profile"))
    parser.add_argument("--temp-root", type=Path, default=Path("/kaggle/temp"))
    args = parser.parse_args()
    status, count, names, bootstrap_wall = "incomplete", None, [], None
    manifest_digest, setup_digest, archive_digest, unpack_digest = None, None, None, None
    def remaining():
        seconds = 1200 - (time.perf_counter()-allocation_started) - 2
        if seconds <= 0:
            raise TimeoutError("Entry allocation ceiling exhausted")
        return seconds
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()
    try:
        if EXPECTED_PROFILE_MANIFEST_SHA256 is None or EXPECTED_SETUP_SHA256 is None:
            raise ValueError("Parent-pinned manifest/setup required")
        candidates = []
        for path in args.input_root.rglob("payload_manifest.json"):
            remaining()
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and value.get("format") == FORMAT:
                candidates.append((path, value))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one native-T4 profile payload")
        manifest_path, manifest = candidates[0]
        root = manifest_path.parent.resolve()
        manifest_digest = digest(manifest_path)
        if manifest_digest != EXPECTED_PROFILE_MANIFEST_SHA256:
            raise ValueError("Profile manifest differs from pinned entry")
        archive = root / "source_payload.bin"
        if archive.exists():
            if EXPECTED_PROFILE_ARCHIVE_SHA256 is None or EXPECTED_UNPACK_SHA256 is None:
                raise ValueError("Parent-pinned archive/extractor required")
            extractor = root / "unpack.py"
            archive_digest, unpack_digest = digest(archive), digest(extractor)
            if archive.is_symlink() or extractor.is_symlink() or archive_digest != EXPECTED_PROFILE_ARCHIVE_SHA256:
                raise ValueError("Frozen profile archive differs")
            if unpack_digest != EXPECTED_UNPACK_SHA256:
                raise ValueError("Frozen profile extractor differs")
            args.temp_root.mkdir(parents=True, exist_ok=True)
            package = args.temp_root / "profile-package"
            payload = args.temp_root / "profile-payload"
            if package.exists() or payload.exists():
                raise FileExistsError("Extraction requires fresh profile package/payload")
            package.mkdir()
            payload.mkdir()
            shutil.copyfile(archive, package / "source.tar.gz")
            if digest(package / "source.tar.gz") != archive_digest:
                raise ValueError("Copied profile archive differs")
            subprocess.run([sys.executable, "-I", "-S", "-c", UNPACK_COMMAND,
                            str(extractor), unpack_digest, str(package), str(payload), manifest_digest],
                           check=True, timeout=remaining())
            remaining()
            inner_manifest = payload / "payload_manifest.json"
            if digest(inner_manifest) != manifest_digest or inner_manifest.read_bytes() != manifest_path.read_bytes():
                raise ValueError("Outer/inner profile manifest differs")
            root = payload.resolve()
        elif EXPECTED_PROFILE_ARCHIVE_SHA256 is not None or EXPECTED_UNPACK_SHA256 is not None:
            raise ValueError("Pinned flat profile archive is missing")
        for name in ("setup.sh", "bootstrap.py", "run.sh"):
            path = root / name
            if path.is_symlink() or digest(path) != manifest["files"][name]:
                raise ValueError(f"Frozen bootstrap/setup differs: {name}")
        setup_digest = digest(root / "setup.sh")
        if setup_digest != EXPECTED_SETUP_SHA256:
            raise ValueError("Setup differs from pinned entry")
        observed = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                  check=True, capture_output=True, text=True, timeout=remaining())
        names = [line.strip() for line in observed.stdout.splitlines() if line.strip()]
        count = len(names)
        if count != 2 or any("T4" not in name for name in names):
            raise ValueError("Observed two native T4 devices required")
        env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONNOUSERSITE="1",
                   PYTHONDONTWRITEBYTECODE="1")
        budget = int(remaining())
        bootstrap_started = time.perf_counter()
        try:
            subprocess.run(["timeout", "--signal=KILL", f"{budget}s", sys.executable,
                str(root / "bootstrap.py"), "--root", str(root), "--output", str(args.output.resolve()),
                "--expected-manifest-sha256", manifest_digest, "--setup", str(root / "setup.sh"),
                "--setup-sha256", setup_digest, "--wall-budget-seconds", str(budget)], env=env, check=True)
        finally:
            bootstrap_wall = time.perf_counter() - bootstrap_started
        remaining()
        status = "complete"
    finally:
        elapsed = time.perf_counter() - allocation_started
        report = {"format": "molgap-k1-t4-entry-observation-v1", "status": status,
                  "entry_wall_seconds_including_verify_setup_bootstrap": elapsed,
                  "allocation_ceiling_seconds": 1200, "observed_allocated_gpu_count": count,
                  "observed_gpu_names": names,
                  "allocated_T4_device_hours": elapsed * count / 3600
                      if count is not None and names and all("T4" in name for name in names) else None,
                  "bootstrap_wall_seconds_including_setup": bootstrap_wall,
                  "payload_manifest_sha256": manifest_digest, "setup_sha256": setup_digest,
                  "source_archive_sha256": archive_digest, "unpack_sha256": unpack_digest,
                  "gpu_busy_seconds": None, "scientific_outcome": "NO_TRAIN"}
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "entry_observation.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
