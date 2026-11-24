"""Kaggle entrypoint for Single-Point Substitution DeltaGap Zero-Training Diagnostic."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

# Embedded SHA256 bindings populated and verified before upload.
EXPECTED_BINDINGS_SHA256 = "7fe5ad8ad133b2534eb1afc352b7b8a008f8ba5ec46d1dd3d67824fb0fe8eb45"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/temp/delta-gap-diag")
OUTPUT_ROOT = Path("/kaggle/working/delta_gap_diag")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".alloc-", delete=False) as f:
            tmp = Path(f.name)
            json.dump(value, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _find_unique_file(root: Path, filename: str, expected_sha: str | None = None) -> Path:
    cands = [p for p in root.rglob(filename) if p.is_file() and "checkpoint_chunks" not in p.parts]
    if expected_sha is not None:
        cands = [p for p in cands if _sha256_file(p) == expected_sha]
    if len(cands) != 1:
        raise RuntimeError(f"Expected exactly 1 '{filename}' under {root}, found {len(cands)}: {[str(c) for c in cands]}")
    return cands[0]


def _unpack_molgap_source(input_root: Path, extract_dir: Path) -> Path:
    src_meta = _find_unique_file(input_root, "SOURCE_ARCHIVE_SHA256.txt")
    expected_src_sha = src_meta.read_text(encoding="utf-8").strip().split()[0]
    payload = src_meta.parent / "source_payload.bin"
    if _sha256_file(payload) != expected_src_sha:
        raise RuntimeError("Frozen source_payload.bin SHA256 mismatch")
    extract_dir.mkdir(parents=True, exist_ok=True)
    base_res = extract_dir.resolve()
    with tarfile.open(payload, "r:gz") as tar:
        for member in tar.getmembers():
            if not (member.name == "src" or member.name.startswith("src/")):
                continue
            dest = (extract_dir / member.name).resolve()
            if not dest.is_relative_to(base_res):
                raise RuntimeError(f"Path traversal in archive: {member.name}")
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src_f, dest.open("wb") as out_f:
                    out_f.write(src_f.read())
    src_dir = extract_dir / "src"
    if not (src_dir / "molgap" / "family_training_kernel.py").is_file():
        raise RuntimeError("Unpacked source is missing molgap/family_training_kernel.py")
    return src_dir


def main() -> None:
    t_start = time.perf_counter()
    observation = {
        "format": "molgap-kaggle-allocation-entry-v1",
        "status": "running",
        "error": None,
        "wall_start_utc": datetime.now(timezone.utc).isoformat(),
        "observed_gpu_count": None,
        "gpu_name": None,
        "versions": {},
    }
    try:
        try:
            names = [line.strip() for line in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10
            ).splitlines() if line.strip()]
            if names:
                observation["observed_gpu_count"] = len(names)
                observation["gpu_name"] = names[0]
        except Exception:
            pass

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "numpy<2", "torch-geometric==2.6.1", "ogb==1.3.6", "rdkit==2025.3.5", "scipy", "networkx"],
            check=True,
            timeout=600,
        )
        # Verify RDKit & SciPy imports cleanly after numpy constraint
        subprocess.run(
            [sys.executable, "-c",
             "import numpy, scipy, rdkit, networkx, torch_geometric, ogb; "
             "from scipy.stats import ks_2samp; from rdkit.Chem import rdFingerprintGenerator, rdMMPA"],
            check=True,
            timeout=60,
        )

        for pkg in ("torch", "numpy", "scipy", "rdkit", "networkx", "torch-geometric", "ogb"):
            try:
                observation["versions"][pkg] = importlib.metadata.version(pkg)
            except Exception:
                pass

        bindings_path = _find_unique_file(INPUT_ROOT, "launch_bindings.json", EXPECTED_BINDINGS_SHA256)
        bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
        probe_path = _find_unique_file(INPUT_ROOT, "probe_delta_gap.py", bindings["probe_sha256"])
        _find_unique_file(INPUT_ROOT, "protocol.md", bindings["protocol_sha256"])
        _find_unique_file(INPUT_ROOT, "trajectory.json", bindings["prospective_sha256"])

        src_dir = _unpack_molgap_source(INPUT_ROOT, WORK_ROOT / "molgap-src")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cpu_n = min(4, os.cpu_count() or 4)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        subprocess.run(
            [sys.executable, str(probe_path),
             "--mode", "kaggle",
             "--input-root", str(INPUT_ROOT),
             "--out-dir", str(OUTPUT_ROOT),
             "--workers", str(cpu_n)],
            check=True,
            env=env,
        )
        observation["bindings"] = bindings
        observation["status"] = "complete"
    except BaseException as exc:
        observation["status"] = "failed"
        observation["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        elapsed = time.perf_counter() - t_start
        observation["monotonic_seconds"] = elapsed
        observation["wall_end_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(OUTPUT_ROOT / "allocation_entry_observation.json", observation)


if __name__ == "__main__":
    main()
