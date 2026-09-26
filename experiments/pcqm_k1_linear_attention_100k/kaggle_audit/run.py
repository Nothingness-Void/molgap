"""Frozen-checkpoint audit recovery only; no optimizer, training or model selection."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

# Set before CUDA initialization, including every fresh subprocess.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OUT = Path("/kaggle/working/pcqm_k1_linear_attention_audit")
COMMIT = "63ef7266facf38ca2fd4be3102031ad0ceb2f996"
SOURCE_SHA = "974c198ac45d759d2988ece962dc288130e64390dc7022a7fe689c5f15e7271f"
MODE = "neural_atom_k1_linear_attention"
FIXED_100K = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
FIXED_500K = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(name):
    found = list(Path("/kaggle/input").rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {name}, got {len(found)}")
    return found[0]


def fixed_root(digest):
    found = [p.parent for p in Path("/kaggle/input").rglob("manifest.json") if sha(p) == digest]
    if len(found) != 1:
        raise RuntimeError(f"Accepted immutable cache unavailable: {digest}")
    return found[0]


def unpack(blob, destination):
    with tarfile.open(blob, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def audit():
    source = one("source_payload.bin")
    spec = json.loads(one("AUDIT_RECOVERY.json").read_text())
    if (sha(source) != SOURCE_SHA or one("SOURCE_COMMIT.txt").read_text().strip() != COMMIT
        or spec["source_commit"] != COMMIT or spec["source_archive_sha256"] != SOURCE_SHA
        or spec["mode"] != MODE or spec["training_authorized"] is not False
        or spec["cublas_workspace_config"] != os.environ["CUBLAS_WORKSPACE_CONFIG"]):
        raise RuntimeError("Frozen scientific source or NO_TRAIN authority changed")
    unpack(source, OUT / "source")
    for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
        if sha(OUT / "source" / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Scientific source file changed: {item['path']}")
    candidate_blob = one("candidate_payload.bin")
    if sha(candidate_blob) != spec["candidate_archive_sha256"]:
        raise RuntimeError("Frozen candidate archive changed")
    unpack(candidate_blob, OUT / "frozen_candidate")
    for name, digest in spec["candidate_files"].items():
        if sha(OUT / "frozen_candidate" / MODE / name) != digest:
            raise RuntimeError(f"Frozen candidate artifact changed: {name}")
    sys.path.insert(0, str(OUT / "source/src"))
    import torch
    from molgap.k1_portability_audit import run
    from molgap.training_reproducibility import atomic_json
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Audit needs exactly one visible CUDA device")
    provenance = {
        "format": "molgap-audit-recovery-execution-v1", "complete": False,
        "spec": spec, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "training_executed": False, "optimizer_steps": 0,
    }
    atomic_json(OUT / "recovery_execution.json", provenance)
    started = time.monotonic()
    print(json.dumps(provenance), flush=True)
    try:
        run(cache_100k=fixed_root(FIXED_100K), cache_500k=fixed_root(FIXED_500K),
            reference_model=one("K1_REFERENCE_BEST_MODEL.pt"),
            reference_payload=one("K1_REFERENCE_DEVELOPMENT.pt"),
            transform_asset=one("K1_REFERENCE_TARGET_TRANSFORM.json"),
            candidate_root=OUT / "frozen_candidate", output=OUT / "post100k_audit",
            source_commit=COMMIT, source_archive_sha256=SOURCE_SHA, modes=(MODE,))
        for name, digest in spec["candidate_files"].items():
            if sha(OUT / "frozen_candidate" / MODE / name) != digest:
                raise RuntimeError("Audit mutated frozen inputs")
        provenance["complete"] = True
    except Exception as error:
        provenance["failure"] = repr(error)
        raise
    finally:
        provenance["elapsed_seconds"] = time.monotonic() - started
        atomic_json(OUT / "recovery_execution.json", provenance)


def main():
    if sys.argv[1:] == ["--audit-only"]:
        audit()
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "torch==2.4.1+cu121", "--index-url", "https://download.pytorch.org/whl/cu121"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "torch-geometric==2.6.1", "ogb==1.3.6"])
    subprocess.run([sys.executable, __file__, "--audit-only"], check=True, timeout=5400)


if __name__ == "__main__":
    main()
