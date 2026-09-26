"""Hash-bound NO_TRAIN launcher; frozen model code is never replaced."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
OUT = Path("/kaggle/working/pcqm_k1_relation_audit")


def one(name):
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name}; found {len(matches)}")
    return matches[0]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(blob, target):
    with tarfile.open(blob, "r:gz") as archive:
        archive.extractall(target, filter="data")


def main():
    started = time.monotonic()
    spec = json.loads(one("AUDIT_RELEASE.json").read_text())
    if spec["training_authorized"] is not False:
        raise ValueError("NO_TRAIN release required")
    if sys.argv[1:] == ["--infer"]:
        source = one("source_payload.bin")
        if sha(source) != spec["source_archive_sha256"] or one("SOURCE_COMMIT.txt").read_text().strip() != spec["source_commit"]:
            raise ValueError("Frozen model implementation changed")
        unpack(source, OUT / "source")
        for item in json.loads(one("SOURCE_FILES.json").read_text())["files"]:
            if sha(OUT / "source" / item["path"]) != item["sha256"]:
                raise ValueError("Source file inventory changed")
        inputs = one("audit_inputs.bin")
        helper = one("audit_impl.py")
        if sha(inputs) != spec["input_archive_sha256"] or sha(helper) != spec["audit_helper_sha256"]:
            raise ValueError("Released audit helper or inputs changed")
        unpack(inputs, OUT / "inputs")
        sys.path.insert(0, str(OUT / "source/src"))
        module_spec = importlib.util.spec_from_file_location("accepted_relation_audit", helper)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        def cache(digest):
            paths = [p.parent for p in Path("/kaggle/input").rglob("manifest.json") if sha(p) == digest]
            if len(paths) != 1:
                raise ValueError("Fixed cache manifest missing or ambiguous")
            return paths[0]
        module.run(OUT / "inputs",
            cache("1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"),
            cache("630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"),
            OUT / "post100k_audit", spec)
        return
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
    OUT.mkdir(parents=True, exist_ok=False)
    result = {"run_id": spec["run_id"], "complete": False, "training_executed": False,
        "optimizer_steps": 0, "allocated_devices": devices, "used_device_count": 1,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "spec": spec}
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.4.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121"], check=True, timeout=600)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "torch-geometric==2.6.1", "ogb==1.3.6"], check=True, timeout=180)
        remaining = spec["max_allocated_device_seconds"] / len(devices) - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Audit allocation budget exhausted during setup")
        subprocess.run([sys.executable, __file__, "--infer"], check=True, timeout=remaining)
        result["complete"] = True
    except Exception as error:
        result["failure"] = repr(error)
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["allocated_device_seconds"] = result["wall_seconds"] * len(devices)
        tmp = OUT / "execution.json.tmp"
        tmp.write_text(json.dumps(result, indent=2))
        os.replace(tmp, OUT / "execution.json")


if __name__ == "__main__":
    main()
