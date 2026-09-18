from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "molgap_v5_colab_bootstrap.ipynb"
SOURCE_COMMIT = "04cc7bfbcae5b1ae76ffaa6307a91f948e97904d"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> None:
    cells = [
        markdown(
            """# MolGap V5 Colab bootstrap

This notebook installs platform infrastructure only. It does not select a
model, authorize training, read protected OGB roles, or connect an accelerator
by itself. Run the CPU setup first; enable data staging and accelerator
preflight only when their explicit gates are satisfied.
"""
        ),
        code(
            f'''# 0. Frozen identities and explicit gates.
from pathlib import Path
import hashlib, json, os, platform, shutil, subprocess, sys, tempfile, time

SOURCE_REPOSITORY = "https://github.com/Nothingness-Void/molgap.git"
SOURCE_COMMIT = "{SOURCE_COMMIT}"
COMMON_CONTRACT = "MOLGAP-COMMON-V5-FINAL"
DESKTOP_CONTRACT = "MOLGAP-DESKTOP-V5-FINAL"
PHYSICAL_BATCH_PER_DEVICE = 128
PRECISION = "fp32"
TF32_ALLOWED = False

FIXED_DATASETS = {{
    "100k": {{
        "ref": "nvoid912/pcqm4mv2-ogb-fixed-100k-v1",
        "manifest_sha256": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
        "train_rows": 100000,
        "development_rows": 50000,
    }},
    "500k": {{
        "ref": "nvoid912/pcqm4mv2-ogb-fixed-500k-scnet-v1",
        "manifest_sha256": "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751",
        "train_rows": 500000,
        "development_rows": 50000,
    }},
}}

STAGE_FIXED_DATA = False
FIXED_DATA_ROLE = "100k"  # "100k" or "500k"
RUN_ACCELERATOR_PREFLIGHT = False

assert FIXED_DATA_ROLE in FIXED_DATASETS
assert not STAGE_FIXED_DATA, "Review the staging cell before enabling data transfer."
assert not RUN_ACCELERATOR_PREFLIGHT, "Select the intended runtime before enabling GPU use."
print("Frozen source:", SOURCE_COMMIT)
print("Training remains unauthorized.")'''
        ),
        code(
            '''# 1. Mount Drive and create the durable V5 layout.
from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/MolGap/V5")
CONTRACT_ROOT = DRIVE_ROOT / "contracts"
FIXED_ROOT = DRIVE_ROOT / "fixed_data"
RUN_ROOT = DRIVE_ROOT / "runs"
CHECKPOINT_ROOT = DRIVE_ROOT / "checkpoints"
RECORD_ROOT = DRIVE_ROOT / "records"
for directory in (CONTRACT_ROOT, FIXED_ROOT, RUN_ROOT, CHECKPOINT_ROOT, RECORD_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    os.replace(temporary, path)

print("Durable V5 root:", DRIVE_ROOT)'''
        ),
        code(
            '''# 2. Install the exact source snapshot; never train from a moving branch.
SOURCE_ROOT = Path("/content/molgap-v5-source")
if SOURCE_ROOT.exists():
    shutil.rmtree(SOURCE_ROOT)
subprocess.check_call([
    "git", "clone", "--filter=blob:none", "--no-checkout",
    SOURCE_REPOSITORY, str(SOURCE_ROOT),
])
subprocess.check_call(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=SOURCE_ROOT)
observed_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True
).strip()
assert observed_commit == SOURCE_COMMIT

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(SOURCE_ROOT)])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch-geometric==2.6.1", "ogb==1.3.6", "kaggle==1.7.4.5",
])

# Editable-install .pth files are processed only when Python starts. Colab
# keeps the current kernel alive, so expose the frozen src tree immediately.
import importlib
source_python = str(SOURCE_ROOT / "src")
if source_python not in sys.path:
    sys.path.insert(0, source_python)
importlib.invalidate_caches()

from molgap.v5_desktop import V5_CONTRACT_ID, V5_DESKTOP_CONTRACT_ID
assert V5_CONTRACT_ID == COMMON_CONTRACT
assert V5_DESKTOP_CONTRACT_ID == DESKTOP_CONTRACT
print("Accepted source commit:", observed_commit)'''
        ),
        code(
            '''# 3. Persist the source/contract infrastructure identity.
source_files = sorted(
    path.relative_to(SOURCE_ROOT).as_posix()
    for path in (SOURCE_ROOT / "src" / "molgap").rglob("*.py")
)
source_hash = hashlib.sha256()
for relative in source_files:
    path = SOURCE_ROOT / relative
    source_hash.update(relative.encode("utf-8"))
    source_hash.update(b"\\0")
    source_hash.update(path.read_bytes())

infrastructure = {
    "format": "molgap-colab-v5-infrastructure-v1",
    "status": "source-and-drive-ready",
    "common_contract": COMMON_CONTRACT,
    "desktop_contract": DESKTOP_CONTRACT,
    "source_repository": SOURCE_REPOSITORY,
    "source_commit": SOURCE_COMMIT,
    "source_tree_sha256": source_hash.hexdigest(),
    "source_file_count": len(source_files),
    "physical_batch_per_device": PHYSICAL_BATCH_PER_DEVICE,
    "precision": PRECISION,
    "tf32_allowed": TF32_ALLOWED,
    "fixed_datasets": FIXED_DATASETS,
    "official_validation_role_read": False,
    "test_dev_role_read": False,
    "test_challenge_role_read": False,
    "training_authorized": False,
}
atomic_json(CONTRACT_ROOT / "infrastructure_manifest.json", infrastructure)
print(json.dumps(infrastructure, indent=2))'''
        ),
        code(
            '''# 4. Optional fixed-data staging from the accepted private Kaggle3 mirror.
# Enable only after adding KAGGLE_USERNAME and KAGGLE_KEY to Colab Secrets.
def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def accept_fixed_dataset(root: Path, expected: dict) -> dict:
    manifest_path = root / "manifest.json"
    assert manifest_path.is_file(), manifest_path
    observed_manifest = sha256_file(manifest_path)
    assert observed_manifest == expected["manifest_sha256"], (
        observed_manifest, expected["manifest_sha256"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["source"]["external_data_used"] is False
    assert manifest["roles"]["train"]["rows"] == expected["train_rows"]
    assert manifest["roles"]["development"]["rows"] == expected["development_rows"]
    assert manifest["official_validation_role_read"] is False
    assert manifest["test_dev_role_read"] is False
    assert manifest["test_challenge_role_read"] is False
    for shard in manifest["geometry_shards"]:
        path = root / shard["file"]
        assert path.is_file(), path
        assert path.stat().st_size == shard["bytes"]
        assert sha256_file(path) == shard["sha256"]
    return {
        "format": "molgap-colab-v5-fixed-data-acceptance-v1",
        "status": "accepted",
        "role": FIXED_DATA_ROLE,
        "ref": expected["ref"],
        "manifest_sha256": observed_manifest,
        "graph_shards": len(manifest["geometry_shards"]),
        "protected_roles_included": False,
    }

if STAGE_FIXED_DATA:
    from google.colab import userdata
    username = userdata.get("KAGGLE_USERNAME")
    key = userdata.get("KAGGLE_KEY")
    assert username and key, "Add Kaggle credentials to Colab Secrets first."
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    expected = FIXED_DATASETS[FIXED_DATA_ROLE]
    destination = FIXED_ROOT / FIXED_DATA_ROLE
    with tempfile.TemporaryDirectory(dir="/content") as temporary:
        temporary = Path(temporary)
        subprocess.check_call([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", expected["ref"], "-p", str(temporary), "--unzip",
        ], env=env)
        acceptance = accept_fixed_dataset(temporary, expected)
        if destination.exists():
            existing = accept_fixed_dataset(destination, expected)
            assert existing == acceptance
        else:
            shutil.copytree(temporary, destination)
        atomic_json(RECORD_ROOT / f"fixed_data_{{FIXED_DATA_ROLE}}_acceptance.json", acceptance)
        print(json.dumps(acceptance, indent=2))
else:
    print("Fixed-data staging is disabled; no credentials or network transfer used.")'''
        ),
        code(
            '''# 5. Optional paid-accelerator preflight. This is not a training authorization.
def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

if RUN_ACCELERATOR_PREFLIGHT:
    import torch
    assert torch.cuda.is_available(), "Select a GPU runtime before preflight."
    assert torch.cuda.device_count() == 1, "V5 Colab runs require one visible GPU."
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    def two_step_hash() -> str:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1)
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        generator = torch.Generator(device="cuda").manual_seed(43)
        x = torch.randn(128, 16, generator=generator, device="cuda")
        y = torch.randn(128, 1, generator=generator, device="cuda")
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
            optimizer.step()
        return state_sha256(model.state_dict())

    first_hash = two_step_hash()
    second_hash = two_step_hash()
    assert first_hash == second_hash
    certificate = {
        "format": "molgap-colab-v5-runtime-certificate-v1",
        "status": "accepted",
        "platform_id": "google-colab",
        "accelerator": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH_PER_DEVICE,
        "optimizer_step_repeatability": {
            "steps": 2,
            "repeat_count": 2,
            "state_sha256": [first_hash, second_hash],
            "exact": True,
        },
        "source_commit": SOURCE_COMMIT,
        "training_authorized": False,
    }
    certificate["certificate_id"] = hashlib.sha256(
        json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    atomic_json(RECORD_ROOT / "runtime_certificate.json", certificate)
    print(json.dumps(certificate, indent=2))
else:
    print("Accelerator preflight is disabled; no paid accelerator was requested.")'''
        ),
        code(
            '''# 6. Create an immutable, resumable run directory only from an explicit run spec.
def bind_run(run_id: str, run_spec: dict) -> dict:
    assert run_id and "/" not in run_id and "\\\\" not in run_id
    required = {
        "experiment_id", "model_family", "dataset_role", "source_commit",
        "seed", "epochs", "physical_batch_per_device", "precision",
        "checkpoint_interval", "resume_contract",
    }
    assert required <= set(run_spec), sorted(required - set(run_spec))
    assert run_spec["source_commit"] == SOURCE_COMMIT
    assert run_spec["dataset_role"] in FIXED_DATASETS
    assert run_spec["physical_batch_per_device"] == PHYSICAL_BATCH_PER_DEVICE
    assert run_spec["precision"] == PRECISION
    run_directory = RUN_ROOT / run_id
    checkpoint_directory = CHECKPOINT_ROOT / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    spec_path = run_directory / "run_spec.json"
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        assert existing == run_spec, "Run ID already belongs to a different immutable spec."
    else:
        atomic_json(spec_path, run_spec)
    binding = {
        "format": "molgap-colab-v5-run-binding-v1",
        "owner": "desktop",
        "run_id": run_id,
        "run_spec": str(spec_path),
        "run_directory": str(run_directory),
        "checkpoint_directory": str(checkpoint_directory),
        "source_config_frozen": True,
        "remote_artifacts_durable": True,
        "resume_supported": True,
        "training_authorized": False,
    }
    atomic_json(run_directory / "binding.json", binding)
    return binding

print("Run binder ready. No run has been created or authorized.")'''
        ),
        code(
            '''# 7. Infrastructure acceptance summary.
manifest_path = CONTRACT_ROOT / "infrastructure_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert manifest["status"] == "source-and-drive-ready"
assert manifest["source_commit"] == SOURCE_COMMIT
assert manifest["common_contract"] == COMMON_CONTRACT
assert manifest["desktop_contract"] == DESKTOP_CONTRACT
assert manifest["training_authorized"] is False

summary = {
    "format": "molgap-colab-v5-infrastructure-acceptance-v1",
    "status": "accepted",
    "source_commit": SOURCE_COMMIT,
    "drive_root": str(DRIVE_ROOT),
    "fixed_data_staged": STAGE_FIXED_DATA,
    "accelerator_preflight_run": RUN_ACCELERATOR_PREFLIGHT,
    "training_authorized": False,
}
atomic_json(RECORD_ROOT / "infrastructure_acceptance.json", summary)
print(json.dumps(summary, indent=2))'''
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
