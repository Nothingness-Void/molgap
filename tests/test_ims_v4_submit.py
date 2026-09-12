import importlib.util
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "platforms/ims/v4_submit/launcher.py"
SPEC = importlib.util.spec_from_file_location("ims_v4_launcher", MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(launcher)
SUBMIT_PATH = ROOT / "platforms/ims/v4_submit/submit_from_windows.py"
SUBMIT_SPEC = importlib.util.spec_from_file_location("ims_v4_submit", SUBMIT_PATH)
submit = importlib.util.module_from_spec(SUBMIT_SPEC)
assert SUBMIT_SPEC.loader is not None
SUBMIT_SPEC.loader.exec_module(submit)


def valid_spec():
    return {
        "format": "molgap-ims-v4-run-v1",
        "run_id": "gptrans-t-100k-v4-s42",
        "model_id": "gptrans-t",
        "dataset_id": "ogb-train-100k",
        "runtime_id": "ims-a100-torch271-v1",
        "source": {"commit": "a" * 40, "archive_sha256": "b" * 64},
        "contract": {
            "path": "experiments/model/training_contract.json",
            "sha256": "c" * 64,
            "physical_batch_per_device": 128,
            "precision": "fp32",
            "tf32": False,
            "tail_batch_policy": "drop_last",
        },
        "resources": {
            "queue": "H",
            "ncpus": 8,
            "ompthreads": 4,
            "ngpus": 1,
            "preflight_walltime": "00:45:00",
            "train_walltime": "08:00:00",
        },
        "stages": {
            name: {
                "entrypoint": "experiments/model/run.py",
                "args": [name, "--root", "{dataset_root}"],
                "required_outputs": [f"{name}/complete.json"],
            }
            for name in ("preflight", "train")
        },
        "sealed_roles": {
            "official_validation_read": False,
            "test_dev_read": False,
            "test_challenge_read": False,
        },
    }


DATASETS = {"datasets": {"ogb-train-100k": {}}}
RUNTIMES = {"runtimes": {"ims-a100-torch271-v1": {"queue": "H"}}}


def test_v4_spec_accepts_frozen_contract():
    launcher.validate_spec(valid_spec(), DATASETS, RUNTIMES)


@pytest.mark.parametrize(
    ("field", "value"),
    [("physical_batch_per_device", 96), ("precision", "amp"), ("tf32", True), ("tail_batch_policy", "keep")],
)
def test_v4_spec_rejects_contract_drift(field, value):
    spec = valid_spec()
    spec["contract"][field] = value
    with pytest.raises(RuntimeError):
        launcher.validate_spec(spec, DATASETS, RUNTIMES)


def test_v4_spec_rejects_sealed_role_and_path_escape():
    spec = valid_spec()
    spec["sealed_roles"]["test_dev_read"] = True
    with pytest.raises(RuntimeError):
        launcher.validate_spec(spec, DATASETS, RUNTIMES)
    spec = valid_spec()
    spec["stages"]["train"]["entrypoint"] = "../outside.py"
    with pytest.raises(RuntimeError):
        launcher.validate_spec(spec, DATASETS, RUNTIMES)


def test_job_id_parser_and_walltime():
    assert launcher.parse_job_id("1484401.ccpbs1") == "1484401.ccpbs1"
    assert launcher.walltime_seconds("08:00:00") == 28800


def test_maintenance_guard_rejects_preceding_sunday():
    sunday = datetime(2026, 10, 4, 10, tzinfo=ZoneInfo("Asia/Tokyo"))
    with pytest.raises(RuntimeError):
        launcher.maintenance_guard(3600, now=sunday)


def test_expand_args_rejects_unknown_placeholder():
    with pytest.raises(RuntimeError):
        launcher.expand_args(["{unknown}"], {})


def test_source_inventory_is_checked(tmp_path):
    archive = tmp_path / "source.tar.gz"
    payload = b"frozen source\n"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("src/model.py")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    inventory = tmp_path / "source_files.json"
    inventory.write_text(
        json.dumps(
            {
                "format": "molgap-source-files-v1",
                "files": [
                    {"path": "src/model.py", "sha256": hashlib.sha256(payload).hexdigest()}
                ],
            }
        ),
        encoding="utf-8",
    )
    launcher.validate_source_inventory(archive, inventory)
    inventory.write_text(
        json.dumps(
            {
                "format": "molgap-source-files-v1",
                "files": [{"path": "src/model.py", "sha256": "0" * 64}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        launcher.validate_source_inventory(archive, inventory)


def test_safe_extract_rejects_path_escape(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../outside.py")
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError):
        launcher.safe_extract(archive, tmp_path / "code")


def test_windows_packager_fills_source_and_contract_hashes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "experiments/model").mkdir(parents=True)
    contract = repo / "experiments/model/training_contract.json"
    contract.write_text('{"physical_batch_per_device": 128}\n', encoding="utf-8")
    (repo / "experiments/model/run.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    run_spec = valid_spec()
    template = tmp_path / "spec.json"
    template.write_text(json.dumps(run_spec), encoding="utf-8")
    output = tmp_path / "payload"
    packaged = submit.package(repo, template, output)
    assert packaged["source"]["commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    assert packaged["source"]["archive_sha256"] == submit.sha256_file(
        output / "source.tar.gz"
    )
    assert packaged["contract"]["sha256"] == hashlib.sha256(
        contract.read_bytes()
    ).hexdigest()
    assert (output / "source_files.json").is_file()
