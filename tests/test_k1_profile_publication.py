import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flat_publication_exact_bytes_and_existing_extractor(tmp_path):
    stage = load("stage_publication")
    payload = tmp_path / "payload"
    sources = ["src/molgap/" + name + ".py" for name in
               ("k1_execution_profile", "k1_frozen_inference", "pcqm_wedge",
                "k1_screen_training", "k1_pretrained_combo")]
    for name in [*sources, "selected.pt", "train_probe.pt", "setup.sh", "bootstrap.py", "run.sh",
                 "kaggle_entry.py", "protocol.md"]:
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# fixture\n")
    atomic_json(payload / "prospective/trajectory.json", {
        "record_mode": "prospective", "decision": {"outcome": "ACTIVE"}})
    manifest = {"format": "molgap-k1-native-t4-profile-payload-v1",
        "source_files": sources, "sample_source_idx": list(range(4096)),
        "checkpoint": {"sha256": sha256_file(payload / "selected.pt")},
        "files": {p.relative_to(payload).as_posix(): sha256_file(p)
                  for p in payload.rglob("*") if p.is_file()}}
    atomic_json(payload / "payload_manifest.json", manifest)
    output = tmp_path / "publication"
    result = stage.stage(payload, sha256_file(payload / "payload_manifest.json"), output)
    assert not result["remote_action"] and not result["sample_decoded"]
    entry = load("kaggle_entry")
    package, extracted = tmp_path / "package", tmp_path / "extracted"
    package.mkdir()
    extracted.mkdir()
    archive = output / "source_dataset/source_payload.bin"
    (package / "source.tar.gz").write_bytes(archive.read_bytes())
    subprocess.run([sys.executable, "-I", "-S", "-c", entry.UNPACK_COMMAND,
        str(output / "source_dataset/unpack.py"), result["pins"]["EXPECTED_UNPACK_SHA256"],
        str(package), str(extracted), result["payload_manifest_sha256"]], check=True)
    for path in payload.rglob("*"):
        if path.is_file():
            assert (extracted / path.relative_to(payload)).read_bytes() == path.read_bytes()
    entry_text = (output / "kernel/run.py").read_text()
    assert all(name + " = None" not in entry_text for name in result["pins"])
    assert all(p.is_file() for p in (output / "source_dataset").iterdir())
    with pytest.raises(FileExistsError):
        stage.stage(payload, result["payload_manifest_sha256"], output)
