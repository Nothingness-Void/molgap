"""Synthetic metadata acceptance only; never accept/finalize the retained run."""
import importlib.util
import json
from pathlib import Path

import pytest

from molgap.training_reproducibility import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t4_close", ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/close.py")
close = importlib.util.module_from_spec(spec)
spec.loader.exec_module(close)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # Retained JSON shapes seed a synthetic tree; all executable/pickle bytes are inert.
    for directory in (close.REL, close.RESULTS):
        target = tmp_path / directory
        target.mkdir(parents=True)
        for source in (ROOT / directory).glob("*.json"):
            (target / source.name).write_bytes(source.read_bytes())
    local = tmp_path / close.REL
    (local / "rml").mkdir()
    for name in ("trajectory.json",):
        (local / "rml" / name).write_bytes((ROOT / close.REL / "rml" / name).read_bytes())
    for name in ("protocol.md", "terminal_decision.md", "attribution.md"):
        (local / name).write_text("synthetic authority", encoding="utf-8")
    payload = tmp_path / close.STAGING / "payload"
    payload.mkdir(parents=True)
    manifest = json.loads((ROOT / close.STAGING / "payload/payload_manifest.json").read_text())
    for name in manifest["files"]:
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.startswith("prospective/"):
            source = local / ("rml/trajectory.json" if name.endswith("trajectory.json") else "plan_receipt.json")
            path.write_bytes(source.read_bytes())
        else:
            path.write_bytes(b"synthetic inert bytes\n")
        manifest["files"][name] = sha256_file(path)
    manifest["checkpoint"]["sha256"] = manifest["files"]["selected.pt"]
    atomic_json(payload / "payload_manifest.json", manifest)
    digest = sha256_file(payload / "payload_manifest.json")
    monkeypatch.setattr(close, "MANIFEST_SHA", digest)
    plan = json.loads((local / "payload_plan.json").read_text())
    plan["checkpoint"]["sha256"] = manifest["checkpoint"]["sha256"]
    plan["train_probe"]["sha256"] = manifest["files"]["train_probe.pt"]
    plan["source_files"] = {n: manifest["files"][n] for n in manifest["source_files"]}
    for name, pin in plan["packaging_files"].items():
        pin["sha256"] = manifest["files"][name]
    atomic_json(local / "payload_plan.json", plan)
    publication = json.loads((local / "publication_binding.json").read_text())
    publication["payload_manifest_sha256"] = digest
    for name in publication["files"]:
        path = tmp_path / close.STAGING / "publication" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((payload / "payload_manifest.json").read_bytes()
                         if name.endswith("payload_manifest.json") else b"synthetic publication")
        publication["files"][name] = sha256_file(path)
    pins = publication["pins"]
    pins["EXPECTED_PROFILE_MANIFEST_SHA256"] = digest
    pins["EXPECTED_SETUP_SHA256"] = manifest["files"]["setup.sh"]
    pins["EXPECTED_PROFILE_ARCHIVE_SHA256"] = publication["files"]["source_dataset/source_payload.bin"]
    pins["EXPECTED_UNPACK_SHA256"] = publication["files"]["source_dataset/unpack.py"]
    atomic_json(local / "publication_binding.json", publication)
    result_dir = tmp_path / close.RESULTS
    runtime = json.loads((result_dir / "runtime.json").read_text())
    runtime["payload_sha256"] = digest
    atomic_json(result_dir / "runtime.json", runtime)
    entry = json.loads((result_dir / "entry_observation.json").read_text())
    for field, pin in (("payload_manifest_sha256", "EXPECTED_PROFILE_MANIFEST_SHA256"),
                       ("setup_sha256", "EXPECTED_SETUP_SHA256"),
                       ("source_archive_sha256", "EXPECTED_PROFILE_ARCHIVE_SHA256"),
                       ("unpack_sha256", "EXPECTED_UNPACK_SHA256")):
        entry[field] = pins[pin]
    atomic_json(result_dir / "entry_observation.json", entry)
    scratch = result_dir / "scratch_publication_probe.pt"
    scratch.write_bytes(b"inert checkpoint")
    sample = json.loads((result_dir / "sample_timings.json").read_text())
    sample["localFS_checkpoint_bytes"] = scratch.stat().st_size
    atomic_json(result_dir / "sample_timings.json", sample)
    result = json.loads((result_dir / "result.json").read_text())
    result["sample_timings"] = sample
    atomic_json(result_dir / "result.json", result)
    rebind(tmp_path)
    return tmp_path


def rebind(root):
    directory = root / close.RESULTS
    completion = json.loads((directory / "completion.json").read_text())
    completion["artifacts"] = {n: sha256_file(directory / n) for n in completion["artifacts"]}
    atomic_json(directory / "completion.json", completion)


def test_read_only_acceptance_and_cost(fixture, monkeypatch):
    monkeypatch.setattr(close, "finalize", lambda *a: pytest.fail("finalize called"))
    before = {p: p.read_bytes() for p in fixture.rglob("*") if p.is_file()}
    report = close.close(fixture)
    assert report["single_step_saving_fraction"] == pytest.approx(0.37630898043)
    assert report["allocated_T4_device_hours"] == pytest.approx(0.17095981645388886)
    assert report["worker_subset_seconds"] == pytest.approx(56.740963478)
    assert report["accuracy_causality"] == "insufficient_evidence"
    assert before == {p: p.read_bytes() for p in fixture.rglob("*") if p.is_file()}
    bindings = close.accept(fixture)["artifact_hashes"]
    assert close.REL + "/rml/trajectory.json" not in bindings
    assert close.STAGING + "/payload/prospective/trajectory.json" in bindings


@pytest.mark.parametrize("file,field,value", [
    ("runtime.json", "visible_gpu_count", 1),
    ("runtime.json", "scientific_training", True),
    ("result.json", "scratch_optimizer_steps_total", 58),
    ("result.json", "development_rows", [1]),
    ("result.json", "protected_roles", "used"),
    ("result.json", "consistency_coefficient", 0.1),
    ("result.json", "training_replay_ready", True),
    ("result.json", "accuracy_acceptance", True),
    ("result.json", "training_rows", list(range(4096))),
    ("entry_observation.json", "allocated_T4_device_hours", 0.0315),
    ("entry_observation.json", "source_archive_sha256", "0" * 64),
    ("phase_timings.json", "optimizer_steps", 6),
    ("sample_timings.json", "full50k_development_seconds", 10),
])
def test_rejects_scope_or_binding_drift(fixture, file, field, value):
    path = fixture / close.RESULTS / file
    obj = json.loads(path.read_text())
    obj[field] = value
    atomic_json(path, obj)
    rebind(fixture)
    with pytest.raises(ValueError):
        close.accept(fixture)


@pytest.mark.parametrize("target", ["scratch", "source", "checkpoint", "prospective", "publication"])
def test_rejects_byte_drift(fixture, target):
    paths = {"scratch": close.RESULTS + "/scratch_publication_probe.pt",
             "source": close.STAGING + "/payload/src/molgap/pcqm_wedge.py",
             "checkpoint": close.STAGING + "/payload/selected.pt",
             "prospective": close.REL + "/rml/trajectory.json",
             "publication": close.STAGING + "/publication/kernel/run.py"}
    (fixture / paths[target]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        close.accept(fixture)


def test_completion_requires_scratch_and_confines_paths(fixture):
    path = fixture / close.RESULTS / "completion.json"
    obj = json.loads(path.read_text())
    obj["artifacts"]["../escape"] = obj["artifacts"].pop("scratch_publication_probe.pt")
    atomic_json(path, obj)
    with pytest.raises(ValueError, match="inventory"):
        close.accept(fixture)


@pytest.mark.parametrize("field,value", [("status", "RUNNING"), ("failureMessage", "failed")])
def test_requires_actual_scheduler_complete(fixture, field, value):
    path = fixture / close.REL / "scheduler_reconciliation_complete.json"
    obj = json.loads(path.read_text())
    obj["state"][field] = value
    atomic_json(path, obj)
    with pytest.raises(ValueError, match="Scheduler"):
        close.accept(fixture)


def test_explicit_closure_uses_shared_finalizer_with_no_double_count(fixture, monkeypatch):
    calls = []
    monkeypatch.setattr(close, "finalize", lambda *args: calls.append(args) or {"status": "MOCK_ONLY"})
    with pytest.raises(ValueError, match="timestamp"):
        close.close(fixture, execute=True)
    assert close.close(fixture, execute=True, finalized_at="2026-10-09T12:00:00+00:00") == {"status": "MOCK_ONLY"}
    assert len(calls) == 1
    terminal = json.loads((fixture / close.REL / "terminal.json").read_text())
    assert len(terminal["costs"]) == 1
    assert terminal["costs"][0]["measurement"]["device_hours"]["value"] == pytest.approx(0.17095981645)
    assert terminal["costs"][0]["measurement"]["queue_hours"]["value"] is None
    assert len(terminal["roles"]) == 2
    assert all(e["role_name"] == "train_probe" for e in terminal["roles"])
    assert terminal["decision"]["outcome"] == "NO_TRAIN"
    assert terminal["decision"]["next_allowed_actions"] == []
    assert close.REL + "/rml/trajectory.json" not in terminal["artifact_hashes"]
