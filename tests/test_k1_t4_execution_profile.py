import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from molgap import k1_execution_profile as profile
from molgap.training_reproducibility import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def payload(root):
    sources = ["src/molgap/" + name + ".py" for name in (
        "k1_execution_profile", "k1_frozen_inference", "k1_screen_training", "pcqm_wedge", "k1_pretrained_combo")]
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic fixture\n")
    (root / "selected.pt").write_bytes(b"not a pickle")
    (root / "train_probe.pt").write_bytes(b"not a pickle")
    for name in ("bootstrap.py", "run.sh", "setup.sh", "kaggle_entry.py", "protocol.md"):
        (root / name).write_bytes(b"synthetic packaging fixture\n")
    (root / "prospective").mkdir()
    atomic_json(root / "prospective/trajectory.json", {
        "record_mode": "prospective", "decision": {"outcome": "ACTIVE"}})
    manifest = {"format": "molgap-k1-native-t4-profile-payload-v1",
                "source_files": sources, "sample_source_idx": list(range(4096)),
                "checkpoint": {"sha256": sha256_file(root / "selected.pt")},
                "files": {p.relative_to(root).as_posix(): sha256_file(p)
                          for p in root.rglob("*") if p.is_file()}}
    atomic_json(root / "payload_manifest.json", manifest)
    return manifest, sha256_file(root / "payload_manifest.json")


def test_payload_verification_before_deserialization(tmp_path):
    manifest, digest = payload(tmp_path)
    assert profile.verify_t4_payload(tmp_path, digest) == manifest
    (tmp_path / "train_probe.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Payload bytes differ"):
        profile.verify_t4_payload(tmp_path, digest)


@pytest.mark.parametrize("mutation", ["digest", "wedge", "duplicate", "role", "extra_source", "escape", "missing_setup"])
def test_payload_fails_closed(tmp_path, mutation):
    manifest, digest = payload(tmp_path)
    if mutation == "digest":
        digest = "0" * 64
    elif mutation == "wedge":
        manifest["source_files"].remove("src/molgap/pcqm_wedge.py")
    elif mutation == "duplicate":
        manifest["sample_source_idx"][0] = 1
    elif mutation == "role":
        atomic_json(tmp_path / "prospective/trajectory.json", {
            "record_mode": "retrospective_partial", "decision": {"outcome": "ACTIVE"}})
        manifest["files"]["prospective/trajectory.json"] = sha256_file(tmp_path / "prospective/trajectory.json")
    elif mutation == "extra_source":
        (tmp_path / "src/molgap/unpinned.py").write_bytes(b"pass")
    elif mutation == "missing_setup":
        del manifest["files"]["setup.sh"]
    else:
        manifest["files"]["../escape"] = "0" * 64
    if mutation != "digest":
        atomic_json(tmp_path / "payload_manifest.json", manifest)
        digest = sha256_file(tmp_path / "payload_manifest.json")
    with pytest.raises(ValueError):
        profile.verify_t4_payload(tmp_path, digest)


def test_mean2_reuses_old_supervised_component_not_default_combined(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    def old_objective(first, second, target, *, mode):
        calls.append(mode)
        supervised = 0.5 * ((first-target).abs().mean() + (second-target).abs().mean())
        return supervised + 0.1 * (first-second).square().mean(), {"supervised_l1": supervised}
    monkeypatch.setitem(sys.modules, "molgap.k1_pretrained_combo", types.SimpleNamespace(objective=old_objective))
    first = torch.tensor([2.0], requires_grad=True)
    second = torch.tensor([-4.0], requires_grad=True)
    loss = profile.mean2_loss(first, second, torch.zeros(1))
    assert loss.item() == 3
    loss.backward()
    assert first.grad.item() == 0.5
    assert second.grad.item() == -0.5
    assert calls == ["pretrained_consistency"]


def test_native_cost_is_allocated_count_not_busy():
    cost = profile.t4_cost(900, 2, 600, phase_seconds=60, eval_write_seconds=30)
    assert cost["allocated_T4_device_hours"] == 0.5
    assert cost["active_case_T4_device_hours"] == pytest.approx(1/6)
    assert cost["active_work_T4_device_hours"] == pytest.approx(690/3600)
    assert cost["active_work_seconds"] == {"cases": 600, "synchronized_phases": 60,
                                          "train_eval_localFS_write": 30}
    assert cost["gpu_busy_seconds"] is None
    assert cost["setup_device_hours"] is None
    assert profile.T4_CASES == (("single_w2", 1, 2), ("double_mean2_w2", 2, 2))
    assert profile.CASES == (("single_w2", 1, 2), ("double_w0", 2, 0), ("double_w2", 2, 2), ("double_w4", 2, 4))
    assert (profile.WARMUP, profile.MEASURE, profile.BATCH_SIZE) == (5, 24, 128)


def test_shared_synchronized_phase_probe_steps_and_boundaries(monkeypatch):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    counts = {"guard": 0, "loss": 0, "optimizer": 0, "synchronize": 0}
    original_step = optimizer.step
    def step():
        counts["optimizer"] += 1
        return original_step()
    def guard():
        counts["guard"] += 1
    def synchronize():
        counts["synchronize"] += 1
    def loss_for(actual_model, batch, passes, meta):
        counts["loss"] += 1
        assert actual_model is model and passes == 2 and meta == {"scratch": True}
        return actual_model(batch.x).square().mean()
    class Batch:
        x = torch.ones(2, 1)
        def cuda(self, *, non_blocking):
            assert non_blocking
            return self
    monkeypatch.setattr(optimizer, "step", step)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    before = model.weight.detach().clone()
    batches = [Batch() for _ in range(8)]
    phases, last_batch, last_loss = profile._synchronized_phases(model, optimizer, iter(batches),
        {"scratch": True}, loss_for=loss_for, guard=guard)
    assert counts == {"guard": 8, "loss": 8, "optimizer": 8, "synchronize": 40}
    assert len(phases) == 6
    assert all(set(row) == {"loader", "h2d", "forward_loss", "backward", "clip", "optimizer"}
               and all(value >= 0 for value in row.values()) for row in phases)
    assert last_batch is batches[-1] and bool(torch.isfinite(last_loss))
    assert not torch.equal(before, model.weight.detach())


def test_supervisor_has_hard_900_second_timeout(monkeypatch, tmp_path):
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, "argv", ["profile", "--native-t4", "--root", str(tmp_path),
                                     "--output", str(tmp_path / "out"), "--expected-manifest-sha256", "a"*64])
    profile.main()
    assert calls[0][1] == {"timeout": 900, "check": True}
    assert "--bounded-child" in calls[0][0]
    shell = (ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/run.sh").read_text()
    assert "remaining=${5:-1200}" in shell
    assert 'timeout --signal=KILL "${remaining}s"' in shell


def test_supervisor_timeout_keeps_incomplete_receipt(monkeypatch, tmp_path):
    import subprocess
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    monkeypatch.setattr(sys, "argv", ["profile", "--native-t4", "--root", str(tmp_path),
                                     "--output", str(tmp_path / "out")])
    with pytest.raises(subprocess.TimeoutExpired):
        profile.main()
    receipt = json.loads((tmp_path / "out/worker_process_observation.json").read_text())
    assert receipt["status"] == "incomplete"
    assert receipt["allocated_T4_device_hours"] is None
    assert json.loads((tmp_path / "out/failure.json").read_text())["type"] == "TimeoutExpired"
    assert not (tmp_path / "out/completion.json").exists()


def test_prepare_rejects_nonprospective_without_loading_graphs(tmp_path):
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/prepare.py"
    spec = importlib.util.spec_from_file_location("t4_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trajectory = tmp_path / "trajectory.json"
    atomic_json(trajectory, {"record_mode": "retrospective_partial", "decision": {"outcome": "ACTIVE"}})
    plan = tmp_path / "plan.json"
    atomic_json(plan, {"trajectory": {"path": str(trajectory), "sha256": sha256_file(trajectory)}})
    with pytest.raises(ValueError, match="Parent must publish"):
        module.prepare(plan, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_loaded_sources_reject_current_frozen_mix(tmp_path):
    with pytest.raises(ValueError, match="Imported nonpayload source"):
        profile.verify_loaded_sources(tmp_path, {"source_files": [], "files": {}})


def test_default_plan_and_prepare_reuse_bytes_without_pickle_decode(monkeypatch, tmp_path):
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/prepare.py"
    spec = importlib.util.spec_from_file_location("t4_byte_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    accepted_root = tmp_path / "accepted"
    accepted_root.mkdir()
    manifest, _ = payload(accepted_root)
    manifest["checkpoint"]["source_sha256"] = "0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a"
    manifest.update(parent_manifest_sha256="a"*64, parent_training_shards=[{"role": "train"}])
    atomic_json(accepted_root / "payload_manifest.json", manifest)
    trajectory = tmp_path / "trajectory.json"
    atomic_json(trajectory, {"trajectory_id": "synthetic", "record_mode": "prospective", "decision": {"outcome": "ACTIVE"}})
    receipt = tmp_path / "receipt.json"
    atomic_json(receipt, {"trajectory_id": "synthetic", "status": "PLANNED"})
    torch = pytest.importorskip("torch")
    def forbidden_load(*args, **kwargs):
        pytest.fail("Exact-byte preparation must not deserialize any graph/checkpoint")
    monkeypatch.setattr(torch, "load", forbidden_load)
    plan = module.default_plan(accepted_root, trajectory, receipt)
    plan_path = tmp_path / "plan.json"
    atomic_json(plan_path, plan)
    output = tmp_path / "payload"
    result = module.prepare(plan_path, output)
    assert (output / "train_probe.pt").read_bytes() == b"not a pickle"
    assert (output / "selected.pt").read_bytes() == b"not a pickle"
    assert (output / "bootstrap.py").is_file()
    assert profile.verify_t4_payload(output, result["payload_manifest_sha256"])["sample_preparation"] == "exact-byte reuse; no pickle decode"
    assert json.loads((output / "payload_manifest.json").read_text())["source_files"] == list(plan["source_files"])
    with pytest.raises(FileExistsError):
        module.prepare(plan_path, output)


def test_synthetic_cpu_case_loop_reset_eval_and_local_write(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    import torch_geometric.loader
    import molgap.k1_frozen_inference as frozen
    import molgap.k1_screen_training as family
    import molgap.training_reproducibility as repro
    monkeypatch.setitem(sys.modules, "molgap.k1_pretrained_combo", types.SimpleNamespace(objective=lambda *args: None))
    manifest, digest = payload(tmp_path)
    manifest["checkpoint"]["source_sha256"] = "0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a"
    atomic_json(tmp_path / "payload_manifest.json", manifest)
    monkeypatch.setattr(profile, "verify_t4_payload", lambda *args: manifest)
    monkeypatch.setattr(profile, "verify_loaded_sources", lambda *args: None)
    monkeypatch.setattr(repro, "configure_fp32_determinism", lambda seed: torch.manual_seed(seed) and {"seed": seed})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    for method in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, method, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda model: model)

    class Graph(dict):
        def __init__(self, index):
            self.source_idx = index
    graphs = [Graph(i) for i in range(4096)]
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: graphs)
    class Batch:
        y = torch.zeros(128)
        def cuda(self, **kwargs):
            return self
    loaders = []
    def loader(items, **kwargs):
        loaders.append(kwargs)
        return [Batch() for _ in range(32)]
    monkeypatch.setattr(torch_geometric.loader, "DataLoader", loader)
    initial_states = []
    def load_selected(*args, **kwargs):
        assert kwargs["expected_epoch"] == 48
        model = torch.nn.Linear(1, 1)
        initial_states.append(model.weight.detach().clone())
        return model, {"mean": 0.0, "std": 1.0}
    monkeypatch.setattr(frozen, "load_native500k_k1", load_selected)
    monkeypatch.setattr(family, "_forward", lambda model, batch: model(torch.ones(128, 1)).view(-1))
    monkeypatch.setattr(profile, "mean2_loss", lambda a, b, y: 0.5*((a-y).abs().mean()+(b-y).abs().mean()))
    original_step = torch.optim.AdamW.step
    optimizer_steps = []
    def counted_step(optimizer, *args, **kwargs):
        optimizer_steps.append(id(optimizer))
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    accepted_bytes = (tmp_path / "selected.pt").read_bytes()
    profile.run(tmp_path, tmp_path / "out", native_t4=True, expected_manifest_sha256=digest)
    result = json.loads((tmp_path / "out/result.json").read_text())
    assert [len(case["steps"]) for case in result["cases"]] == [24, 24]
    assert result["scratch_optimizer_steps_per_case"] == 29
    assert result["scientific_outcome"] == "NO_TRAIN"
    assert all(torch.equal(initial_states[0], state) for state in initial_states)
    assert len(initial_states) == 4
    assert result["phase_optimizer_steps"] == 8
    assert result["scratch_optimizer_steps_total"] == 66
    assert len(optimizer_steps) == 66
    assert len(result["phase_samples"]) == 6
    assert all(set(row) == {"loader", "h2d", "forward_loss", "backward", "clip", "optimizer"}
               for row in result["phase_samples"])
    assert result["gradient_norm_probe"] is None
    assert not result["gradient_relation_probe_performed"]
    assert not (tmp_path / "out/gradient_relation.json").exists()
    cost = result["cost"]
    assert all(value > 0 for value in cost["active_work_seconds"].values())
    assert cost["active_work_T4_device_hours"] == pytest.approx(sum(cost["active_work_seconds"].values())/3600)
    phase_artifact = json.loads((tmp_path / "out/phase_timings.json").read_text())
    assert phase_artifact["warmup_steps"] == 2
    assert phase_artifact["optimizer_steps"] == 8
    assert phase_artifact["samples"] == result["phase_samples"]
    assert all(args["num_workers"] == 2 and args["batch_size"] == 128 for args in loaders)
    timings = result["sample_timings"]
    assert timings["eval_optimizer_steps"] == 0
    assert timings["eval_rows"] == 512
    assert timings["full50k_development_seconds"] is None
    assert timings["remote_upload_seconds"] is None
    assert timings["localFS_checkpoint_bytes"] > 0
    assert (tmp_path / "selected.pt").read_bytes() == accepted_bytes
    assert (tmp_path / "out/completion.json").is_file()


def test_packaged_bootstrap_isolated_and_budgeted(monkeypatch, tmp_path, capsys):
    import subprocess
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/bootstrap.py"
    spec = importlib.util.spec_from_file_location("t4_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "payload"
    root.mkdir()
    for name in ("bootstrap.py", "run.sh"):
        (root / name).write_bytes(b"synthetic bootstrap")
    module.__file__ = str(root / "bootstrap.py")
    atomic_json(root / "payload_manifest.json", {"files": {name: sha256_file(root / name) for name in ("bootstrap.py", "run.sh")}})
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--root", str(root), "--output", str(tmp_path / "output"),
                                     "--expected-manifest-sha256", sha256_file(root / "payload_manifest.json")])
    module.main()
    command, kwargs = calls[0]
    assert 0 < int(command[-1]) < 1200
    assert kwargs["env"]["PYTHONPATH"] == str(root / "src")
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


@pytest.mark.parametrize("failure", [None, "setup", "bootstrap", "multiple", "execution"])
def test_kaggle_entry_verifies_before_execution_and_records_allocation(monkeypatch, tmp_path, capsys, failure):
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/kaggle_entry.py"
    spec = importlib.util.spec_from_file_location("t4_kaggle_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "mount" / "payload"
    root.mkdir(parents=True)
    _, digest = payload(root)
    module.EXPECTED_PROFILE_MANIFEST_SHA256 = digest
    module.EXPECTED_SETUP_SHA256 = sha256_file(root / "setup.sh")
    if failure in ("setup", "bootstrap"):
        (root / ("setup.sh" if failure == "setup" else "bootstrap.py")).write_bytes(b"changed")
    if failure == "multiple":
        other = tmp_path / "mount" / "other"
        other.mkdir()
        payload(other)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if failure == "execution" and command[0] == "timeout":
            raise subprocess.CalledProcessError(137, command)
        return types.SimpleNamespace(stdout="Tesla T4\nTesla T4\n")
    monkeypatch.setattr(module.subprocess, "run", run)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["entry", "--input-root", str(tmp_path / "mount"), "--output", str(output)])
    if failure == "execution":
        import subprocess
        with pytest.raises(subprocess.CalledProcessError):
            module.main()
        assert len(calls) == 2
    elif failure:
        with pytest.raises(ValueError):
            module.main()
        assert calls == []
    else:
        module.main()
        assert len(calls) == 2
        command, kwargs = calls[1]
        assert command[:2] == ["timeout", "--signal=KILL"]
        assert 0 < int(command[-1]) < 1200
        assert command[command.index("--setup-sha256")+1] == module.EXPECTED_SETUP_SHA256
        assert kwargs["env"]["PYTHONPATH"] == str(root / "src")
    report = json.loads((output / "entry_observation.json").read_text())
    assert report["status"] == ("incomplete" if failure else "complete")
    assert report["gpu_busy_seconds"] is None
    assert report["observed_allocated_gpu_count"] == (None if failure in ("setup", "bootstrap", "multiple") else 2)
    if failure in (None, "execution"):
        assert report["allocated_T4_device_hours"] == pytest.approx(report["entry_wall_seconds_including_verify_setup_bootstrap"]*2/3600)
    assert json.loads(capsys.readouterr().out) == report


@pytest.mark.parametrize("failure", [None, "archive", "extractor", "inner_manifest", "inner_file"])
def test_flat_binary_mount_uses_pinned_shared_unpack(monkeypatch, tmp_path, capsys, failure):
    import io
    import subprocess
    import tarfile
    path = ROOT / "experiments/pcqm_k1_t4_cost_quality/profile/kaggle_entry.py"
    spec = importlib.util.spec_from_file_location("flat_t4_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_run = subprocess.run
    original_root = tmp_path / "original"
    original_root.mkdir()
    _, manifest_digest = payload(original_root)
    flat = tmp_path / "mount" / "flat"
    flat.mkdir(parents=True)
    (flat / "payload_manifest.json").write_bytes((original_root / "payload_manifest.json").read_bytes())
    (flat / "unpack.py").write_bytes((ROOT / "src/molgap/experiment_preflight.py").read_bytes())
    if failure == "inner_manifest":
        (original_root / "payload_manifest.json").write_bytes(b"{}")
    if failure == "inner_file":
        (original_root / "train_probe.pt").write_bytes(b"changed")
    archive = flat / "source_payload.bin"
    with tarfile.open(archive, "w:gz") as handle:
        for source in original_root.rglob("*"):
            if source.is_file():
                data = source.read_bytes()
                info = tarfile.TarInfo(source.relative_to(original_root).as_posix())
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
    module.EXPECTED_PROFILE_MANIFEST_SHA256 = manifest_digest
    module.EXPECTED_SETUP_SHA256 = sha256_file(original_root / "setup.sh")
    module.EXPECTED_PROFILE_ARCHIVE_SHA256 = sha256_file(archive)
    module.EXPECTED_UNPACK_SHA256 = sha256_file(flat / "unpack.py")
    if failure == "archive":
        archive.write_bytes(b"tampered")
    if failure == "extractor":
        (flat / "unpack.py").write_bytes(b"raise AssertionError('must not execute')")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "-I" in command:
            return original_run(command, **kwargs, capture_output=True)
        return types.SimpleNamespace(stdout="Tesla T4\nTesla T4\n")
    monkeypatch.setattr(module.subprocess, "run", run)
    output, temp = tmp_path / "output", tmp_path / "temp"
    monkeypatch.setattr(sys, "argv", ["entry", "--input-root", str(flat.parent),
        "--output", str(output), "--temp-root", str(temp)])
    if failure in ("archive", "extractor"):
        with pytest.raises(ValueError):
            module.main()
        assert not calls
        assert not (temp / "profile-payload").exists()
    elif failure:
        with pytest.raises(subprocess.CalledProcessError):
            module.main()
        assert len(calls) == 1
    else:
        module.main()
        assert len(calls) == 3
        assert (temp / "profile-package/source.tar.gz").read_bytes() == archive.read_bytes()
        assert (temp / "profile-payload/train_probe.pt").read_bytes() == b"not a pickle"
        assert calls[-1][1]["env"]["PYTHONPATH"] == str(temp / "profile-payload/src")
        assert (temp / "profile-payload/payload_manifest.json").read_bytes() == (flat / "payload_manifest.json").read_bytes()
    report = json.loads((output / "entry_observation.json").read_text())
    assert report["status"] == ("incomplete" if failure else "complete")
    assert report["unpack_sha256"] is not None and report["source_archive_sha256"] is not None
    assert json.loads(capsys.readouterr().out) == report
