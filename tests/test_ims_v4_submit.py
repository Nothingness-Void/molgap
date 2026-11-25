import importlib.util
import hashlib
import io
import json
import subprocess
import sys
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


def test_safe_extract_rejects_path_escape(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("../outside.py")
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError):
        launcher.safe_extract(archive, tmp_path / "code")


@pytest.mark.parametrize("lost_stage,failure", [(None, None), ("preflight", "timeout"),
    ("training", "timeout"), ("preflight", "parse"), ("training", "parse")],
    ids=["observed", "preflight-unknown", "training-unknown", "preflight-unparseable", "training-unparseable"])
def test_windows_packager_fills_source_and_contract_hashes(tmp_path, monkeypatch, capsys, lost_stage, failure):
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
    boundary = tmp_path / "boundary"
    infrastructure = boundary / "infrastructure"
    output = infrastructure / "inbox/fixture"
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
    assert submit.reusable(None, output) == packaged
    changed = json.loads(template.read_text(encoding="utf-8"))
    changed["resources"]["train_walltime"] = "00:42:00"
    template.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen payload"):
        submit.reusable(template, output)
    # Extend the same committed payload through the real registry, archive,
    # extraction, contract and PBS owners. Only scheduler/clock/platform paths
    # are substituted; no worker, data or accelerator is invoked.
    fixed = boundary / "fixed"
    acceptance, manifest = fixed / "acceptance.json", fixed / "manifest.json"
    launcher.atomic_json(acceptance, {"status": "accepted", "verify_content": True})
    launcher.atomic_json(manifest, {"status": "complete", "identity": {"name": "ogb-train-100k"},
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False})
    launcher.atomic_json(infrastructure / "dataset_registry.json", {
        "boundary": str(boundary), "fixed_root": str(fixed),
        "acceptance": {"path": "acceptance.json", "sha256": launcher.sha256_file(acceptance)},
        "datasets": {"ogb-train-100k": {"manifest": "manifest.json", "manifest_sha256": launcher.sha256_file(manifest)}}})
    launcher.atomic_json(infrastructure / "runtime_registry.json", {"runtimes": {
        "ims-a100-torch271-v1": {"queue": "H", "python": "synthetic-python", "pythonpath": []}}})
    monkeypatch.setattr(launcher, "BOUNDARY", boundary)
    guard = launcher.maintenance_guard
    monkeypatch.setattr(launcher, "maintenance_guard", lambda seconds: guard(seconds,
        now=datetime(2026, 10, 22, 12, tzinfo=ZoneInfo("Asia/Tokyo"))))
    run_root = infrastructure / "runs" / run_spec["run_id"]
    journal = run_root / "submission.json"
    calls = []
    def scheduler(command, **kwargs):
        stage = "preflight" if command[-1].endswith("preflight.pbs") else "training"
        state = launcher.load_json(journal)
        if state["status"] == "resume_submitting":
            calls.append("resume")
            launcher.atomic_json(run_root / "output/train/stage_status.json", {"status": "complete"})
            return subprocess.CompletedProcess(command, 0, stdout="102.ccpbs1", stderr="")
        assert state["status"] == stage + "_submitting"
        assert command[0] == "jsub" and kwargs == dict(check=True, capture_output=True, text=True)
        calls.append(stage)
        if stage == "training":
            assert state["preflight_job"] == "100.ccpbs1"
            assert "depend=afterok:100.ccpbs1" in command
        if stage == lost_stage and failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        result = "100.ccpbs1" if stage == "preflight" else "101.ccpbs1"
        if stage == lost_stage and failure == "parse":
            result = "unparseable synthetic response"
        return subprocess.CompletedProcess(command, 0, stdout=result, stderr="")
    monkeypatch.setattr(launcher.subprocess, "run", scheduler)
    if lost_stage:
        expected_error = subprocess.TimeoutExpired if failure == "timeout" else RuntimeError
        with pytest.raises(expected_error):
            launcher.submit_run(output, infrastructure)
        state = launcher.load_json(journal)
        assert state["status"] == lost_stage + "_unknown"
        assert state["error_type"] == expected_error.__name__
        assert "training_job" not in state
        if lost_stage == "preflight":
            assert "preflight_job" not in state
        else:
            assert state["preflight_job"] == "100.ccpbs1"
    else:
        state = launcher.submit_run(output, infrastructure)
        assert state["status"] == "submitted"
        assert state["preflight_job"] == "100.ccpbs1" and state["training_job"] == "101.ccpbs1"
        assert state["dependency"] == "afterok:100.ccpbs1"
    retained = journal.read_bytes()
    previous_calls = list(calls)
    if lost_stage:
        with pytest.raises(FileExistsError, match="requires reconciliation"):
            launcher.submit_run(output, infrastructure)
    else:
        assert launcher.submit_run(output, infrastructure) == state
    assert calls == previous_calls
    assert journal.read_bytes() == retained
    assert (run_root / "source.tar.gz").read_bytes() == (output / "source.tar.gz").read_bytes()
    assert launcher.load_json(run_root / "staging_manifest.json")["status"] == "complete"
    launcher.atomic_json(run_root / "output/preflight/stage_status.json", {"status": "complete"})
    launcher.atomic_json(run_root / "output/train/stage_status.json", {"status": "paused"})
    if lost_stage:
        with pytest.raises(RuntimeError, match="bound submission"):
            launcher.submit_run(output, infrastructure, resume=True)
        assert calls == previous_calls
    else:
        assert launcher.submit_run(output, infrastructure) == state
        assert calls == previous_calls
        assert launcher.submit_run(output, infrastructure, resume=True)["resume_jobs"] == ["102.ccpbs1"]
        assert launcher.inspect_run(run_root, packaged)["stages"]["train"]["status"] == "complete"
        with pytest.raises(RuntimeError, match="exact paused run"):
            launcher.submit_run(output, infrastructure, resume=True)
        assert calls == previous_calls + ["resume"]
        monkeypatch.setattr(launcher, "INFRA_ROOT", infrastructure)
        monkeypatch.setattr(sys, "argv", ["submit", "--payload", str(output), "--key", "unused"])

        def transport(command, **kwargs):
            assert command[0] == "ssh" and kwargs["input"] == SUBMIT_PATH.with_name("launcher.py").read_bytes()
            monkeypatch.setattr(sys, "argv", ["launcher", *command[-1].split(" - ", 1)[1].split()])
            launcher.main()
            observed = capsys.readouterr().out.encode()
            return subprocess.CompletedProcess(command, 0, stdout=observed)

        monkeypatch.setattr(submit.subprocess, "run", transport)
        submit.main()
        assert json.loads(capsys.readouterr().out)["resume_jobs"] == ["102.ccpbs1"]
        destination = tmp_path / "retrieved"
        with pytest.raises(ValueError, match="worker digest"):
            submit.retrieve_declared(packaged, launcher.inspect_run(run_root, packaged), ["scp"], "ims", destination)
        monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0))
        for stage in launcher.stage_names(packaged):
            launcher.atomic_json(run_root / "output" / stage / "complete.json", {"stage": stage})
            launcher.worker(run_root, stage)
        observed = launcher.inspect_run(run_root, packaged)
        def copy_output(command):
            relative = command[-2].split("/output/", 1)[1]
            Path(command[-1]).write_bytes((run_root / "output" / relative).read_bytes())
        monkeypatch.setattr(submit, "run", lambda command: Path(command[-1]).write_bytes(b"wrong"))
        with pytest.raises(ValueError, match="differs from worker hash"):
            submit.retrieve_declared(packaged, observed, ["scp"], "ims", destination)
        assert not destination.exists()
        monkeypatch.setattr(submit, "run", copy_output)
        submit.retrieve_declared(packaged, observed, ["scp"], "ims", destination)
        monkeypatch.setattr(submit, "run", lambda command: pytest.fail("Retrieval must reuse verified bytes"))
        submit.retrieve_declared(packaged, observed, ["scp"], "ims", destination)
        assert json.loads((destination / "retrieval.json").read_bytes())["observation"] == observed


@pytest.mark.parametrize("assigned,index,selected", [("4,7", 1, "7"), ("GPU-fixture", 0, "GPU-fixture")])
def test_family_bridge_uses_only_scheduler_assigned_device(monkeypatch, assigned, index, selected):
    from molgap import experiment_training_worker
    spec = importlib.util.spec_from_file_location("ims_family_bridge", ROOT / "platforms/ims/run_family_training.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    calls = []
    monkeypatch.setattr(experiment_training_worker, "main", lambda argv: calls.append(argv) or "fixture")
    assert bridge.main(["--device", str(index), "--prepared", "synthetic.json"]) == "fixture"
    assert calls == [["--prepared", "synthetic.json"]]
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == selected


@pytest.mark.parametrize("assigned,index", [("", 0), ("4,7", 2), ("4", -1),
    ("4,,7", 1), ("4,", 1), ("4, 7", 1)])
def test_family_bridge_rejects_unassigned_device_without_dispatch(monkeypatch, assigned, index):
    from molgap import experiment_training_worker
    spec = importlib.util.spec_from_file_location("ims_family_bridge_reject", ROOT / "platforms/ims/run_family_training.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    monkeypatch.setattr(experiment_training_worker, "main", lambda argv: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="scheduler"):
        bridge.main(["--device", str(index)])


def test_dashboard_snapshot_workflow(tmp_path, monkeypatch):
    """Real parsers, atomic files and HTTP; only SSH is replaced."""
    import concurrent.futures
    import threading
    import time
    import urllib.request
    spec = importlib.util.spec_from_file_location("ims_dashboard", ROOT / "platforms/ims/check_queue.py")
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    sections = {
        "JOBINFO_C": "Job Status at 2026-11-24 12:00:00\nH 123 model Run 16 sm2/ew6 1:00 node-secret",
        "JOBINFO_M": "gpu 123 model Run 16+1 sm2/ew6 100 200",
        "JOBINFO_W": 'Job Status at 2026-11-24 12:00:00\nH 123 model Run "/lustre/home/users/sm2/chou/molgap-v4-submit/runs/example"',
        "JOBINFO_S": "Job Status at 2026-11-24 12:00:00\nUser/Group Stat:\nuser(sm2) group(ew6)\nGPUs | 1/0/0/6 | 1/0/0/6\nQueue Status (H)\nJobs within a week\ngpu | 2 | 32(2) | 3 | 48(3)",
        "SHOWLIM_C": "CPU resource information: Group=ew6\nTOTAL 1000 100 50 50",
        "CHOU_RUNS": '{"runs": [{"training_job": "123", "items": {"a": {"status": "NOT_STARTED"}, "b": {"status": "SKIPPED_FOR_TIME"}}}]}',
    }
    calls = []
    def ssh(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["timeout"] == 45
        out = "\n".join(f"===__SEC_{k}__===\n{v}\n===__RC_{k}__0===" for k,v in sections.items())
        return subprocess.CompletedProcess(args, 0, out.encode(), b"")
    monkeypatch.setattr(dashboard.subprocess, "run", ssh)
    conn = dict(host="offline", port=22, user="sm2", key_path=tmp_path / "unused", remote_root=dashboard.REMOTE_BOUNDARY)
    path = tmp_path / "status.json"
    reader = dashboard.SnapshotState(path)
    assert not reader.get()["available"]
    assert dashboard.collect_once(path, **conn)
    result = reader.get()
    assert result["summary"]["gpu_within_week"]["jobs_waiting"] == 3
    assert result["jobs"][0]["job_id"] == "123" and not result["stale"]
    assert [i["status"] for i in result["chou_runs"][0]["items"]] == ["NOT_STARTED", "SKIPPED_FOR_TIME"]
    assert not any(secret in path.read_text() for secret in ["node-secret", "sm2", "ew6", "/lustre", "raw", "workdir"])
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.make_handler(reader))
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/status?refresh=1"
        before = len(calls)
        def get(_):
            with urllib.request.urlopen(url) as response:
                return json.load(response)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            assert all(r["jobs"] for r in pool.map(get, range(16)))
        assert len(calls) == before
    finally:
        server.shutdown(); server.server_close(); worker.join()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 45)
    monkeypatch.setattr(dashboard.subprocess, "run", timeout)
    assert not dashboard.collect_once(tmp_path / "never.json", **conn)
    assert not dashboard.SnapshotState(tmp_path / "never.json").get()["available"]
    assert not dashboard.collect_once(path, **conn)
    assert reader.get()["stale"] and reader.get()["failure"] == "ssh_timeout"
    assert reader.get()["jobs"] == result["jobs"]
    monkeypatch.setattr(dashboard.subprocess, "run", ssh)
    sections["CHOU_RUNS"] = "bad json"
    assert not dashboard.collect_once(path, **conn)
    assert reader.get()["failure"] == "parse"
    local = dashboard.MonitorState(**conn, min_interval=15).get()
    assert local["jobs"][0]["job_id"] == "123" and "CHOU_RUNS" in local["error"]
    assert "123" in dashboard.render_terminal(local)
    sections["CHOU_RUNS"] = '{"runs": []}'
    original = ssh
    def partial(*args, **kwargs):
        proc = original(*args, **kwargs)
        proc.stdout = proc.stdout.replace(b"__RC_JOBINFO_M__0", b"__RC_JOBINFO_M__1")
        return proc
    monkeypatch.setattr(dashboard.subprocess, "run", partial)
    assert not dashboard.collect_once(path, **conn)
    assert reader.get()["failure"] == "partial" and reader.get()["jobs"]
    monkeypatch.setattr(dashboard.subprocess, "run", ssh)
    for name in ("JOBINFO_C", "JOBINFO_M", "JOBINFO_W"):
        sections[name] = "Job Status at 2026-11-24 12:00:00"
    assert dashboard.collect_once(path, **conn)
    assert reader.get()["available"] and reader.get()["jobs"] == []
    envelope = json.loads(path.read_text()); envelope["snapshot"]["fetched_at_epoch"] = time.time() - 300
    dashboard.atomic_snapshot(path, envelope)
    assert reader.get()["stale"]
    path.write_text("broken")
    assert reader.get()["failure"] == "snapshot_unreadable" and not reader.get()["available"]
    before = len(calls)
    with pytest.raises(ValueError):
        dashboard.fetch_snapshot(**{**conn, "remote_root": dashboard.REMOTE_BOUNDARY + "-other"})
    assert len(calls) == before
    sections["JOBINFO_C"] += "\nH 123 example Run 16 sm2/ew6 1:00 node-name"
    full_path = tmp_path / "full.json"
    assert dashboard.collect_once(full_path, full=True, **conn)
    reader = dashboard.SnapshotState(full_path, full=True)
    before = len(calls)
    result = reader.get(force=True)
    assert result["jobs"][0]["name"] == "example"
    assert "node-name" in result["raw"]["jobinfo_c"]
    assert len(calls) == before
    monkeypatch.setattr(dashboard.subprocess, "run", timeout)
    assert not dashboard.collect_once(full_path, full=True, **conn)
    assert reader.get()["stale"] and reader.get()["jobs"] == result["jobs"]
