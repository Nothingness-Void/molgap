"""CPU-only contract tests for the bounded Modern2D qualification runtime.

These tests deliberately stop at the runtime boundary.  They use tiny JSON,
tar, and subprocess fixtures; no GPU, network, SDK, model, graph, or user
dataset is required.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import types
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "src/molgap/modern2d_qualification_runtime.py"
ENTRY_PATH = ROOT / "platforms/kaggle/run_modern2d_qualification.py"
HEX = "a" * 64


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    return _load_module(RUNTIME_PATH, "_test_modern2d_qualification_runtime")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _sha(path)


def _launch_payload(runtime, *, max_wall_seconds: int = 1800) -> dict:
    return {
        "format": runtime.LAUNCH_FORMAT,
        "spec_identity": "1" * 64,
        "expected_package_identity": "2" * 64,
        "expected_source_archive_sha256": "3" * 64,
        "account": "fixture",
        "run_reference": "fixture/run",
        "authority": {"path": "authority.json", "sha256": "4" * 64},
        "private_asset": {
            "dataset_source": "fixture/accepted-data",
            "mount": "accepted-data",
            "data_root": "data",
            "acceptance_sha256": "5" * 64,
            "version_identity": "6" * 64,
            "train_logical_content_sha256": "7" * 64,
            "development_logical_content_sha256": "8" * 64,
        },
        "max_wall_seconds": max_wall_seconds,
        "jobs": [
            {
                "arm_id": "ppgt_author_v2",
                "device": 0,
                "context": {"path": "contexts/ppgt.json", "sha256": "9" * 64},
                "recipe": {"path": "recipes/ppgt.json", "sha256": "a" * 64},
                "initial_state": {"path": "state/ppgt.pt", "sha256": "b" * 64},
            },
            {
                "arm_id": "esa_native_v3",
                "device": 1,
                "context": {"path": "contexts/esa.json", "sha256": "c" * 64},
                "recipe": {"path": "recipes/esa.json", "sha256": "d" * 64},
                "initial_state": {"path": "state/esa.pt", "sha256": "e" * 64},
            },
        ],
    }


def _write_launch(root: Path, payload: dict) -> tuple[Path, str]:
    path = root / "modern2d_qualification_launch.json"
    _write_json(path, payload)
    return path, _sha(path)


def test_runtime_module_and_read_launch_are_stdlib_first_and_digest_bound(tmp_path, monkeypatch):
    """The frozen read path is importable before the package/model stack."""

    real_import = builtins.__import__
    stdlib = set(sys.stdlib_module_names)

    def stdlib_only(name, *args, **kwargs):
        if name.partition(".")[0] not in stdlib:
            raise AssertionError(f"non-stdlib import during frozen launch read: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stdlib_only)
    isolated = _load_module(RUNTIME_PATH, "_test_modern2d_stdlib_runtime")
    payload = _launch_payload(isolated)
    launch, digest = _write_launch(tmp_path, payload)

    assert isolated.read_launch(launch, digest) == payload
    with pytest.raises(ValueError, match="pinned entrypoint"):
        isolated.read_launch(launch, "f" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(scope="QUALIFICATION_ONLY"),
        lambda p: p.pop("jobs"),
        lambda p: p["jobs"][0].update(arm_id="unexpected"),
        lambda p: p["jobs"].__setitem__(1, copy.deepcopy(p["jobs"][0])),
        lambda p: p["jobs"][0].update(device=True),
        lambda p: p.update(max_wall_seconds=1801),
        lambda p: p.update(max_wall_seconds=0),
        lambda p: p.update(max_wall_seconds=True),
    ],
    ids=["extra-field", "missing-field", "wrong-arm", "duplicate-arm", "bool-device",
         "wall-too-large", "wall-too-small", "bool-wall"],
)
def test_read_launch_enforces_exact_fields_two_roles_devices_and_wall(runtime, tmp_path, mutation):
    payload = _launch_payload(runtime)
    mutation(payload)
    launch, digest = _write_launch(tmp_path, payload)

    with pytest.raises((ValueError, TypeError)):
        runtime.read_launch(launch, digest)

    assert runtime._FIELDS == set(_launch_payload(runtime))
    assert runtime.INTERFACES == {
        "ppgt_author_v2": ("ppgt_small", "2"),
        "esa_native_v3": ("esa_small", "3"),
    }


@pytest.mark.parametrize(
    "relative",
    [
        "../escape",
        "a/../../escape",
        "a//b",
        "a/./b",
        "a/../b",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "a?.json",
        "a/child.",
        "a/child ",
    ],
)
def test_confined_paths_reject_traversal_and_windows_ambiguous_names(runtime, tmp_path, relative):
    with pytest.raises(ValueError):
        runtime._under(tmp_path, relative)


def test_confined_paths_reject_linked_components(runtime, tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        # The contract is still tested on hosts where creating a Windows
        # symlink needs an unavailable privilege.
        real_is_symlink = Path.is_symlink

        def mocked_is_symlink(path):
            return path == link or real_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", mocked_is_symlink)

    with pytest.raises(ValueError, match="Links are forbidden"):
        runtime._under(tmp_path, "linked/file.json")


def test_expired_worker_budget_rejects_before_graph_or_model_import(runtime, tmp_path, monkeypatch):
    payload = _launch_payload(runtime)
    fake_contexts = {job["arm_id"]: {} for job in payload["jobs"]}
    monkeypatch.setattr(
        runtime,
        "validate_launch",
        lambda **_: (payload, object(), {}, fake_contexts, tmp_path, tmp_path / "authority.json"),
    )

    real_import = builtins.__import__

    def no_payload_import(name, *args, **kwargs):
        if name.startswith(("torch", "rdkit", "torch_geometric", "kaggle")) or name in {
            "molgap.modern2d_qualification",
            "molgap.family_training_runner",
        }:
            raise AssertionError(f"payload/model import occurred before budget rejection: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_payload_import)
    args = SimpleNamespace(
        device=0,
        launch=tmp_path / "launch.json",
        expected_launch_sha256=HEX,
        source_root=tmp_path,
        package_dir=tmp_path,
        mounted_root=tmp_path,
        arm="ppgt_author_v2",
        output=tmp_path / "scratch",
        allocation_started=time.perf_counter() - 10_000,
    )

    with pytest.raises(TimeoutError, match="deadline exhausted"):
        runtime.run_worker(args)


def test_fresh_output_is_rejected_before_hardware_or_children(runtime, tmp_path, monkeypatch):
    output = tmp_path / "qualification"
    output.mkdir()
    (output / "sentinel").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(runtime, "_observe_hardware", lambda *_: pytest.fail("hardware was probed"))

    with pytest.raises(ValueError, match="fresh output root"):
        runtime.run_qualification_pair(
            source_root=tmp_path,
            package_dir=tmp_path,
            mounted_root=tmp_path,
            launch_path=tmp_path / "missing-launch.json",
            expected_launch_sha256=HEX,
            output=output,
            runtime_root=tmp_path / "runtime",
        )

    assert (output / "sentinel").read_text(encoding="utf-8") == "preserve"


class _FakeChild:
    _next_pid = 41_000

    def __init__(self):
        self.pid = _FakeChild._next_pid
        _FakeChild._next_pid += 1
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_stop_owned_kills_only_live_owned_processes(runtime):
    def spawn(code):
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        return subprocess.Popen([sys.executable, "-c", code], **kwargs)

    finished = spawn("raise SystemExit(7)")
    assert finished.wait(timeout=10) == 7
    owned = spawn("import time; time.sleep(30)")
    unrelated = spawn("import time; time.sleep(30)")
    try:
        runtime._stop_owned([finished, owned])
        assert finished.returncode == 7
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.terminate()
            owned.wait(timeout=5)
        if unrelated.poll() is None:
            unrelated.terminate()
            unrelated.wait(timeout=5)


def test_parent_timeout_places_parent_log_and_atomically_seals_failure_cost(runtime, tmp_path, monkeypatch):
    payload = _launch_payload(runtime, max_wall_seconds=120)
    launch, digest = _write_launch(tmp_path, payload)
    output = tmp_path / "qualification"
    children = []
    popen_calls = []
    stopped = []
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "99")

    monkeypatch.setattr(runtime, "_observe_hardware", lambda *_: ["Tesla T4", "Tesla T4"])
    monkeypatch.setattr(
        runtime,
        "validate_launch",
        lambda **_: (payload, None, None, {}, tmp_path, tmp_path / "authority.json"),
    )
    def fake_popen(*args, **kwargs):
        process = _FakeChild()
        children.append(process)
        popen_calls.append(kwargs)
        return process

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)
    wait_calls = 0

    def timeout_wait(children, *, deadline, tick):
        nonlocal wait_calls
        tick()
        if wait_calls == 0:
            wait_calls += 1
            children[0].returncode = 0
            _write_json(output / "environment_result.json", {"executable": sys.executable})
            _write_json(output / "environment_qualification.json", {
                "packages": {"torch": "2.7.1+cu126"},
                "status": "IMPORTS_QUALIFIED_NOT_GPU_CALIBRATED",
            })
            return
        raise TimeoutError("synthetic parent timeout")

    monkeypatch.setattr(runtime, "_wait", timeout_wait)

    def stop_only_live(children):
        stopped.extend(process for process in children if process.poll() is None)
        for process in stopped:
            process.terminate()

    monkeypatch.setattr(runtime, "_stop_owned", stop_only_live)

    with pytest.raises(TimeoutError, match="synthetic parent timeout"):
        runtime.run_qualification_pair(
            source_root=tmp_path,
            package_dir=tmp_path,
            mounted_root=tmp_path,
            launch_path=launch,
            expected_launch_sha256=digest,
            output=output,
            runtime_root=tmp_path / "runtime",
        )

    assert len(popen_calls) == 3
    assert popen_calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert popen_calls[0]["env"]["OMP_NUM_THREADS"] == "1"
    assert popen_calls[0]["env"]["MKL_NUM_THREADS"] == "1"
    assert popen_calls[0]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    for call, device in zip(popen_calls[1:], ("0", "1")):
        assert call["env"]["CUDA_VISIBLE_DEVICES"] == device
        assert call["env"]["OMP_NUM_THREADS"] == "1"
        assert call["env"]["MKL_NUM_THREADS"] == "1"
        assert call["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert stopped == children[1:]
    assert all(process.terminate_calls == 1 for process in children[1:])
    assert children[0].returncode == 0
    assert (output / "bootstrap.log").is_file()
    assert not (output / "ppgt_author_v2" / "bootstrap.log").exists()
    assert not (output / "esa_native_v3" / "bootstrap.log").exists()

    failure = json.loads((output / "failure.log").read_text(encoding="utf-8"))
    assert "synthetic parent timeout" in failure["traceback"]
    report = json.loads((output / "execution_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["phase"] == "sealed"
    assert report["scope"] == "QUALIFICATION_ONLY"
    assert report["formal_training_exposure"] == 0
    assert report["native_execution_released"] is False
    ledger = json.loads((output / "allocation_ledger.json").read_text(encoding="utf-8"))
    assert ledger["status"] == "failed"
    manifest = json.loads((output / "qualification_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["scope"] == "QUALIFICATION_ONLY"
    assert manifest["scientific_acceptance"] == "NOT_EVALUATED"
    assert manifest["native_execution_released"] is False
    assert not list(output.glob("*.tmp"))


def _tar_member(name: str, data: bytes, *, symlink: str | None = None) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    if symlink is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = symlink
        info.size = 0
    else:
        info.size = len(data)
    return info


def _write_source_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for info, data in members:
            bundle.addfile(info, io.BytesIO(data) if info.size else None)


def test_bootstrap_checks_launch_marker_and_uses_one_frozen_extractor(runtime, tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    archive = stage / "source_payload.bin"
    extractor = (
        b"from pathlib import Path\n"
        b"def _unpack(package, source):\n"
        b"    target = source / 'src' / 'molgap'\n"
        b"    target.mkdir(parents=True)\n"
        b"    (target / 'frozen_marker.txt').write_text('ok', encoding='utf-8')\n"
    )
    _write_source_archive(
        archive,
        [(_tar_member("src/molgap/experiment_preflight.py", extractor), extractor)],
    )
    payload = _launch_payload(runtime)
    payload["expected_source_archive_sha256"] = _sha(archive)
    launch, digest = _write_launch(stage, payload)
    for name in (
        "experiment_spec.json",
        "package_manifest.json",
        "SOURCE_COMMIT.txt",
        "SOURCE_ARCHIVE_SHA256.txt",
        "SOURCE_FILES.json",
    ):
        (stage / name).write_text("fixture", encoding="utf-8")

    captured = {}

    def fake_pair(**kwargs):
        captured.update(kwargs)
        return {"status": "complete", "scope": "QUALIFICATION_ONLY"}

    monkeypatch.setattr(runtime, "run_qualification_pair", fake_pair)
    result = runtime.bootstrap(
        launch_path=launch,
        expected_launch_sha256=digest,
        allocation_started=time.perf_counter(),
        mounted_root=tmp_path / "mounted",
        output=tmp_path / "output",
        scratch_root=tmp_path / "scratch",
    )

    assert result["status"] == "complete"
    source = tmp_path / "scratch" / "source"
    package = tmp_path / "scratch" / "package"
    assert (source / "src/molgap/frozen_marker.txt").read_text(encoding="utf-8") == "ok"
    assert (package / "source.tar.gz").is_file()
    assert captured["source_root"] == source
    assert captured["package_dir"] == package
    assert captured["output"] == tmp_path / "output"

    duplicate = tmp_path / "duplicate.tar.gz"
    _write_source_archive(
        duplicate,
        [
            (_tar_member("src/molgap/experiment_preflight.py", extractor), extractor),
            (_tar_member("src/molgap/experiment_preflight.py", extractor), extractor),
        ],
    )
    duplicate_payload = _launch_payload(runtime)
    duplicate_payload["expected_source_archive_sha256"] = _sha(duplicate)
    duplicate_stage = tmp_path / "duplicate-stage"
    duplicate_stage.mkdir()
    (duplicate_stage / "source_payload.bin").write_bytes(duplicate.read_bytes())
    duplicate_launch, duplicate_digest = _write_launch(duplicate_stage, duplicate_payload)
    for name in (
        "experiment_spec.json",
        "package_manifest.json",
        "SOURCE_COMMIT.txt",
        "SOURCE_ARCHIVE_SHA256.txt",
        "SOURCE_FILES.json",
    ):
        (duplicate_stage / name).write_text("fixture", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or duplicated"):
        runtime.bootstrap(
            launch_path=duplicate_launch,
            expected_launch_sha256=duplicate_digest,
            allocation_started=time.perf_counter(),
            mounted_root=tmp_path / "mounted",
            output=tmp_path / "duplicate-output",
            scratch_root=tmp_path / "duplicate-scratch",
        )


def _validation_case(runtime, tmp_path: Path, *, layout="flat"):
    stage = tmp_path / "stage"
    source = tmp_path / "source"
    package = tmp_path / "package"
    mounted = tmp_path / "mounted"
    if layout == "flat":
        dataset_root = mounted / "accepted-data"
    elif layout == "owner_qualified":
        dataset_root = mounted / "fixture" / "accepted-data"
    elif layout == "owner_namespaced":
        dataset_root = mounted / "datasets" / "fixture" / "accepted-data"
    else:
        raise AssertionError(f"unsupported fixture layout: {layout}")
    data_root = dataset_root / "data"
    for path in (stage, source / "recipes", package, dataset_root, data_root):
        path.mkdir(parents=True, exist_ok=True)

    acceptance = {
        "rule_identity": "6" * 64,
        "status": "DATA_ROLE_ACCEPTED",
        "scope": "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",
        "roles": {
            "train_100k": {"logical_content_sha256": "7" * 64, "sha256": "1" * 64},
            "common_dev": {"logical_content_sha256": "8" * 64, "sha256": "2" * 64},
        },
    }
    acceptance_path = data_root / "acceptance.json"
    acceptance_sha = _write_json(acceptance_path, acceptance)
    binding = {
        "format": "molgap-indexed-training-input-v1",
        "scope": "ACCEPTED_INDEXED_INPUT_V1",
        "data_root": "data",
    }
    binding_path = dataset_root / "indexed-binding.json"
    binding_sha = _write_json(binding_path, binding)

    recipes = {}
    recipe_digests = {}
    for arm, filename in (("ppgt_author_v2", "ppgt.json"), ("esa_native_v3", "esa.json")):
        recipe = {
            "qualification": {
                "purpose": "numerical-resume-resources-only",
                "steps_per_path": 2,
                "scratch_updates_per_model": 6,
            },
            "dataset": {"manifest": "indexed-binding.json", "manifest_sha256": binding_sha},
        }
        path = source / "recipes" / filename
        recipe_digests[arm] = _write_json(path, recipe)
        recipes[arm] = recipe

    spec_identity = "1" * 64
    package_identity = "2" * 64
    archive_identity = "3" * 64
    account = "fixture"
    run_reference = "fixture/run"
    authority = {
        "purpose": "EXECUTION_QUALIFICATION_ONLY",
        "user_authorized": True,
        "gpu": {"owner": account},
        "scientific_training_authorized": False,
        "protected_role_access_authorized": False,
        "extra_seed_authorized": False,
        "automatic_successor_authorized": False,
    }
    authority_path = _write_json(stage / "authority.json", authority)

    context_records = {}
    expected_contexts = {}
    jobs = []
    source_inventory = []
    for index, (arm, recipe_name) in enumerate(
        (("ppgt_author_v2", "ppgt.json"), ("esa_native_v3", "esa.json"))
    ):
        recipe_sha = recipe_digests[arm]
        expected_context = SimpleNamespace(
            training_recipe_sha256=recipe_sha,
            to_dict=lambda arm=arm: {"arm_id": arm, "reviewed": True},
        )
        expected_contexts[arm] = expected_context
        train_manifest_sha = "1" * 64
        dev_manifest_sha = "2" * 64
        context = {
            "run_context": expected_context.to_dict(),
            "authority_sha256": authority_path,
            "max_wall_seconds": 1800,
            "train_permit": {
                "role": "train_100k",
                "acceptance_sha256": acceptance_sha,
                "version_identity": "6" * 64,
                "manifest_sha256": train_manifest_sha,
            },
            "development_permit": {
                "role": "common_dev",
                "acceptance_sha256": acceptance_sha,
                "version_identity": "6" * 64,
                "manifest_sha256": dev_manifest_sha,
            },
        }
        context_path = stage / "contexts" / f"{arm}.json"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_digest = _write_json(context_path, context)
        initial_path = stage / "state" / f"{arm}.pt"
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_bytes(b"tiny-state-" + arm.encode())
        initial_digest = _sha(initial_path)
        relative_recipe = f"recipes/{recipe_name}"
        source_inventory.append({"path": relative_recipe, "sha256": recipe_sha})
        jobs.append(
            {
                "arm_id": arm,
                "device": index,
                "context": {"path": f"contexts/{arm}.json", "sha256": context_digest},
                "recipe": {"path": relative_recipe, "sha256": recipe_sha},
                "initial_state": {"path": f"state/{arm}.pt", "sha256": initial_digest},
            }
        )

    spec = SimpleNamespace(
        identity=spec_identity,
        to_dict=lambda: {
            "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2},
            "arms": [
                {"arm_id": "ppgt_author_v2", "family": {"name": "ppgt_small", "version": "2"}},
                {"arm_id": "esa_native_v3", "family": {"name": "esa_small", "version": "3"}},
            ],
        },
    )
    _write_json(package / "experiment_spec.json", {"identity": spec_identity})
    _write_json(package / "SOURCE_FILES.json", {"files": source_inventory})

    payload = _launch_payload(runtime)
    payload.update(
        {
            "spec_identity": spec_identity,
            "expected_package_identity": package_identity,
            "expected_source_archive_sha256": archive_identity,
            "authority": {"path": "authority.json", "sha256": authority_path},
            "private_asset": {
                **payload["private_asset"],
                "acceptance_sha256": acceptance_sha,
            },
            "jobs": jobs,
        }
    )
    launch, launch_digest = _write_launch(stage, payload)
    manifest = {
        "package_identity": package_identity,
        "archive_sha256": archive_identity,
        "spec_identity": spec_identity,
        "source_commit": "4" * 40,
    }
    return SimpleNamespace(
        stage=stage,
        source=source,
        package=package,
        mounted=mounted,
        dataset_root=dataset_root,
        layout=layout,
        data_root=data_root,
        acceptance=acceptance,
        acceptance_path=acceptance_path,
        binding_path=binding_path,
        launch=launch,
        launch_digest=launch_digest,
        payload=payload,
        manifest=manifest,
        spec=spec,
        expected_contexts=expected_contexts,
        context_records=context_records,
        authority_path=stage / "authority.json",
    )


def _install_validation_mocks(monkeypatch, case):
    package_module = types.ModuleType("molgap.experiment_package")
    package_module.verify_experiment_source_package = lambda path: case.manifest
    spec_module = types.ModuleType("molgap.experiment_spec")

    class FakeExperimentSpec:
        @classmethod
        def from_json(cls, path):
            return case.spec

    spec_module.ExperimentSpec = FakeExperimentSpec
    workflow_module = types.ModuleType("molgap.experiment_family_workflow")

    class FakeRunContext:
        @classmethod
        def for_training(cls, spec, package_dir, *, arm_id, **kwargs):
            return case.expected_contexts[arm_id]

    workflow_module.RunContext = FakeRunContext
    monkeypatch.setitem(sys.modules, "molgap.experiment_package", package_module)
    monkeypatch.setitem(sys.modules, "molgap.experiment_spec", spec_module)
    monkeypatch.setitem(sys.modules, "molgap.experiment_family_workflow", workflow_module)


def test_validate_launch_binds_selection_metadata_hashes_and_logical_roles_without_labels(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    label_path = case.data_root / "labels.json"
    label_path.write_text("this must never be opened", encoding="utf-8")
    real_open = Path.open

    def no_label_open(path, *args, **kwargs):
        if path == label_path:
            raise AssertionError("label payload was read during selection metadata validation")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_label_open)
    result = runtime.validate_launch(
        launch_path=case.launch,
        expected_launch_sha256=case.launch_digest,
        source_root=case.source,
        package_dir=case.package,
        mounted_root=case.mounted,
    )

    assert result[0]["jobs"] == case.payload["jobs"]
    assert result[3].keys() == {"ppgt_author_v2", "esa_native_v3"}
    assert result[4] == case.dataset_root
    assert result[4] / "data" == case.data_root


@pytest.mark.parametrize(
    "layout, selected_root",
    [("flat", "accepted-data"),
     ("owner_qualified", "fixture/accepted-data"),
     ("owner_namespaced", "datasets/fixture/accepted-data")],
)
def test_validate_launch_accepts_declared_dataset_mount_layouts(
        runtime, tmp_path, monkeypatch, layout, selected_root):
    case = _validation_case(runtime, tmp_path, layout=layout)
    _install_validation_mocks(monkeypatch, case)
    discovery_report = {}

    result = runtime.validate_launch(
        launch_path=case.launch,
        expected_launch_sha256=case.launch_digest,
        source_root=case.source,
        package_dir=case.package,
        mounted_root=case.mounted,
        discovery_report=discovery_report,
    )

    assert result[4] == case.dataset_root
    assert discovery_report["status"] == "selected"
    assert discovery_report["candidate_count"] == 1
    assert discovery_report["selected_root"] == selected_root
    assert len(discovery_report["candidates"]) == 3
    assert discovery_report["recursive_scan"] is False
    assert discovery_report["file_contents_read"] is False


def test_validate_launch_rejects_both_dataset_mount_layouts_before_payload(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path, layout="flat")
    _install_validation_mocks(monkeypatch, case)
    (case.mounted / "datasets" / "fixture" / "accepted-data").mkdir(parents=True)
    discovery_report = {}

    with pytest.raises(ValueError, match="exactly one dataset mount"):
        runtime.validate_launch(
            launch_path=case.launch,
            expected_launch_sha256=case.launch_digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
            discovery_report=discovery_report,
        )

    assert discovery_report["status"] == "rejected"
    assert discovery_report["candidate_count"] == 2
    assert len(discovery_report["candidates"]) == 3
    assert discovery_report["file_contents_read"] is False


def test_validate_launch_rejects_wrong_dataset_mount_slug_before_payload(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    changed = copy.deepcopy(case.payload)
    changed["private_asset"]["mount"] = "wrong-data"
    launch = case.stage / "wrong-mount-launch.json"
    digest = _write_json(launch, changed)

    with pytest.raises(ValueError, match="exact dataset_source slug"):
        runtime.validate_launch(
            launch_path=launch,
            expected_launch_sha256=digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
        )


def test_validate_launch_rejects_acceptance_sha_tamper_before_payload(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    payload_path = case.data_root / "opaque-label-payload.bin"
    payload_path.write_bytes(b"must not be opened")
    real_open = Path.open

    def reject_payload(path, *args, **kwargs):
        if path == payload_path:
            raise AssertionError("payload was opened before acceptance SHA rejection")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload)
    case.acceptance_path.write_bytes(case.acceptance_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="accepted-data receipt differs"):
        runtime.validate_launch(
            launch_path=case.launch,
            expected_launch_sha256=case.launch_digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
        )


def test_validate_launch_rejects_binding_tamper_before_payload(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    payload_path = case.data_root / "opaque-label-payload.bin"
    payload_path.write_bytes(b"must not be opened")
    real_open = Path.open

    def reject_payload(path, *args, **kwargs):
        if path == payload_path:
            raise AssertionError("payload was opened before binding rejection")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload)
    case.binding_path.write_bytes(case.binding_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Exact indexed binding is absent"):
        runtime.validate_launch(
            launch_path=case.launch,
            expected_launch_sha256=case.launch_digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
        )


def test_validate_launch_rejects_protected_role_authority_before_payload(
        runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    payload_path = case.data_root / "opaque-label-payload.bin"
    payload_path.write_bytes(b"must not be opened")
    real_open = Path.open

    def reject_payload(path, *args, **kwargs):
        if path == payload_path:
            raise AssertionError("payload was opened before protected-role rejection")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload)
    authority = json.loads(case.authority_path.read_text(encoding="utf-8"))
    authority["protected_role_access_authorized"] = True
    authority_sha = _write_json(case.authority_path, authority)
    changed = copy.deepcopy(case.payload)
    changed["authority"]["sha256"] = authority_sha
    launch = case.stage / "protected-role-launch.json"
    digest = _write_json(launch, changed)

    with pytest.raises(ValueError, match="Qualification-only authority required"):
        runtime.validate_launch(
            launch_path=launch,
            expected_launch_sha256=digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
        )


def test_validate_launch_rejects_logical_role_hash_drift_before_payload(runtime, tmp_path, monkeypatch):
    case = _validation_case(runtime, tmp_path)
    _install_validation_mocks(monkeypatch, case)
    changed = copy.deepcopy(case.payload)
    changed["private_asset"]["train_logical_content_sha256"] = "f" * 64
    launch = case.stage / "changed-launch.json"
    _write_json(launch, changed)
    digest = _sha(launch)

    with pytest.raises(ValueError, match="logical identity differs"):
        runtime.validate_launch(
            launch_path=launch,
            expected_launch_sha256=digest,
            source_root=case.source,
            package_dir=case.package,
            mounted_root=case.mounted,
        )


def test_worker_calls_only_bounded_reference_preflight_and_never_formal_training(
        runtime, tmp_path, monkeypatch):
    payload = _launch_payload(runtime)
    contexts = {job["arm_id"]: {"context": job["arm_id"]} for job in payload["jobs"]}
    fake_spec = object()

    env_seen = {}

    def fake_validate(**_):
        env_seen.update({name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS")})
        return payload, fake_spec, {}, contexts, tmp_path, tmp_path / "authority.json"

    monkeypatch.setattr(
        runtime,
        "validate_launch",
        fake_validate,
    )
    monkeypatch.setattr(runtime, "_ref", lambda root, record: tmp_path / "bound.json")
    output = tmp_path / "worker-output"
    output.mkdir()
    calls = []

    fake_qualification = types.ModuleType("molgap.modern2d_qualification")

    class FakeQualificationContext:
        @classmethod
        def from_dict(cls, value):
            return SimpleNamespace(identity="context-identity")

    fake_qualification.Modern2DQualificationContext = FakeQualificationContext
    fake_runner = types.ModuleType("molgap.family_training_runner")

    def reference_preflight(**kwargs):
        calls.append(kwargs)
        assert kwargs["mode"] == "reference"
        assert kwargs["output"] == output
        certificate = {
            "format": "molgap-modern2d-qualification-certificate-v1",
            "status": "QUALIFICATION_VERIFIED",
            "scratch_optimizer_updates": 6,
            "formal_training_exposure": 0,
            "native_execution_released": False,
        }
        (output / "runtime_certificate.json").write_text(
            json.dumps(certificate, sort_keys=True), encoding="utf-8"
        )
        return certificate

    fake_runner.run_screen_preflight = reference_preflight
    monkeypatch.setitem(sys.modules, "molgap.modern2d_qualification", fake_qualification)
    monkeypatch.setitem(sys.modules, "molgap.family_training_runner", fake_runner)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    args = SimpleNamespace(
        device=0,
        launch=tmp_path / "launch.json",
        expected_launch_sha256=HEX,
        source_root=tmp_path,
        package_dir=tmp_path,
        mounted_root=tmp_path,
        arm="ppgt_author_v2",
        output=output,
        allocation_started=time.perf_counter() - 1,
    )
    runtime.run_worker(args)

    assert env_seen == {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert len(calls) == 1
    assert calls[0]["qualification"].identity == "context-identity"
    result = json.loads((output / "worker_result.json").read_text(encoding="utf-8"))
    assert result["scope"] == "QUALIFICATION_ONLY"
    assert result["native_execution_released"] is False
    assert result["context_identity"] == "context-identity"


def test_platform_entry_uses_launch_marker_and_frozen_runtime_extract(tmp_path, monkeypatch):
    entry = _load_module(ENTRY_PATH, "_test_modern2d_qualification_entry")
    input_root = tmp_path / "input"
    mount = input_root / "qualification-mount"
    output_root = tmp_path / "working"
    mount.mkdir(parents=True)
    output_root.mkdir()
    runtime_code = (
        b"def bootstrap(**kwargs):\n"
        b"    assert kwargs['expected_launch_sha256']\n"
        b"    return {'status': 'complete', 'hardware': ['Tesla T4', 'Tesla T4']}\n"
    )
    archive = mount / "source_payload.bin"
    helper_source = (ROOT / "src/molgap/kaggle_mounts.py").read_bytes()
    _write_source_archive(
        archive,
        [
            (_tar_member("src/molgap/kaggle_mounts.py", helper_source), helper_source),
            (_tar_member("src/molgap/modern2d_qualification_runtime.py", runtime_code), runtime_code),
        ],
    )
    payload = {"expected_source_archive_sha256": _sha(archive)}
    launch = mount / "modern2d_qualification_launch.json"
    launch_digest = _write_json(launch, payload)
    entry.EXPECTED_LAUNCH_SHA256 = launch_digest
    entry.EXPECTED_SOURCE_DATASET = "fixture/qualification-mount"
    entry.MOUNT_RESOLVER_SOURCE = helper_source.decode("utf-8")

    real_path = Path
    monkeypatch.setattr(
        entry,
        "Path",
        lambda raw: {
            "/kaggle/input": input_root,
            "/kaggle/working": output_root,
        }.get(str(raw), real_path(raw)),
    )
    monkeypatch.setattr(entry.subprocess, "check_output", lambda *args, **kwargs: "Tesla T4\nTesla T4\n")

    class FakeSignal:
        SIGALRM = object()
        ITIMER_REAL = object()

        @staticmethod
        def signal(*args):
            return None

        @staticmethod
        def setitimer(*args):
            return None

    monkeypatch.setattr(entry, "signal", FakeSignal)
    entry.main()

    allocation = json.loads((output_root / "bootstrap_allocation.json").read_text(encoding="utf-8"))
    assert allocation["status"] == "complete"
    assert allocation["allocation_device_count"] == 2
    assert allocation["cost_status"] == "PYTHON_OBSERVED_WINDOW_ONLY"
    assert allocation["source_discovery"]["status"] == "selected"
    assert allocation["source_discovery"]["candidate_count"] == 1
    assert not (output_root / "bootstrap_allocation.tmp").exists()

