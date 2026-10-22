"""Synthetic release regressions; no real graph roles, models or network."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import torch

from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_spec import ExperimentSpec, FAMILIES
from molgap.v4_runtime import state_dict_sha256, inspect_frozen_state_artifact
from test_experiment_package import git
from test_experiment_spec import payload


COMMON_PACKAGE_MODULES = (
    "common/__init__.py",
    "common/io.py",
    "common/spec.py",
    "common/wire.py",
    "experiment_catalog.py",
)


DATASET_PACKAGE_MODULES = (
    *COMMON_PACKAGE_MODULES,
    "family_training_data.py",
    "experiment_family_workflow.py",
    "experiment_family_artifacts.py",
    "experiment_launch.py",
    "experiment_spec.py",
    "experiment_package.py",
    "screen_policy.py",
    "v4_bundle.py",
    "training_reproducibility.py",
    "v4_runtime.py",
    "evidence_pointers.py",
    "research_memory/__init__.py",
    "research_memory/paths.py",
    "research_memory/trace.py",
    "research_memory/schemas.py",
)


@pytest.fixture
def release(tmp_path, payload, request):
    loader_mode = getattr(request, "param", "generic")
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "synthetic@example.invalid")
    git(root, "config", "user.name", "Synthetic")
    git(root, "config", "core.autocrlf", "false")
    module = root / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_bytes(b"")
    (module / "loader.py").write_bytes(b"from .child import VALUE\n")
    (module / "child.py").write_bytes(b"VALUE = 1\n")
    (root / "recipe.json").write_bytes(b'{"epochs":10}\r\n')
    allowlist = ["src/molgap/__init__.py", "src/molgap/loader.py", "src/molgap/child.py", "recipe.json"]
    if loader_mode not in {"generic", "dataset_binding", "broken_dataset_resolver"}:
        owner = Path(__file__).resolve().parents[1] / "src/molgap"
        for name in (*COMMON_PACKAGE_MODULES, "experiment_spec.py",
                     "packed_graph_dataset.py", "data/__init__.py", "data/packed_graph.py",
                     "k1_screen_training.py", "v4_runtime.py", "screen_policy.py",
                     "training_reproducibility.py"):
            source = (owner / name).read_bytes()
            if name == "k1_screen_training.py" and loader_mode == "broken":
                source += (b"\ndef _load_initial_state(path, recipe):\n"
                           b"    raise ValueError('packaged broken K1 loader')\n")
            if name == "training_reproducibility.py" and loader_mode == "broken_output_io":
                source += (b"\ndef atomic_json(path, value):\n"
                           b"    path.parent.mkdir(parents=True, exist_ok=True)\n"
                           b"    path.write_text(json.dumps(dict(value)))\n")
            if name == "training_reproducibility.py" and loader_mode == "rewrapped_output_io":
                source += (b"\ndef atomic_json(path, value):\n"
                           b"    path.parent.mkdir(parents=True, exist_ok=True)\n"
                           b"    path.write_text(json.dumps({'rows': value} if isinstance(value, list) else value))\n")
            destination = module / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source)
            allowlist.append("src/molgap/" + name)
        # Importable model owner, but any constructor use fails inside the clean worker.
        (module / "qm9_neural_atom.py").write_bytes(
            b"def make_encoder(*args, **kwargs):\n"
            b"    raise AssertionError('release check constructed a model')\n")
        allowlist.append("src/molgap/qm9_neural_atom.py")
        digest = state_dict_sha256({"weight": torch.arange(3, dtype=torch.float32)})
        recipe_bytes = json.dumps({"initialization_sha256": digest}, sort_keys=True,
                                  separators=(",", ":")).encode("ascii")
        (root / "recipe.json").write_bytes(recipe_bytes)
        candidate = payload["arms"][1]
        contract = FAMILIES[("neural_atom_k1", "2")]
        candidate["family"]["version"] = "2"
        candidate["training"]["recipe"]["name"] = contract.recipe
        candidate["training"]["sampler"]["name"] = contract.sampler
        candidate["training"]["transform"]["name"] = contract.transform
        candidate["data"]["roles"] = [dict(candidate["data"]["roles"][0], role=role)
                                      for role in contract.roles]
    dataset_manifests = None
    if loader_mode in {"dataset_binding", "broken_dataset_resolver"}:
        owner = Path(__file__).resolve().parents[1] / "src/molgap"
        for name in DATASET_PACKAGE_MODULES:
            source = (b"" if name == "research_memory/__init__.py" else
                      (owner / name).read_bytes())
            if name == "family_training_data.py" and loader_mode == "broken_dataset_resolver":
                source += (b"\ndef resolve_training_manifest(root, selection):\n"
                           b"    raise ValueError('packaged broken dataset resolver')\n")
            destination = module / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source)
            allowlist.append("src/molgap/" + name)
        dataset_manifest = tmp_path / "dataset" / "slug" / "manifest.json"
        dataset_manifest.parent.mkdir(parents=True)
        dataset_manifest.write_text('{"fixture":"metadata only"}', encoding="utf-8")
        dataset_sha = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
        recipe_record = {
            "format": "molgap-family-training-recipe-v1",
            "dataset": {
                "manifest": "slug/manifest.json",
                "manifest_sha256": dataset_sha,
                "development": False,
            },
        }
        (root / "recipe.json").write_text(
            json.dumps(recipe_record, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        dataset_manifests = {arm["arm_id"]: dataset_manifest for arm in payload["arms"]}
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    states = {}
    for arm in payload["arms"]:
        state = {"weight": torch.arange(3, dtype=torch.float32)}
        arm["initialization"]["state_sha256"] = state_dict_sha256(state)
        packaged_recipe = (root / "recipe.json").read_bytes().replace(b"\r\n", b"\n")
        arm["training"]["recipe"]["sha256"] = hashlib.sha256(packaged_recipe).hexdigest()
        path = tmp_path / (arm["arm_id"] + ".pt")
        torch.save({"model_state": state, "state_sha256": state_dict_sha256(state)}
                   if loader_mode != "generic" else state, path)
        states[arm["arm_id"]] = path
    spec = ExperimentSpec(payload)
    package = tmp_path / "package"
    manifest = build_experiment_source_package(spec, root,
        allowlist, package)
    options = {"expected_package_identity": manifest["package_identity"],
        "recipe_files": {a["arm_id"]: "recipe.json" for a in payload["arms"]},
        "initial_states": states, "required_modules": ["molgap.loader"]}
    if dataset_manifests is not None:
        options["dataset_manifests"] = dataset_manifests
    return spec, package, options


@pytest.mark.parametrize("release", ["broken"], indirect=True)
def test_generic_inspector_pass_cannot_hide_broken_packaged_k1_loader(release):
    spec, package, options = release
    arm_id = "neural_atom_k1"
    digest = spec.to_dict()["arms"][1]["initialization"]["state_sha256"]
    assert inspect_frozen_state_artifact(options["initial_states"][arm_id],
                                         expected_state_sha256=digest)["state_sha256"] == digest
    report = check_release_inputs(spec, package, **options)
    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert report["checks"]["initialization:" + arm_id]["state_sha256"] == digest
    assert report["errors"] == [{"check": "family_initialization", "item": arm_id,
                                  "message": "packaged broken K1 loader", "error_type": "ValueError"}]
    assert "family_initialization:" + arm_id not in report["checks"]
    assert report["checks"]["clean_import:package"]["molgap.k1_screen_training"] == "src/molgap/k1_screen_training.py"


@pytest.mark.parametrize("release", ["repaired"], indirect=True)
def test_packaged_k1_loader_accepts_real_wrapper_without_model_construction(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)
    assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    assert report["errors"] == []
    checks = report["checks"]
    digest = spec.to_dict()["arms"][1]["initialization"]["state_sha256"]
    assert checks["family_initialization:neural_atom_k1"] == {
        "state_sha256": digest, "tensor_count": 1, "device": "cpu"}
    assert checks["family_output_io:neural_atom_k1"] == {
        "object_roundtrip": True, "timing_array_roundtrip": True, "device": "cpu"}
    assert "family_initialization:gptrans_t" not in checks
    assert checks["initialization:neural_atom_k1"]["state_sha256"] == digest
    assert checks["clean_import:package"]["molgap.k1_screen_training"] == "src/molgap/k1_screen_training.py"
    assert checks["clean_import:package"]["molgap.v4_runtime"] == "src/molgap/v4_runtime.py"


@pytest.mark.parametrize("release", ["broken_output_io", "rewrapped_output_io"], indirect=True)
def test_initialization_pass_cannot_hide_packaged_output_writer_failure(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)
    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert "family_initialization:neural_atom_k1" in report["checks"]
    assert "family_output_io:neural_atom_k1" not in report["checks"]
    assert any(error["check"] == "family_output_io" and error["item"] == "neural_atom_k1"
               for error in report["errors"])


def test_lf_recipe_and_isolated_imports_pass(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)
    assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    assert report["errors"] == []
    assert report["checks"]["clean_import:package"]["molgap.child"] == "src/molgap/child.py"
    assert report["checks"]["initialization:gptrans_t"]["device"] == "cpu"


@pytest.mark.parametrize("release", ["dataset_binding"], indirect=True)
def test_shared_family_release_requires_dataset_manifest_binding(release, tmp_path):
    spec, package, options = release
    launch_config = tmp_path / "launch.json"
    launch_config.write_text("{}", encoding="utf-8")

    missing = dict(options)
    missing.pop("dataset_manifests")
    missing_report = check_release_inputs(
        spec, package, **missing, launch_config=launch_config)
    assert any(error["check"] == "dataset_manifest" and
               "requires a pinned dataset manifest" in error["message"]
               for error in missing_report["errors"])

    wrong_manifest = tmp_path / "wrong-manifest.json"
    wrong_manifest.write_text('{"fixture":"wrong identity"}', encoding="utf-8")
    wrong = dict(options)
    wrong["dataset_manifests"] = {
        arm_id: wrong_manifest for arm_id in options["dataset_manifests"]}
    wrong_report = check_release_inputs(
        spec, package, **wrong, launch_config=launch_config)
    assert any(error["check"] == "dataset_manifest" and
               "differs from frozen recipe" in error["message"]
               for error in wrong_report["errors"])


@pytest.mark.parametrize("release", ["dataset_binding"], indirect=True)
def test_package_only_dataset_probe_checks_flat_and_nested_without_decoding(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)

    assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    expected_sha = hashlib.sha256(
        next(iter(options["dataset_manifests"].values())).read_bytes()).hexdigest()
    for arm_id in options["dataset_manifests"]:
        assert report["checks"]["dataset_mount_resolution:" + arm_id] == {
            "manifest_sha256": expected_sha,
            "declared_path": "slug/manifest.json",
            "layouts": ["flat", "nested"],
            "scope": "metadata_only_no_shard_decode",
        }


@pytest.mark.parametrize("release", ["broken_dataset_resolver"], indirect=True)
def test_release_rejects_broken_packaged_dataset_resolver(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)

    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert any(error["check"] == "dataset_mount_resolution" and
               error["message"] == "packaged broken dataset resolver"
               for error in report["errors"])


def test_multiple_release_defects_are_collected(release, tmp_path):
    spec, package, options = release
    options["recipe_files"]["gptrans_t"] = "missing.json"
    options["initial_states"].pop("neural_atom_k1")
    # Protocol-2 GLOBAL models the exact WedgeData dependency which imports of
    # the trainer alone do not discover. Reading this pickle never executes it.
    shard = tmp_path / "synthetic.pt"
    with zipfile.ZipFile(shard, "w") as archive:
        archive.writestr("fixture/data.pkl", b"\x80\x02cmolgap.pcqm_wedge\nWedgeData\n.")
    report = check_release_inputs(spec, package, **options, pickle_inputs=[shard])
    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert {(e["check"], e["item"]) for e in report["errors"]} >= {
        ("recipe", "gptrans_t"), ("initialization", "neural_atom_k1"),
        ("module", "molgap.pcqm_wedge")}


def test_recipe_pin_uses_packaged_bytes_not_raw_windows_file(release):
    spec, package, options = release
    declaration = spec.to_dict()
    declaration["arms"][0]["training"]["recipe"]["sha256"] = hashlib.sha256(b'{"epochs":10}\r\n').hexdigest()
    # Rebuild a real package with the intentionally wrong recipe pin.
    from molgap.experiment_package import verify_experiment_source_package
    root = package.parent / "repo"
    other = package.parent / "wrong-recipe"
    changed = ExperimentSpec(declaration)
    manifest = build_experiment_source_package(changed, root,
        verify_experiment_source_package(package)["relative_allowlist"], other)
    options["expected_package_identity"] = manifest["package_identity"]
    report = check_release_inputs(changed, other, **options)
    assert any(e["check"] == "recipe" and "LF bytes" in e["message"] for e in report["errors"])


def test_wrong_or_nonfinite_state_is_rejected(release):
    spec, package, options = release
    path = options["initial_states"]["gptrans_t"]
    torch.save({"weight": torch.tensor([float("nan")])}, path)
    report = check_release_inputs(spec, package, **options)
    assert any(e["check"] == "initialization" and "non-finite" in e["message"] for e in report["errors"])
    torch.save({"weight": torch.ones(3)}, path)
    with pytest.raises(ValueError, match="tensor SHA"):
        inspect_frozen_state_artifact(path, expected_state_sha256="a" * 64)


def test_missing_transitive_import_does_not_fall_back_to_host(release, monkeypatch):
    spec, package, options = release
    from molgap import experiment_preflight
    unpack = experiment_preflight._unpack
    def incomplete(source, root, **kwargs):
        unpack(source, root, **kwargs)
        (root / "src/molgap/child.py").unlink()
    monkeypatch.setattr(experiment_preflight, "_unpack", incomplete)
    report = check_release_inputs(spec, package, **options)
    assert any(e["check"] == "import" and "molgap.child" in e["message"] for e in report["errors"])


def test_missing_pickled_class_is_detected_without_unpickling(release, tmp_path):
    spec, package, options = release
    shard = tmp_path / "synthetic.pt"
    with zipfile.ZipFile(shard, "w") as archive:
        archive.writestr("fixture/data.pkl", b"\x80\x02cmolgap.loader\nRemovedDataClass\n.")
    report = check_release_inputs(spec, package, **options, pickle_inputs=[shard])
    assert any(e["check"] == "pickle_symbol" and e["item"] == "molgap.loader.RemovedDataClass"
               for e in report["errors"])


@pytest.mark.parametrize("failure", ["binding", "read_sharing"])
def test_stale_release_report_blocks_push_before_network(release, tmp_path, monkeypatch, failure):
    import json
    from molgap import kaggle_accelerator_push
    spec, package, options = release
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    script = kernel / "script.py"
    original_script = b"print('fixture')\n"
    script.write_bytes(original_script)
    (kernel / "kernel-metadata.json").write_text(json.dumps({"code_file": "script.py"}))
    report = check_release_inputs(spec, package, **options, entry_script=script)
    report_path = tmp_path / "release.json"
    report_path.write_text(json.dumps(report))
    assert kaggle_accelerator_push._verify_release_report(report_path, script)["spec_identity"] == spec.identity
    if failure == "read_sharing":
        import sys
        if sys.platform != "win32":
            pytest.skip("Native Windows file-sharing failure")
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        initial = Path(next(iter(options["initial_states"].values())))
        initial_bytes = initial.read_bytes()
        # A real external IO boundary: the frozen bytes remain unchanged, but
        # this native handle refuses the release owner's next read.
        handle = kernel32.CreateFileW(str(initial), 0x80000000, 0, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        script.write_bytes(b"print('changed after check')\n")
    from unittest.mock import Mock
    post = Mock()
    monkeypatch.setattr(kaggle_accelerator_push.requests, "post", post)
    from molgap.experiment_preflight import _ReleaseContentVerification
    try:
        message = "checks failed during recheck" if failure == "read_sharing" else "changed after verification"
        with pytest.raises(ValueError, match=message) as caught:
            kaggle_accelerator_push.push_kernel_with_accelerator(package_dir=kernel,
                credential_path=tmp_path / "not-needed.json", accelerator="NvidiaTeslaT4",
                release_report_path=report_path,
                _content_verification=_ReleaseContentVerification() if failure == "read_sharing" else None)
    finally:
        if failure == "read_sharing":
            kernel32.CloseHandle(handle)
    post.assert_not_called()
    error = caught.value
    assert isinstance(error, kaggle_accelerator_push.ReleaseRecheckError)
    assert error.expected_report == report
    if failure == "read_sharing":
        assert initial.read_bytes() == initial_bytes
        assert error.recheck_kind == "check_failed"
        assert error.observed_report is None
        assert isinstance(error.__cause__, PermissionError)
        assert error.release_recheck["failures"] == [{"check": "release_input_read",
            "error_type": "PermissionError", "errno": 13}]
        assert error.release_recheck["changed_checks"] == []
        assert error.release_recheck["changed_fields"] == []
    else:
        assert error.recheck_kind == "binding_mismatch"
        assert error.changed_checks == ["entry_script:kernel"]
        assert error.observed_report["errors"] == []
        assert error.release_recheck == {"category": "binding_mismatch",
            "changed_checks": ["entry_script"], "changed_fields": [], "failures": []}

        # Reuse this release fixture to check the bytes sent at the final edge:
        # a write after a successful recheck must not replace its verified body.
        metadata_path = kernel / "kernel-metadata.json"
        metadata = {"id": "owner/kernel", "title": "Fixture Kernel", "code_file": "script.py",
                    "language": "python", "kernel_type": "script", "is_private": True,
                    "enable_gpu": True, "enable_tpu": False, "dataset_sources": ["owner/data"]}
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        metadata_path.write_bytes(metadata_bytes)
        launch_path = tmp_path / "launch.json"
        prospective_hashes = {}
        for arm in spec.to_dict()["arms"]:
            arm_id = arm["arm_id"]
            trajectory = launch_path.parent / "prospective" / arm_id / "trajectory.json"
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            trajectory.write_bytes(b"{}")
            prospective_hashes[arm_id] = hashlib.sha256(b"{}").hexdigest()
        launch = {"spec_identity": spec.identity, "run_reference": metadata["id"],
                  "account": "owner", "dataset_sources": metadata["dataset_sources"],
                  "accelerator": "NvidiaTeslaT4", "device_count": spec.to_dict()["platform"]["device_count"],
                  "prospective_sha256": prospective_hashes}
        launch_bytes = json.dumps(launch, sort_keys=True, separators=(",", ":")).encode("utf-8")
        launch_path.write_bytes(launch_bytes)
        original_script = (f'EXPECTED_LAUNCH_SHA256 = "{hashlib.sha256(launch_bytes).hexdigest()}"\n'
                           'print("fixture")\n').encode("utf-8")
        script.write_bytes(original_script)
        report = check_release_inputs(spec, package, **options, entry_script=script,
            launch_config=launch_path, kernel_metadata=metadata_path)
        assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED", report["errors"]
        report_path.write_text(json.dumps(report), encoding="utf-8")
        credentials = tmp_path / "credentials.json"
        credentials.write_text(json.dumps({"username": "owner", "key": "fixture-key"}), encoding="utf-8")
        response = Mock()
        response.json.return_value = {"error": None, "ref": "owner/kernel", "versionNumber": 1}
        post.reset_mock()
        post.return_value = response
        real_verify = kaggle_accelerator_push._verify_release_report

        def verify_then_mutate(*args, **kwargs):
            result = real_verify(*args, **kwargs)
            script.write_bytes(b"changed after the successful recheck\n")
            metadata_path.write_bytes(json.dumps({**metadata, "title": "Changed after recheck"}).encode("utf-8"))
            return result

        monkeypatch.setattr(kaggle_accelerator_push, "_verify_release_report", verify_then_mutate)
        submitted = kaggle_accelerator_push.push_kernel_with_accelerator(
            package_dir=kernel, credential_path=credentials, accelerator="NvidiaTeslaT4",
            release_report_path=report_path)
        assert submitted["status"] == "submitted"
        request = post.call_args.kwargs["json"]
        assert request["text"].encode("utf-8") == original_script
        assert request["slug"] == metadata["id"] and request["newTitle"] == metadata["title"]
        assert script.read_bytes() != original_script

        # A captured metadata snapshot that disagrees with the report must be
        # rejected even if the file is restored before the real verifier reads it.
        script.write_bytes(original_script)
        metadata_path.write_bytes(metadata_bytes)
        captured_bad_metadata = json.dumps({**metadata, "title": "Unverified snapshot"},
                                           sort_keys=True).encode("utf-8")
        metadata_path.write_bytes(captured_bad_metadata)
        post.reset_mock()

        def restore_then_verify(*args, **kwargs):
            metadata_path.write_bytes(metadata_bytes)
            return real_verify(*args, **kwargs)

        monkeypatch.setattr(kaggle_accelerator_push, "_verify_release_report", restore_then_verify)
        with pytest.raises(kaggle_accelerator_push.ReleaseRecheckError) as captured_error:
            kaggle_accelerator_push.push_kernel_with_accelerator(
                package_dir=kernel, credential_path=credentials, accelerator="NvidiaTeslaT4",
                release_report_path=report_path)
        assert captured_error.value.release_recheck["category"] == "binding_mismatch"
        assert "workflow" in captured_error.value.release_recheck["changed_checks"]
        post.assert_not_called()


def test_staged_source_payload_is_bound_to_the_verified_archive(release, tmp_path):
    import shutil
    spec, package, options = release
    inputs = tmp_path / "input"
    inputs.mkdir()
    for name in ("SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json", "experiment_spec.json"):
        shutil.copyfile(package / name, inputs / name)
    shutil.copyfile(package / "source.tar.gz", inputs / "source_payload.bin")
    for arm, path in list(options["initial_states"].items()):
        shutil.copyfile(path, inputs / path.name)
        options["initial_states"][arm] = inputs / path.name
    assert check_release_inputs(spec, package, **options, input_root=inputs)["errors"] == []
    (inputs / "source_payload.bin").write_bytes(b"wrong uploaded archive")
    report = check_release_inputs(spec, package, **options, input_root=inputs)
    assert any(e["check"] == "input_layout" for e in report["errors"])


@pytest.mark.parametrize("release", ["dataset_binding"], indirect=True)
def test_transaction_content_reuse_still_checks_remote_layout_and_initialization_bytes(release, tmp_path, monkeypatch):
    import shutil
    from molgap import experiment_preflight as owner
    from molgap.experiment_package import stage_source_inputs
    spec, package, options = release
    memo = owner._ReleaseContentVerification()
    unpacks = []
    original = owner._unpack
    def unpack(*args, **kwargs):
        unpacks.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, "_unpack", unpack)
    first = owner.check_release_inputs(spec, package, **options, _content_verification=memo)
    assert not first["errors"] and len(unpacks) == 1
    remote = tmp_path / "remote"
    stage_source_inputs(package, remote, options["initial_states"])
    target_manifest = remote / "slug" / "manifest.json"
    target_manifest.parent.mkdir(parents=True)
    shutil.copyfile(next(iter(options["dataset_manifests"].values())), target_manifest)
    changed = dict(options,
        initial_states={arm: remote / "initial_states" / (arm + ".pt")
                        for arm in options["initial_states"]},
        dataset_manifests={arm: target_manifest for arm in options["dataset_manifests"]},
        input_root=remote)
    original_run = owner.subprocess.run
    worker_requests = []
    def track_worker(command, *args, **kwargs):
        worker_requests.append(json.loads(Path(command[-2]).read_bytes()))
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(owner.subprocess, "run", track_worker)
    second = owner.check_release_inputs(spec, package, **changed, _content_verification=memo)
    assert not second["errors"] and len(unpacks) == 2
    assert second["checks"]["input_layout:mount"] is True
    assert second["inputs"]["input_root"] == str(remote)
    assert worker_requests[-1]["modules"] == []
    assert worker_requests[-1]["initializations"] == []
    assert worker_requests[-1]["dataset_manifests"][0]["path"] == str(target_manifest)
    # Local mtime/size never vouch for bytes at a transport boundary.
    sidecar = remote / "SOURCE_COMMIT.txt"
    sidecar.write_bytes(b"0" * 40 + b"\n")
    failed = owner.check_release_inputs(spec, package, **changed, _content_verification=memo)
    assert failed["errors"][0]["check"] == "input_layout" and len(unpacks) == 3
    assert worker_requests[-1]["dataset_manifests"][0]["path"] == str(target_manifest)
    assert worker_requests[-1]["initializations"] == []
    shutil.copyfile(package / "SOURCE_COMMIT.txt", sidecar)
    state = next(iter(changed["initial_states"].values()))
    state.write_bytes(b"different serialized initialization")
    failed = owner.check_release_inputs(spec, package, **changed, _content_verification=memo)
    assert any(row["check"] == "initialization" for row in failed["errors"])
    assert len(unpacks) == 4
    # An independent transaction must execute its own full checks.
    owner.check_release_inputs(spec, package, **options, _content_verification=owner._ReleaseContentVerification())
    assert len(unpacks) == 5


def test_transaction_content_reuse_does_not_allow_initialization_outside_mount(release, tmp_path):
    from molgap import experiment_preflight as owner
    from molgap.experiment_package import stage_source_inputs
    spec, package, options = release
    memo = owner._ReleaseContentVerification()
    assert not owner.check_release_inputs(spec, package, **options, _content_verification=memo)["errors"]
    remote = tmp_path / "remote"
    stage_source_inputs(package, remote, options["initial_states"])
    report = owner.check_release_inputs(spec, package, **options, input_root=remote, _content_verification=memo)
    assert any(row["check"] == "initialization" and "outside" in row["message"] for row in report["errors"])
