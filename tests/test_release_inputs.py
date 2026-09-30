"""Synthetic release regressions; no real graph roles, models or network."""
import hashlib
import zipfile
from pathlib import Path

import pytest
import torch

from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_spec import ExperimentSpec
from molgap.v4_runtime import state_dict_sha256, inspect_frozen_state_artifact
from test_experiment_package import git
from test_experiment_spec import payload


@pytest.fixture
def release(tmp_path, payload):
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
    git(root, "add", ".")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
    states = {}
    for arm in payload["arms"]:
        state = {"weight": torch.arange(3, dtype=torch.float32)}
        arm["initialization"]["state_sha256"] = state_dict_sha256(state)
        arm["training"]["recipe"]["sha256"] = hashlib.sha256(b'{"epochs":10}\n').hexdigest()
        path = tmp_path / (arm["arm_id"] + ".pt")
        torch.save(state, path)
        states[arm["arm_id"]] = path
    spec = ExperimentSpec(payload)
    package = tmp_path / "package"
    manifest = build_experiment_source_package(spec, root,
        ["src/molgap/__init__.py", "src/molgap/loader.py", "src/molgap/child.py", "recipe.json"], package)
    return spec, package, {"expected_package_identity": manifest["package_identity"],
        "recipe_files": {a["arm_id"]: "recipe.json" for a in payload["arms"]},
        "initial_states": states, "required_modules": ["molgap.loader"]}


def test_lf_recipe_and_isolated_imports_pass(release):
    spec, package, options = release
    report = check_release_inputs(spec, package, **options)
    assert report["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    assert report["errors"] == []
    assert report["checks"]["clean_import:package"]["molgap.child"] == "src/molgap/child.py"
    assert report["checks"]["initialization:gptrans_t"]["device"] == "cpu"


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
    def incomplete(source, root):
        unpack(source, root)
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


def test_stale_release_report_blocks_push_before_network(release, tmp_path, monkeypatch):
    import json
    from molgap import kaggle_accelerator_push
    spec, package, options = release
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    script = kernel / "script.py"
    script.write_bytes(b"print('fixture')\n")
    (kernel / "kernel-metadata.json").write_text(json.dumps({"code_file": "script.py"}))
    report = check_release_inputs(spec, package, **options, entry_script=script)
    report_path = tmp_path / "release.json"
    report_path.write_text(json.dumps(report))
    assert kaggle_accelerator_push._verify_release_report(report_path, script)["spec_identity"] == spec.identity
    script.write_bytes(b"print('changed after check')\n")
    from unittest.mock import Mock
    post = Mock()
    monkeypatch.setattr(kaggle_accelerator_push.requests, "post", post)
    with pytest.raises(ValueError, match="changed after verification"):
        kaggle_accelerator_push.push_kernel_with_accelerator(package_dir=kernel,
            credential_path=tmp_path / "not-needed.json", accelerator="NvidiaTeslaT4",
            release_report_path=report_path)
    post.assert_not_called()


def test_staged_source_payload_is_bound_to_the_verified_archive(release, tmp_path):
    import shutil
    spec, package, options = release
    inputs = tmp_path / "input"
    inputs.mkdir()
    from molgap.experiment_package import SIDECARS
    for name in sorted(SIDECARS - {"source.tar.gz"}):
        shutil.copyfile(package / name, inputs / name)
    shutil.copyfile(package / "source.tar.gz", inputs / "source_payload.bin")
    for arm, path in list(options["initial_states"].items()):
        shutil.copyfile(path, inputs / path.name)
        options["initial_states"][arm] = inputs / path.name
    assert check_release_inputs(spec, package, **options, input_root=inputs)["errors"] == []
    (inputs / "source_payload.bin").write_bytes(b"wrong uploaded archive")
    report = check_release_inputs(spec, package, **options, input_root=inputs)
    assert any(e["check"] == "input_layout" for e in report["errors"])


def test_initialization_outside_staged_inputs_is_rejected(release, tmp_path):
    spec, package, options = release
    inputs = tmp_path / "empty-inputs"
    inputs.mkdir()
    report = check_release_inputs(spec, package, **options, input_root=inputs)
    assert any(e["check"] == "initialization" and "outside" in e["message"]
               for e in report["errors"])


def test_stack_global_is_explicitly_unsupported(release, tmp_path):
    spec, package, options = release
    shard = tmp_path / "protocol4.pt"
    with zipfile.ZipFile(shard, "w") as archive:
        archive.writestr("fixture/data.pkl", b"\x80\x04\x8c\x0dmolgap.loader\x8c\x05VALUE\x93.")
    report = check_release_inputs(spec, package, **options, pickle_inputs=[shard])
    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert any(e["check"] == "pickle_dependencies" and "STACK_GLOBAL" in e["message"]
               for e in report["errors"])


def test_failed_release_report_blocks_push_before_credentials_and_network(release, tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock
    from molgap import kaggle_accelerator_push

    spec, package, options = release
    options["initial_states"].clear()
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    script = kernel / "script.py"
    script.write_bytes(b"print('fixture')\n")
    (kernel / "kernel-metadata.json").write_text(json.dumps({"code_file": "script.py"}))
    report = check_release_inputs(spec, package, **options, entry_script=script)
    report_path = tmp_path / "release.json"
    report_path.write_text(json.dumps(report))
    post = Mock()
    monkeypatch.setattr(kaggle_accelerator_push.requests, "post", post)
    with pytest.raises(ValueError, match="did not pass"):
        kaggle_accelerator_push.push_kernel_with_accelerator(package_dir=kernel,
            credential_path=tmp_path / "not-needed.json", accelerator="NvidiaTeslaT4",
            release_report_path=report_path)
    post.assert_not_called()
