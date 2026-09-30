"""Local synthetic staging; no model, cache, credentials or platform calls."""
import hashlib
import json
import subprocess

import pytest

from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_staging import UploadArtifact, stage_release_inputs
from test_experiment_spec import payload


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def staging(tmp_path, payload):
    repo = tmp_path / "repo"
    module = repo / "src/molgap"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("")
    (module / "fixture.py").write_text("VALUE = 42\n")
    (repo / "recipe.json").write_bytes(b'{"synthetic":true}\n')
    entry = repo / "entry.py"
    entry.write_bytes(b'PIN = "__PIN_SOURCE_ARCHIVE_SHA256__"\n')
    metadata = repo / "kernel.json"
    metadata.write_bytes(b'{"synthetic":true}')
    for args in (("init",), ("config", "user.email", "fixture@example.invalid"),
                 ("config", "user.name", "Fixture"), ("config", "core.autocrlf", "false"),
                 ("add", "."), ("-c", "commit.gpgsign=false", "commit", "-m", "Fixture")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    payload["arms"] = payload["arms"][:1]
    import torch
    from molgap.v4_runtime import state_dict_sha256
    state = {"synthetic_weight": torch.tensor([1.0, 2.0])}
    initial = tmp_path / "initial.pt"
    torch.save(state, initial)
    payload["arms"][0]["initialization"] = {
        "kind": "frozen_state", "seed": 42, "state_sha256": state_dict_sha256(state)}
    payload["arms"][0]["training"]["recipe"]["sha256"] = digest(repo / "recipe.json")
    artifact = tmp_path / "trusted.bin"
    artifact.write_bytes(b"synthetic optional input")
    return dict(
        spec=ExperimentSpec(payload), repo_root=repo,
        relative_paths=["src/molgap/__init__.py", "src/molgap/fixture.py", "recipe.json", "entry.py", "kernel.json"],
        output=tmp_path / "staged", artifacts={"optional.bin": UploadArtifact(artifact, digest(artifact)),
                                               "initial.pt": UploadArtifact(initial, digest(initial))},
        recipe_files={"gptrans_t": "recipe.json"}, initial_states={"gptrans_t": "initial.pt"},
        required_modules=["molgap.fixture"], entry_template=entry,
        kernel_metadata=metadata, dataset_metadata={"id": "synthetic/private-source"},
    )


def test_real_package_release_and_mount_layout(staging):
    result = stage_release_inputs(**staging)
    root = staging["output"]
    report = json.loads((root / "release.json").read_text())
    assert result["release_status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    assert result["errors"] == []
    assert report["checks"]["input_layout:mount"] is True
    assert report["checks"]["recipe:gptrans_t"] == digest(staging["repo_root"] / "recipe.json")
    assert result["compute_released"] is result["submitted"] is False
    assert digest(root / "inputs/source_payload.bin") == result["package"]["archive_sha256"]
    from molgap.gptrans_author_screen import restore_source_package
    restored = restore_source_package(root / "inputs", root / "restored", result["package"]["archive_sha256"])
    assert restored["package_identity"] == result["package"]["package_identity"]
    assert digest(root / "inputs/optional.bin") == staging["artifacts"]["optional.bin"].sha256
    assert result["package"]["archive_sha256"] in (root / "kernel/run.py").read_text()
    assert "__PIN_SOURCE_ARCHIVE_SHA256__" not in (root / "kernel/run.py").read_text()
    assert digest(root / "kernel/kernel-metadata.json") == digest(staging["kernel_metadata"])
    assert json.loads((root / "staging.json").read_text()) == result
    with pytest.raises(FileExistsError):
        stage_release_inputs(**staging)


@pytest.mark.parametrize("name", ["../bad.bin", "sub/input.bin", "SOURCE_FILES.json", "source_payload.bin",
                                  "credentials.json", "api-key.json", "C:input.bin", "input.", "CON.pt"])
def test_invalid_artifact_names_fail_before_output(staging, name):
    staging["artifacts"] = {name: next(iter(staging["artifacts"].values()))}
    with pytest.raises(ValueError):
        stage_release_inputs(**staging)
    assert not staging["output"].exists()


def test_case_collision_and_bad_digest(staging):
    artifact = staging["artifacts"]["optional.bin"]
    staging["artifacts"]["OPTIONAL.bin"] = artifact
    with pytest.raises(ValueError, match="Case-colliding"):
        stage_release_inputs(**staging)
    del staging["artifacts"]["OPTIONAL.bin"]
    staging["artifacts"]["optional.bin"] = UploadArtifact(artifact.path, "0" * 64)
    with pytest.raises(ValueError, match="SHA/boundary"):
        stage_release_inputs(**staging)
    assert not staging["output"].exists()


@pytest.mark.parametrize("pin_count", [0, 2])
def test_pin_is_not_an_unbound_manual_replacement(staging, pin_count):
    staging["entry_template"].write_text("__PIN_SOURCE_ARCHIVE_SHA256__" * pin_count)
    with pytest.raises(ValueError, match="exactly one"):
        stage_release_inputs(**staging)


def test_initial_state_cannot_point_outside_uploaded_inputs(staging):
    staging["initial_states"] = {"gptrans_t": "absent.pt"}
    with pytest.raises(ValueError, match="staged artifact"):
        stage_release_inputs(**staging)


def test_unfrozen_entry_template_rejected(staging):
    staging["relative_paths"].remove("entry.py")
    with pytest.raises(ValueError, match="explicit packaged source"):
        stage_release_inputs(**staging)


def test_render_uses_frozen_archive_not_later_worktree_edits(staging, monkeypatch):
    import molgap.experiment_staging as module
    original = module.build_experiment_source_package
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        staging["entry_template"].write_bytes(b"UNPINNED = True\n")
        staging["kernel_metadata"].write_bytes(b'{"changed_after_freeze":true}')
        return result
    monkeypatch.setattr(module, "build_experiment_source_package", build)
    result = stage_release_inputs(**staging)
    assert result["errors"] == []
    assert (staging["output"] / "kernel/run.py").read_bytes().startswith(b"PIN = ")
    assert (staging["output"] / "kernel/kernel-metadata.json").read_bytes() == b'{"synthetic":true}'


def test_failed_checks_are_retained_not_promoted(staging):
    staging["required_modules"] = ["molgap.absent"]
    result = stage_release_inputs(**staging)
    assert result["release_status"] == "RELEASE_INPUTS_FAILED"
    assert result["errors"]
    assert result["compute_released"] is False
    assert (staging["output"] / "release.json").is_file()


def test_package_failure_keeps_diagnostic_and_never_overwrites(staging):
    staging["relative_paths"].append("absent.py")
    with pytest.raises(ValueError):
        stage_release_inputs(**staging)
    diagnostic = json.loads((staging["output"] / "staging_failure.json").read_text())
    assert diagnostic["status"] == "STAGING_FAILED"
    assert diagnostic["compute_released"] is diagnostic["submitted"] is False
    assert not (staging["output"] / "release.json").exists()


def test_initialization_mapping_delegated_using_uploaded_path(staging, monkeypatch):
    import molgap.experiment_staging as module
    observed = {}
    def check(spec, package, **kwargs):
        observed.update(kwargs)
        return {"status": "SYNTHETIC_CHECK", "errors": []}
    monkeypatch.setattr(module, "check_release_inputs", check)
    staging["initial_states"] = {"gptrans_t": "optional.bin"}
    stage_release_inputs(**staging)
    assert observed["initial_states"] == {"gptrans_t": staging["output"] / "inputs/optional.bin"}
    assert observed["entry_script"] == staging["output"] / "kernel/run.py"
    assert observed["input_root"] == staging["output"] / "inputs"


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_release_gate_checks_uploaded_package_manifest(staging, fault):
    from molgap.experiment_preflight import check_release_inputs
    result = stage_release_inputs(**staging)
    root = staging["output"]
    manifest = root / "inputs/package_manifest.json"
    if fault == "missing":
        manifest.unlink()
    else:
        manifest.write_bytes(b'{"fabricated":true}')
    report = check_release_inputs(staging["spec"], root / "source",
        expected_package_identity=result["package"]["package_identity"],
        recipe_files=staging["recipe_files"],
        initial_states={"gptrans_t": root / "inputs/initial.pt"},
        required_modules=staging["required_modules"], entry_script=root / "kernel/run.py",
        input_root=root / "inputs")
    assert report["status"] == "RELEASE_INPUTS_FAILED"
    assert any(error["check"] == "input_layout" for error in report["errors"])
