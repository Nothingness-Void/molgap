"""Registered CPU profiles never become a platform launch or real-data loader."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import hashlib
import subprocess
import sys
import tarfile

import pytest
import torch

from molgap.experiment_execution import TRAINING_ADAPTERS, validate_training_registry, validate_execution_plan, workflow_capabilities
from molgap.experiment_spec import ExperimentSpec
from molgap.shared_model_adapter import build_model, normalized_hook_source_sha256
from molgap.source_dependencies import discover_source_dependencies
from molgap.v4_bundle import build_v4_source_bundle
from modern2d_fixtures import model_config
from test_experiment_spec import arm, payload


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_static_registry_model_abi_and_remote_denial(family, payload):
    validate_training_registry()
    declaration = arm(family)
    adapter = TRAINING_ADAPTERS[(family, "1")]
    declaration["base"]["sha256"] = normalized_hook_source_sha256(adapter.model_factory, function="make_model")
    declaration["model_config"] = model_config(family)
    model = build_model(adapter, declaration)
    assert isinstance(model, torch.nn.Module) and adapter.cpu_synthetic_only
    # Spec is in-memory synthetic metadata only, never a published trajectory.
    declaration.pop("model_config")
    payload["arms"] = [declaration]
    spec = ExperimentSpec(payload)
    assert not workflow_capabilities(spec)["registered_training_arms"]
    from molgap.fast_sandbox import run_arm_sandbox
    with pytest.raises(ValueError, match="bounded configuration"):
        run_arm_sandbox(spec, declaration["arm_id"])
    with pytest.raises(ValueError, match="CPU synthetic"):
        validate_execution_plan(spec, [])


@pytest.mark.parametrize("family", ["ppgt_small", "esa_small"])
def test_registered_native_bootstrap_denies_before_any_real_io(monkeypatch, tmp_path, family):
    from molgap import family_training_runner as runner
    from molgap import family_training_data as data_owner
    from test_modern2d_session import make_session
    session, recipe, _, _ = make_session(tmp_path, family, [])
    monkeypatch.setattr(runner, "read_recipe", lambda path: ({}, recipe))
    def forbidden(*a, **k): raise AssertionError("Real data/native path was reached")
    monkeypatch.setattr(data_owner, "load_training_data", forbidden)
    # _bootstrap's profile rejection must precede any source, Spec or data read.
    with pytest.raises(ValueError, match="CPU profile"):
        runner._bootstrap(recipe_path=tmp_path / "absent.json", spec=None,
                          expected_package_identity=None, mode="reference", account=None,
                          run_reference=None, arm_id=None, package_dir=None, input_root=None,
                          initial_state_path=None, trajectory_id=None)
    with pytest.raises(FrozenInstanceError):
        recipe.model_config.hidden = 128
    exported = recipe.to_dict()
    exported["model_config"]["hidden"] = 128
    assert recipe.model_config.hidden == 16
    session.close()


@pytest.mark.parametrize("module", ["ppgt", "esa"])
def test_model_source_bundle_isolated_import_and_byte_identity(tmp_path, module, record_property):
    root = Path(__file__).parents[1]
    closure = discover_source_dependencies(root, [f"src/molgap/{module}_model.py"], require_static=True)
    assert not any("vendor" in p or "training_runner" in p for p in closure.source_files)
    assert {"xformers", "flash_attn", "admin_torch", "pytorch_lightning", "wandb"}.isdisjoint(closure.external_modules)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    bundle = build_v4_source_bundle(repo_root=root, relative_paths=closure.source_files,
                                   output_dir=tmp_path / "bundle", source_commit=commit)
    record_property("source_archive_sha256", bundle["archive_sha256"])
    record_property("source_commit", commit)
    record_property("source_files", ",".join(closure.source_files))
    archive_path = tmp_path / "bundle/source.tar.gz"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == bundle["archive_sha256"]
    extracted = tmp_path / "isolated"
    with tarfile.open(archive_path) as archive:
        assert archive.getnames() == list(closure.source_files)
        for member in archive.getmembers():
            assert member.isfile() and member.name in closure.source_files
            data = archive.extractfile(member).read()
            assert data == (root / member.name).read_bytes().replace(b"\r\n", b"\n")
            target = extracted / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    script = ("import sys; sys.path.insert(0,sys.argv[1]); "
              f"from molgap.{module}_model import make_model; "
              f"model=make_model({{'model_config':{model_config(module + '_small')!r}}}); "
              "assert all(t.device.type=='cpu' for t in model.parameters()); "
              "assert 'wandb' not in sys.modules; print('ISOLATED_CPU_OK')")
    result = subprocess.run([sys.executable, "-I", "-c", script, str(extracted / "src")],
        cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ISOLATED_CPU_OK"
