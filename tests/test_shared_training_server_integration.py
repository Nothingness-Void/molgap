"""Server compatibility checks; no molecular data or native accelerator use."""
from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest


def test_shared_registry_preserves_historical_interfaces():
    from molgap.experiment_execution import TRAINING_ADAPTERS, validate_training_registry
    validate_training_registry()
    assert TRAINING_ADAPTERS[("neural_atom_k1", "2")].module == "molgap.k1_screen_training"
    assert TRAINING_ADAPTERS[("gptrans_t", "1")].module == "molgap.gptrans_screen_workflow"
    for key in (("neural_atom_k1", "3"), ("gptrans_t", "2")):
        entry = TRAINING_ADAPTERS[key]
        assert entry.module == "molgap.family_training_runner"
        assert entry.target_identity_in_completion
        assert all(hasattr(addon, "mode") for addon in entry.addons)


def test_shared_cli_has_no_platform_submitter():
    from molgap.experiment_cli import _parser
    args = _parser().parse_args(["prepare-family-training", "--spec", "spec.json",
        "--repo-root", ".", "--plan", "plan.json", "--output", "fresh"])
    assert args.command == "prepare-family-training"


@pytest.mark.parametrize("malformed", [False, True])
def test_release_checks_actual_shared_initialization_schema(tmp_path, monkeypatch, malformed):
    from molgap import experiment_preflight as owner
    import sys
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(owner, "_origins", lambda root: {})
    observed = []
    def load(recipe, path):
        observed.append((recipe, path))
        if malformed:
            raise ValueError("Frozen initialization tensor keys/shape/dtype differ")
        return object(), {"state_sha256": "a" * 64, "tensor_count": 1, "device": "cpu"}
    modules = {
        "molgap.family_training_runner": SimpleNamespace(read_recipe=lambda path: ({}, "recipe")),
        "molgap.family_training_kernel": SimpleNamespace(load_family_model=load),
    }
    real_import = importlib.import_module
    monkeypatch.setattr(owner.importlib, "import_module",
        lambda name: modules[name] if name in modules else real_import(name))
    result = owner._release_imports({"source_root": str(tmp_path),
        "dependency_paths": [], "modules": [], "pickle_globals": [],
        "initializations": [{"arm_id": "candidate", "path": str(tmp_path / "init.pt"),
                             "recipe_path": str(tmp_path / "recipe.json")}]})
    assert len(observed) == 1
    if malformed:
        assert result["initialization_checks"] == {}
        assert result["errors"][0]["check"] == "family_initialization"
    else:
        assert not result["errors"]
        assert result["initialization_checks"]["candidate"]["tensor_count"] == 1


def test_scalar_metric_reduction_and_atomic_array(tmp_path):
    import torch
    from molgap.training_reproducibility import atomic_json, sum_scalar_metrics
    assert sum_scalar_metrics([torch.tensor(.5), torch.tensor(.25)]) == .75
    atomic_json(tmp_path / "times.json", [1., 2.])
    assert (tmp_path / "times.json").read_text().strip().startswith("[")
