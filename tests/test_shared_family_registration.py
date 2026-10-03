"""Static extension checks for a genuinely new family, without legacy dispatch."""
from dataclasses import replace
import subprocess
import sys
from types import MappingProxyType

import pytest

from molgap import experiment_execution as execution
from molgap import experiment_spec as declaration
from molgap.experiment_family_artifacts import artifact_adapter
from molgap.experiment_source_inventory import registered_source_files
from test_experiment_workflow import _arm, _spec_payload


@pytest.fixture
def new_family(monkeypatch):
    key = ("synthetic_graph_family", "1")
    contract = declaration.FamilyContract(*key, "molgap.synthetic_graph_model",
        "graph_gap_screen_v1", "ogb-atom9-bond3-rwse16-v1", ("train", "development"),
        "seed42-epoch-global-randperm-v1", "train-mean-unbiased-std")
    families = MappingProxyType({**declaration.FAMILIES, key: contract})
    monkeypatch.setattr(declaration, "FAMILIES", families)
    monkeypatch.setattr(execution, "FAMILIES", families)
    adapter = execution.graph_training_adapter(key,
        model_factory="molgap.synthetic_graph_model:make_model")
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
        MappingProxyType({**execution.TRAINING_ADAPTERS, key: adapter}))
    arm = _arm("neural_atom_k1", "2", "new_family")
    arm["family"] = dict(name=key[0], version=key[1])
    for field, name in (("recipe", contract.recipe), ("sampler", contract.sampler),
                        ("transform", contract.transform)):
        arm["training"][field]["name"] = name
    return key, adapter, arm


def test_new_family_uses_same_artifact_and_lifecycle_owners(new_family):
    key, adapter, arm = new_family
    execution.validate_training_registry()
    spec = declaration.ExperimentSpec(_spec_payload([arm], platform="kaggle"))
    assert artifact_adapter("graph-screen-v1", key).ema_required is False
    capabilities = execution.workflow_capabilities(spec)
    assert capabilities["preparation_entry"] == "prepare-workflow"
    assert capabilities["registered_training_arms"]["new_family"]["trainer"] == "molgap.graph_screen_training"
    sources = registered_source_files(spec)
    assert "src/molgap/synthetic_graph_model.py" in sources
    assert "src/molgap/shared_model_adapter.py" in sources
    assert "src/molgap/k1_screen_training.py" not in sources
    assert "src/molgap/gptrans_screen_workflow.py" not in sources


def test_generic_profile_does_not_accept_undeclared_or_wrong_family():
    with pytest.raises(ValueError, match="reviewed graph training registration"):
        artifact_adapter("graph-screen-v1", ("unknown", "1"))
    with pytest.raises(ValueError, match="reviewed graph training registration"):
        artifact_adapter("graph-screen-v1", ("neural_atom_k1", "2"))


def test_shared_metadata_import_does_not_initialize_ml_dependencies():
    script = ("import sys; from molgap.shared_model_adapter import graph_metadata; "
              "from molgap import experiment_execution; "
              "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_hook_cannot_be_selected_from_unreviewed_external_import():
    for hook in ("os:system", "molgap.foo:factory.run", "molgap.foo;evil:make_model"):
        with pytest.raises(ValueError, match="reviewed molgap"):
            execution.graph_training_adapter(("unknown", "1"), model_factory=hook)


def test_new_family_still_requires_compatible_training_contract(new_family, monkeypatch):
    key, adapter, arm = new_family
    bad = replace(execution.FAMILIES[key], feature_schema="unqualified-geometry")
    monkeypatch.setattr(execution, "FAMILIES", {**execution.FAMILIES, key: bad})
    with pytest.raises(ValueError, match="explicit feature/role/recipe"):
        execution.validate_training_registry()


def test_addon_hook_is_bound_to_its_declared_source(new_family, monkeypatch):
    key, adapter, arm = new_family
    contract = declaration.AddonContract(key[0], "message-delta", "molgap.synthetic_graph_addon")
    addons = {**declaration.ADDONS, ("synthetic_delta", "1"): contract}
    monkeypatch.setattr(declaration, "ADDONS", addons)
    monkeypatch.setattr(execution, "ADDONS", addons)
    addon = execution.TrainingAddon("synthetic_delta", "1", "delta",
        apply_hook="molgap.wrong_source:apply_addon")
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
        {**execution.TRAINING_ADAPTERS, key: replace(adapter, addons=(addon,))})
    with pytest.raises(ValueError, match="model-delta hook"):
        execution.validate_training_registry()


def test_shared_models_compose_registered_addons_without_new_execution_owner(new_family, monkeypatch):
    key, adapter, arm = new_family
    names = ("synthetic_delta", "synthetic_readout")
    contracts = {name: declaration.AddonContract(key[0], "model-delta", f"molgap.{name}")
                 for name in names}
    addons = {**declaration.ADDONS, **{(name, "1"): contract for name, contract in contracts.items()}}
    monkeypatch.setattr(declaration, "ADDONS", addons)
    monkeypatch.setattr(execution, "ADDONS", addons)
    hooks = tuple(execution.TrainingAddon(name, "1", name,
        apply_hook=f"molgap.{name}:apply_addon") for name in names)
    updated = replace(adapter, addons=hooks)
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
        {**execution.TRAINING_ADAPTERS, key: updated})
    arm["addons"] = [{"name": name, "version": "1", "config": {}, "source_sha256": "a" * 64}
                     for name in names]
    arm["addon_semantics"] = "ordered"
    execution.validate_training_registry()
    assert updated.mode(arm) == "composed"
    arm["addons"][1]["name"] = "unregistered_hook"
    with pytest.raises(ValueError, match="no executable family adapter"):
        updated.mode(arm)
