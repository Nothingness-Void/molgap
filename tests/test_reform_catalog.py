"""Executable capabilities follow the one active catalog, not old snapshots."""
from dataclasses import replace
import pytest

from molgap import experiment_execution as execution
from molgap.experiment_catalog import ADDONS,FAMILIES


def test_shared_recipe_owners_follow_catalog_metadata_and_reject_ambiguous_versions(monkeypatch):
    adapter = execution.TRAINING_ADAPTERS[("neural_atom_k1", "3")]
    changed = replace(adapter, family=("neural_atom_k1", "4"), artifact_adapter="fixture-session")
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", {changed.family: changed})
    projected = execution.shared_training_adapters()
    assert projected["neural_atom_k1"] is changed
    with pytest.raises(TypeError):
        projected["other"] = changed
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS",
                        {changed.family: changed, adapter.family: adapter})
    with pytest.raises(ValueError, match="Ambiguous"):
        execution.shared_training_adapters()


def test_shared_recipe_projection_excludes_factory_and_historical_owners():
    projected = execution.shared_training_adapters()
    assert {name: adapter.family for name, adapter in projected.items()} == {
        "neural_atom_k1": ("neural_atom_k1", "3"), "gptrans_t": ("gptrans_t", "2")}

def test_execution_projection_tracks_the_current_catalog_not_a_historical_capability_lock():
    assert set(execution.TRAINING_ADAPTERS)=={key for key,c in FAMILIES.items() if c.execution is not None}
    for key,adapter in execution.TRAINING_ADAPTERS.items():
        entry=FAMILIES[key].execution
        assert adapter.module==entry.module and adapter.artifact_adapter==entry.artifact_adapter
        expected={(name,version,mode) for (name,version),c in ADDONS.items()
                  for family_version,mode in c.execution_modes
                  if c.family==key[0] and family_version==key[1]}
        assert {(a.name,a.version,a.mode) for a in adapter.addons}==expected


def test_ordinary_addon_execution_is_derived_from_its_one_catalog_entry(monkeypatch):
    original=ADDONS[("k1_readout_scale","1")]
    contract=replace(original,exclusive_group="independent-fixture")
    monkeypatch.setattr(execution,"ADDONS",{**ADDONS,("new_fixture_scale","1"):contract})
    derived=execution._catalog_training_adapters()[("neural_atom_k1","3")]
    addon=derived.addon("new_fixture_scale")
    assert addon.apply_hook==contract.apply_hook and addon.mode=="composed"
    assert len(derived.addons)==len(execution.TRAINING_ADAPTERS[("neural_atom_k1","3")].addons)+1


def test_declaration_only_and_cpu_families_are_not_silently_promoted():
    execution.validate_training_registry()
    assert ("neural_atom_k1","1") not in execution.TRAINING_ADAPTERS
    assert ("gptrans_scale_ema","1") not in execution.TRAINING_ADAPTERS
    for key in (("ppgt_small","1"),("esa_small","1"),("esa_small","2")):
        assert execution.TRAINING_ADAPTERS[key].cpu_synthetic_only


def test_session_interfaces_keep_cpu_defaults_separate_from_mature_and_native_families():
    assert execution.SESSION_INTERFACES["ppgt_small"] == ("ppgt_small", "1")
    assert execution.SESSION_INTERFACES["esa_small"] == ("esa_small", "1")
    assert execution.SESSION_INTERFACES["neural_atom_k1"] == ("neural_atom_k1", "3")
    assert execution.SESSION_INTERFACES["gptrans_t"] == ("gptrans_t", "2")

    for key in (("ppgt_small", "1"), ("esa_small", "1"), ("esa_small", "2")):
        assert execution.TRAINING_ADAPTERS[key].cpu_synthetic_only
    for key in (("ppgt_small", "2"), ("ppgt_small", "3"),
                ("esa_small", "3"), ("esa_small", "4")):
        adapter = execution.TRAINING_ADAPTERS[key]
        assert not adapter.cpu_synthetic_only
        assert adapter.integration_preview or adapter.formal_training
