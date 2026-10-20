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


def test_legacy_declaration_only_families_are_not_implicitly_promoted():
    execution.validate_training_registry()
    assert {
        ("neural_atom_k1", "2"), ("neural_atom_k1", "3"),
        ("gptrans_t", "1"), ("gptrans_t", "2"),
    } <= set(execution.TRAINING_ADAPTERS)
    assert ("neural_atom_k1", "1") not in execution.TRAINING_ADAPTERS
    assert ("edge_state_gps", "1") not in execution.TRAINING_ADAPTERS


def test_desktop_replication_seed_and_addon_policy_survive_catalog_projection():
    assert FAMILIES[("neural_atom_k1", "3")].allow_any_seed
    assert FAMILIES[("gptrans_t", "2")].allow_any_seed
    assert FAMILIES[("neural_atom_k1", "2")].replication_samplers == (
        (43, "seed43-python-epoch-shuffle-v4"),)
    assert next(field for field in ADDONS[("k1_joint_aggregation", "1")].config_fields
                if field.name == "seed").kind == "initialization_seed"
    for version in ("2", "3"):
        assert execution.TRAINING_ADAPTERS[("neural_atom_k1", version)].addon("k1_two_pass_mean").mode == "mean2"
    entry = execution.TRAINING_ADAPTERS[("gptrans_t", "2")].addon("gptrans_degree_bond_local")
    assert entry.mode == "degree_bond_local_ema999"
    assert entry.policy_hook == "molgap.gptrans_bond_local:policy"
    for historical in ("k1_fused_layout", "k1_initialized_layout"):
        assert (historical, "1") in ADDONS
        assert ADDONS[(historical, "1")].execution_modes == ()
