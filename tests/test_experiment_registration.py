"""Reviewed addon declarations and dependency selection, without model execution."""
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from molgap import experiment_execution as execution
from molgap import experiment_spec as declarations
from molgap.experiment_source_inventory import registered_source_files
from molgap.v4_runtime import normalized_source_sha256
from test_experiment_workflow import _arm, _spec_payload


ROOT = Path(__file__).resolve().parents[1]


def test_addon_helper_binds_reviewed_config_and_actual_source():
    addon = execution.build_addon_declaration(("neural_atom_k1", "2"), "k1_joint_aggregation", repo_root=ROOT)
    assert addon["source_sha256"] == normalized_source_sha256(ROOT / "src/molgap/k1_joint_aggregation.py")
    arm = _arm("neural_atom_k1", "2", "candidate", addon="k1_joint_aggregation")
    arm["addons"] = [addon]
    declarations.ExperimentSpec(_spec_payload([arm]))
    with pytest.raises(ValueError, match="configuration"):
        execution.build_addon_declaration(("neural_atom_k1", "2"), "k1_joint_aggregation", repo_root=ROOT,
                                         config={**addon["config"], "kappa": 5})


def test_single_family_package_excludes_unselected_trainers_and_addons():
    spec = declarations.ExperimentSpec(_spec_payload([_arm("gptrans_t", "1", "candidate", addon="pair_prenorm")]))
    sources = registered_source_files(spec, ["recipes/arm.json"])
    assert "src/molgap/gptrans_variants.py" in sources
    assert "src/molgap/gptrans_screen_workflow.py" in sources
    assert "src/molgap/pcqm_wedge.py" in sources
    assert "src/molgap/k1_screen_training.py" not in sources
    assert "src/molgap/k1_joint_aggregation.py" not in sources
    assert "src/molgap/gptrans_memory.py" not in sources


def test_new_reviewed_addon_extends_config_and_dependencies_without_generic_dispatch_changes(tmp_path, monkeypatch):
    source = tmp_path / "src/molgap/reviewed_addon.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic reviewed mechanism\n", encoding="utf-8")
    contract = declarations.AddonContract("gptrans_t", "attention-replacement", "molgap.reviewed_addon",
        (declarations.AddonConfigField("width", "integer", minimum=2, maximum=8),))
    monkeypatch.setattr(declarations, "ADDONS", MappingProxyType({**declarations.ADDONS, ("reviewed_addon", "1"): contract}))
    monkeypatch.setattr(execution, "ADDONS", declarations.ADDONS)
    original = execution.TRAINING_ADAPTERS[("gptrans_t", "1")]
    addon = execution.TrainingAddon("reviewed_addon", "1", "reviewed_mode", ("src/molgap/addon_dependency.py",))
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", MappingProxyType({**execution.TRAINING_ADAPTERS,
        ("gptrans_t", "1"): replace(original, addons=(*original.addons, addon))}))
    declaration = execution.build_addon_declaration(("gptrans_t", "1"), "reviewed_addon", repo_root=tmp_path, config={"width": 4})
    arm = _arm("gptrans_t", "1", "candidate", addon="pair_prenorm")
    arm["addons"] = [declaration]
    spec = declarations.ExperimentSpec(_spec_payload([arm]))
    assert execution.training_adapter(arm).mode(arm) == "reviewed_mode"
    assert {"src/molgap/reviewed_addon.py", "src/molgap/addon_dependency.py"} <= set(registered_source_files(spec))
    arm["addons"][0]["config"]["width"] = 9
    with pytest.raises(ValueError, match="requires width"):
        declarations.ExperimentSpec(_spec_payload([arm]))


def test_registration_rejects_undeclared_execution_addon(monkeypatch):
    original = execution.TRAINING_ADAPTERS[("gptrans_t", "1")]
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", {("gptrans_t", "1"): replace(original,
        addons=(execution.TrainingAddon("unreviewed", "1", "unreviewed"),))})
    with pytest.raises(ValueError, match="declaration contract"):
        execution.validate_training_registry()


def test_capability_route_does_not_upgrade_declaration_only_addon():
    arm = _arm("gptrans_t", "1", "candidate", addon="degree_scale")
    spec = declarations.ExperimentSpec(_spec_payload([arm], platform="kaggle"))
    result = execution.workflow_capabilities(spec)
    assert result["preparation_entry"] == "prepare-release"
    assert "candidate" in result["unsupported_training_arms"]
    assert result["submitted"] is False


def test_literal_config_rejects_boolean_as_integer():
    contract = declarations.ADDONS[("k1_joint_aggregation", "1")]
    config = {field.name: field.value for field in contract.config_fields}
    config["kappa"] = True
    with pytest.raises(ValueError, match="configuration"):
        declarations.validate_addon_config(contract, config)
