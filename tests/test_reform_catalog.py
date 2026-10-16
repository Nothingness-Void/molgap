"""Single catalog projection preserves the pre-reform executable capabilities."""
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from molgap import experiment_execution as execution
from molgap.experiment_catalog import ADDONS

ROOT=Path(__file__).resolve().parents[1]


def comparable(adapter):
    value=asdict(adapter)
    value["addons"]={ (a["name"],a["version"]):a for a in value["addons"] }
    return value


def test_every_previous_execution_mapping_is_preserved():
    raw=subprocess.check_output(["git","show","11790547:src/molgap/experiment_execution.py"],cwd=ROOT,text=True)
    module=ModuleType("molgap._reform_baseline");module.__package__="molgap"
    sys.modules[module.__name__]=module
    try:
        exec(compile(raw,"pinned-before-reform","exec"),module.__dict__)
        assert set(module.TRAINING_ADAPTERS)==set(execution.TRAINING_ADAPTERS)
        for key,before in module.TRAINING_ADAPTERS.items():
            assert comparable(before)==comparable(execution.TRAINING_ADAPTERS[key]),key
    finally:sys.modules.pop(module.__name__,None)


def test_ordinary_addon_execution_is_derived_from_its_one_catalog_entry(monkeypatch):
    original=ADDONS[("k1_readout_scale","1")]
    contract=replace(original,exclusive_group="independent-fixture")
    monkeypatch.setattr(execution,"ADDONS",{**ADDONS,("new_fixture_scale","1"):contract})
    derived=execution._catalog_training_adapters()[("neural_atom_k1","3")]
    addon=derived.addon("new_fixture_scale")
    assert addon.apply_hook==contract.apply_hook and addon.mode=="composed"
    assert len(derived.addons)==7


def test_declaration_only_and_cpu_families_are_not_silently_promoted():
    execution.validate_training_registry()
    assert ("neural_atom_k1","1") not in execution.TRAINING_ADAPTERS
    assert ("gptrans_scale_ema","1") not in execution.TRAINING_ADAPTERS
    for key in (("ppgt_small","1"),("esa_small","1"),("esa_small","2")):
        assert execution.TRAINING_ADAPTERS[key].cpu_synthetic_only
