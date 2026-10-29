"""Portable common-contract regressions; synthetic CPU/files only."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from molgap.common import io, paths, spec as common_spec, wire
from molgap import experiment_launch, experiment_spec, experiment_terminal, screen_policy
from test_experiment_spec import payload
from test_research_memory_plan_batch import _item, planning_root, plan_module


def test_existing_imports_share_the_common_mechanisms(payload):
    assert experiment_spec.FamilyContract is common_spec.FamilyContract
    assert experiment_spec.AddonContract is common_spec.AddonContract
    assert issubclass(experiment_spec.ExperimentSpec, common_spec.ExperimentSpec)
    assert experiment_launch.publish_immutable_bytes is io.publish_immutable_bytes
    assert experiment_launch._safe_local is io._safe_local
    assert experiment_terminal._path is paths.validate_relative_path
    assert screen_policy.canonical_fingerprint is wire.canonical_fingerprint
    expected = hashlib.sha256(json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")).hexdigest()
    assert experiment_spec.ExperimentSpec(payload).identity == expected


def test_compatibility_spec_remains_frozen_and_wire_only(payload, monkeypatch):
    value = experiment_spec.ExperimentSpec(payload)
    assert set(vars(value)) == {"_canonical_json"}
    for name, replacement in (("_canonical_json", "{}"), ("extra", "mutable")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, replacement)
    monkeypatch.setattr(experiment_spec, "FAMILIES", {})
    with pytest.raises(KeyError):
        value.family_contract("neural_atom_k1")
    with pytest.raises(ValueError, match="Unknown family"):
        experiment_spec.ExperimentSpec(payload)


def test_common_parser_takes_explicit_catalog_without_registering_it(payload):
    declaration = deepcopy(payload)
    arm = declaration["arms"][1]
    arm["family"]["name"] = "portable_k1"
    families = dict(experiment_spec.FAMILIES)
    families[("portable_k1", "1")] = replace(
        families[("neural_atom_k1", "1")], name="portable_k1")
    before = deepcopy(declaration)
    value = common_spec.ExperimentSpec(
        declaration, families=families, addons=experiment_spec.ADDONS)
    assert declaration == before
    families.clear()
    assert value.family_contract("neural_atom_k1").name == "portable_k1"
    with pytest.raises(ValueError, match="Unknown family"):
        experiment_spec.ExperimentSpec(declaration)


@pytest.mark.parametrize("attribute", ["_families", "unreviewed_override"])
def test_launch_snapshot_guard_still_rejects_injected_state(payload, tmp_path, attribute):
    value = experiment_spec.ExperimentSpec(payload)
    object.__setattr__(value, attribute, {})
    with pytest.raises(ValueError, match="Invalid ExperimentSpec snapshot"):
        experiment_launch._binding(value, tmp_path, "a" * 64)


def test_spec_publication_is_atomic_idempotent_and_never_overwrites(payload, tmp_path):
    value = experiment_spec.ExperimentSpec(payload)
    target = tmp_path / "spec.json"
    value.write(target)
    value.write(target)
    assert target.read_bytes() == value.to_json().encode("utf-8")
    changed = deepcopy(payload)
    changed["logical_run_id"] = "another-run"
    with pytest.raises(ValueError, match="conflict"):
        experiment_spec.ExperimentSpec(changed).write(target)
    assert target.read_bytes() == value.to_json().encode("utf-8")


@pytest.mark.parametrize("value", [
    "", " ../x", "../x", "/x", "x//y", "x/./y", "x/../y", "C:/x",
    "x\\y", "x/CON", "x/file.", "x/file ", "x/key:stream", "x/?", None, True,
])
def test_common_path_validation_rejects_nonportable_spellings(value):
    with pytest.raises(ValueError):
        paths.validate_relative_path(value)


def test_common_path_validation_accepts_portable_repository_path():
    assert paths.validate_relative_path("experiments/example/trace.json") is None


def test_reviewed_package_paths_do_not_admit_credentials():
    from molgap.experiment_package import _name
    for name in ("family_training_outputs", "checkpoint_io", "experiment_catalog", "k1_pair_token"):
        path = "src/molgap/" + name + ".py"
        assert _name(path) == path
    for name in ("src/molgap/api_token.py", "src/molgap/credentials.py",
                 "src/molgap/secrets/loader.py", "outputs/example.py",
                 "src/molgap/result.pt", "src/molgap/K1_PAIR_TOKEN.py"):
        with pytest.raises(ValueError):
            _name(name)


def test_new_common_owners_are_in_static_inventory_and_import_in_isolation(tmp_path):
    from molgap.experiment_source_inventory import COMMON_SOURCE_FILES
    from molgap.source_dependencies import discover_source_dependencies
    root = Path(__file__).resolve().parents[1]
    required = {"src/molgap/common/" + name + ".py"
                for name in ("io", "paths", "spec", "wire")}
    required.add("src/molgap/experiment_catalog.py")
    assert required <= set(COMMON_SOURCE_FILES)
    entry = "src/molgap/experiment_spec.py"
    closure = discover_source_dependencies(root, [entry])
    assert {"src/molgap/common/spec.py", "src/molgap/common/wire.py",
            "src/molgap/experiment_catalog.py"} <= set(closure.source_files)
    copied = tmp_path / "isolated"
    for name in set(closure.source_files) | required:
        destination = copied / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    probe = (
        "import pathlib,sys; sys.path.insert(0,sys.argv[1]); "
        "from molgap.experiment_spec import ExperimentSpec; "
        "from molgap.common import io,paths,spec,wire; "
        "paths.validate_relative_path('experiments/probe/spec.json'); "
        "root=pathlib.Path(sys.argv[1]).resolve(); "
        "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) "
        "for m in (io,paths,spec,wire)); "
        "assert 'molgap.server_control' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(copied / "src")],
        cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("owner", ["desktop", "server"])
def test_single_plan_validation_has_explicit_owner_and_no_publication(planning_root, owner):
    root, _ = planning_root
    item = _item("a")
    item["spec"]["trajectory"]["owner"] = owner
    def tree():  # Git byte-preservation qualification may add an empty-tree object to .git.
        return {p.relative_to(root): p.read_bytes() for p in root.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(root).parts}
    before = tree()
    result = plan_module.plan(root, item["spec"], item["output"], validate_only=True)
    assert result["status"] == "VALIDATED"
    assert before == tree()
    assert not (root / "research_memory/.staging").exists()


@pytest.mark.parametrize("owner", [None, "historical", "auto"])
def test_single_plan_never_infers_owner(planning_root, owner):
    root, _ = planning_root
    item = _item("a")
    if owner is None:
        item["spec"]["trajectory"].pop("owner")
    else:
        item["spec"]["trajectory"]["owner"] = owner
    with pytest.raises(plan_module.PlanBatchError, match="explicit desktop/server owner"):
        plan_module.plan(root, item["spec"], item["output"], validate_only=True)
    assert not (root / "experiments/arm-a").exists()


def test_invalid_validation_batch_retains_error_index_and_publishes_nothing(planning_root):
    root, _ = planning_root
    second = _item("b")
    second["spec"]["costs"][0]["measurement"]["device_hours"] = {
        "value": 1.0, "status": "measured"}
    with pytest.raises(plan_module.PlanBatchError) as caught:
        plan_module.plan_many(root, [_item("a"), second], validate_only=True)
    assert caught.value.failed_index == 1
    assert caught.value.completed_results == ()
    assert not (root / "experiments/arm-a").exists()
    assert not (root / "experiments/arm-b").exists()


def test_explicit_complete_clip_sequence_preserves_default_update():
    import torch
    from molgap.family_training_core import optimizer_update
    reference = torch.nn.Linear(2, 1)
    candidate = deepcopy(reference)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.1)
                  for model in (reference, candidate)]
    for model, optimizer, parameters in (
        (reference, optimizers[0], None),
        (candidate, optimizers[1], tuple(candidate.parameters())),
    ):
        optimizer_update(model, optimizer, lambda: (model(torch.ones(2, 2)).sum(), None),
                         clip_norm=0.5, check_finite=True, clip_parameters=parameters)
    assert all(torch.equal(a, b) for a, b in zip(reference.parameters(), candidate.parameters()))


@pytest.mark.parametrize("unit,expected", [("normalized-Gap", 0.5), ("eV", 0.25)])
def test_retained_training_metric_unit_is_explicit(tmp_path, unit, expected):
    from molgap.training_logging import TrainingLogger, TrainingLogPolicy
    recipe = SimpleNamespace(selection="best-live", target_std_eV=2.0)
    logger = TrainingLogger(
        tmp_path, SimpleNamespace(arm_id="arm"), recipe, TrainingLogPolicy(),
        observations=[{"event": "observation", "epoch_or_pass": 1,
                       "live_train_metric": 0.25, "live_dev_metric": 0.4}],
        observation_train_unit=unit)
    assert logger.train_mae == expected
    assert logger.dev_mae == 0.4
    assert not list(tmp_path.iterdir())
