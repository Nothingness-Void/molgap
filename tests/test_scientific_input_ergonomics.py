"""Compact inputs retain the compiler's source, authority and baseline boundaries."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from molgap.experiment_execution import build_addon_declaration
from molgap.experiment_input import freeze_experiment_input, resolve_experiment_input
from molgap.family_training_runner import read_recipe
from test_experiment_workflow import _git, _pointer, _write_json
from test_scientific_input import scientific


@pytest.fixture
def compact(scientific, request):
    case = scientific
    name = "ergonomic-" + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:12]
    # The fixture was originally authored from dataclasses.asdict(), whose
    # tuple fields become JSON arrays at the input-file boundary. Compare the
    # resolver's wire-format round trip against the same JSON-normalized value.
    value = json.loads(json.dumps(case.value))
    value.update(experiment_id=name, logical_run_id=name + "-run")
    path = case.repo / "experiments" / name / "input.json"
    return SimpleNamespace(repo=case.repo, path=path, value=value)


def commit(case, *additional):
    _write_json(case.path, case.value)
    _git(case.repo, "add", "--", case.path.relative_to(case.repo).as_posix(), *additional)
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic compact input")


def inline(case):
    for name in ("authority", "platform"):
        case.value[name] = json.loads((case.repo / case.value.pop(name + "_ref")).read_bytes())


@pytest.mark.parametrize("form", ["string", "name", "full"])
def test_addon_shorthand_uses_existing_catalog_defaults(compact, form):
    case = compact
    arm = case.value["arms"][0]
    family = arm["model"]["family"], arm["model"]["version"]
    name = "k1_readout_scale" if family[0] == "neural_atom_k1" else "gptrans_readout_scale"
    supplied = name if form == "string" else {"name": name}
    if form == "full": supplied.update(version="1", config=None)
    arm["model"]["addons"] = [supplied]
    arm.update(training={}, purpose="architecture_comparison")
    commit(case)
    raw, spec, _, _ = resolve_experiment_input(case.path, case.repo)
    assert raw == case.value
    assert spec.to_dict()["arms"][0]["addons"] == [
        build_addon_declaration(family, name, repo_root=case.repo)]


def test_inline_declarations_freeze_original_identity_across_head_changes(compact):
    case = compact
    inline(case)
    commit(case)
    raw, spec, plan, paths = resolve_experiment_input(case.path, case.repo)
    assert raw == case.value and "authority_ref" not in raw and "platform_ref" not in raw
    authority_paths = [name for name in plan["generated_records"] if name.endswith("/authority.json")]
    assert len(authority_paths) == 1
    assert json.loads((case.repo / authority_paths[0]).read_bytes()) == case.value["authority"]
    freeze_experiment_input(case.path, case.repo)
    run = paths.run_directory(case.value["logical_run_id"], "v1")
    before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    binding = json.loads(before["inputs/input_binding.json"])
    assert binding["authority_ref"] == authority_paths[0]
    original_source = binding["source_commit"]
    note = case.repo / "ergonomic-unrelated.md"
    note.write_text("Independent source history advance\n", encoding="utf-8")
    _git(case.repo, "add", "--", note.name)
    _git(case.repo, "-c", "commit.gpgsign=false", "commit", "-m", "Unrelated source change")
    assert _git(case.repo, "rev-parse", "HEAD") != original_source
    # Reentry uses the frozen plan, including when original development pins vanish.
    (case.repo / authority_paths[0]).unlink()
    again, frozen_spec, frozen_plan, _ = resolve_experiment_input(case.path, case.repo)
    assert again == raw and frozen_spec.identity == spec.identity and frozen_plan == plan
    freeze_experiment_input(case.path, case.repo)
    assert before == {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    case.value["authority"]["allocated_t4_hours_ceiling"] += 1
    _write_json(case.path, case.value)
    with pytest.raises(ValueError, match="Frozen scientific input changed"):
        resolve_experiment_input(case.path, case.repo)


@pytest.mark.parametrize("fault", ["both_authority", "both_platform", "missing_authority", "invalid_authority",
                                  "missing_platform", "invalid_platform", "missing_role", "empty_decision"])
def test_inline_forms_do_not_waive_existing_validation_or_publish_early(compact, fault):
    case = compact
    inline(case)
    extra = []
    if fault == "both_authority": case.value["authority_ref"] = "experiments/question/authority.json"
    elif fault == "both_platform": case.value["platform_ref"] = "experiments/question/platform.json"
    elif fault == "missing_authority": del case.value["authority"]
    elif fault == "missing_platform": del case.value["platform"]
    elif fault == "invalid_authority": case.value["authority"]["user_authorized"] = False
    elif fault == "invalid_platform": del case.value["platform"]["kaggle"]["account"]
    elif fault == "missing_role": case.value["role_ref"] = "experiments/no-role.json"
    elif fault == "empty_decision":
        name = case.path.parent.relative_to(case.repo).as_posix() + "/empty.md"
        (case.repo / name).parent.mkdir(parents=True, exist_ok=True)
        (case.repo / name).write_text("", encoding="utf-8")
        case.value["decision_ref"] = name
        extra.append(name)
    commit(case, *extra)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        resolve_experiment_input(case.path, case.repo)
    assert not (case.path.parent / "generated").exists()


def install_preset(case):
    arm = case.value["arms"][0]
    reference_plan = json.loads((case.repo / arm["reference_ref"]["plan_ref"]).read_bytes())
    reference = next(e for e in reference_plan["arms"] if e["arm_id"] == arm["reference_ref"]["arm_id"])
    record, recipe = read_recipe(case.repo / arm["recipe_ref"])
    # The legacy synthetic reference fixture has placeholder metadata. Supply
    # truthful encoding metadata before pinning it as a new preset fixture.
    target_path = case.repo / reference["reference_artifacts"]["target_manifest"]["path"]
    _write_json(target_path, {"development_target_sha256": record["acceptance_requirements"]["target_sha256"],
                              "target_encoding": "little-endian-float32-contiguous-raw-bytes"})
    target = _pointer(case.repo, target_path)
    for entry in reference_plan["arms"]:
        entry["reference_artifacts"]["target_manifest"] = target
    _write_json(case.repo / arm["reference_ref"]["plan_ref"], reference_plan)
    preset = {"format": "molgap-scientific-baseline-preset-v1", "name": "synthetic-reviewed-screen",
        "version": "v1", "model": {k: arm["model"][k] for k in ("family", "version")},
        "recipe": _pointer(case.repo, case.repo / arm["recipe_ref"]),
        "data": {**_pointer(case.repo, case.repo / arm["data_ref"]["spec_ref"]), "arm_id": arm["data_ref"]["arm_id"]},
        "target_manifest": reference["reference_artifacts"]["target_manifest"],
        "initialization_state_sha256": recipe.initialization_sha256,
        "reference": {**_pointer(case.repo, case.repo / arm["reference_ref"]["plan_ref"]),
                      "arm_id": arm["reference_ref"]["arm_id"]}}
    path = case.path.parent / "baseline-v1.json"
    _write_json(path, preset)
    arm["preset_ref"] = _pointer(case.repo, path)
    for key in ("recipe_ref", "data_ref", "reference_ref"): arm.pop(key)
    arm["model"] = {"addons": []}
    return path, preset, [target["path"], preset["reference"]["path"]]


def test_explicit_preset_expands_exact_baseline_without_machine_defaults(compact):
    case = compact
    path, preset, dependencies = install_preset(case)
    commit(case, path.relative_to(case.repo).as_posix(), *dependencies)
    value, spec, plan, _ = resolve_experiment_input(case.path, case.repo)
    assert value == case.value
    assert spec.to_dict()["arms"][0]["family"] == {
        "name": preset["model"]["family"], "version": preset["model"]["version"]}
    assert spec.to_dict()["arms"][0]["initialization"]["state_sha256"] == preset["initialization_state_sha256"]
    assert plan["arms"][0]["initial_state"] == case.value["arms"][0]["initial_state_ref"]
    assert path.relative_to(case.repo).as_posix() in plan["source_records"]
    target = preset["target_manifest"]["path"]
    assert target in plan["source_records"]


def test_same_run_presets_supply_one_exact_target_without_changing_raw_input(compact):
    case = compact
    path, preset, dependencies = install_preset(case)
    preset["reference"] = None
    _write_json(path, preset)
    shared = deepcopy(case.value["arms"][0])
    shared.pop("arm_id")
    shared["preset_ref"] = _pointer(case.repo, path)
    shared["training"] = {}
    case.value.update(shared=shared,
        arms=[{"arm_id": "baseline", "training": {}},
              {"arm_id": "candidate", "training": {"learning_rate": .0008}}],
        same_run={"reference_arm_id": "baseline"},
        authority={"user_authorized": True, "allocated_t4_hours_ceiling": 1,
                   "arms": ["baseline", "candidate"], "seeds": [42],
                   "same_run_reference_arm_id": "baseline"})
    case.value.pop("authority_ref")
    commit(case, path.relative_to(case.repo).as_posix(), *dependencies)
    raw, spec, plan, _ = resolve_experiment_input(case.path, case.repo)
    assert raw == case.value and "target_manifest_ref" not in raw["same_run"]
    acceptance = json.loads((case.repo / plan["acceptance_plan"]).read_bytes())
    assert [arm["scientific_role"] for arm in spec.to_dict()["arms"]] == ["reference", "candidate"]
    assert all(entry["target_manifest"] == preset["target_manifest"] for entry in acceptance["arms"])
    freeze_experiment_input(case.path, case.repo)
    assert resolve_experiment_input(case.path, case.repo)[1].identity == spec.identity


@pytest.mark.parametrize("fault", ["pin", "recipe", "state", "target", "model", "missing_location", "version"])
def test_presets_reject_changed_or_missing_frozen_bindings(compact, fault):
    case = compact
    path, preset, dependencies = install_preset(case)
    arm = case.value["arms"][0]
    if fault == "pin": arm["preset_ref"]["sha256"] = "0" * 64
    elif fault == "recipe": arm["recipe_ref"] = "experiments/other-recipe.json"
    elif fault == "state": preset["initialization_state_sha256"] = "0" * 64
    elif fault == "target": preset["target_manifest"]["sha256"] = "0" * 64
    elif fault == "model": arm["model"]["version"] = "999"
    elif fault == "missing_location": arm.pop("initial_state_ref")
    elif fault == "version": preset["version"] = "latest"
    if fault in {"state", "target", "version"}:
        _write_json(path, preset)
        arm["preset_ref"] = _pointer(case.repo, path)
    commit(case, path.relative_to(case.repo).as_posix(), *dependencies)
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_experiment_input(case.path, case.repo)
    assert not (case.path.parent / "generated").exists()


def test_mechanical_pointers_and_candidate_policy_are_derived_when_absent(compact):
    case = compact
    folder = case.path.parent
    for name in ("protocol.md", "decision.md"):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("Retained " + name + "\n", encoding="utf-8")
    relative = folder.relative_to(case.repo).as_posix()
    policy_id = case.value["question"]["policy_id"]
    case.value["question"]["policy_id"] = policy_id + "-fresh"
    for key in ("monitor_paths", "role_ref", "decision_ref"): case.value.pop(key)
    commit(case, relative + "/protocol.md", relative + "/decision.md")
    value, _, _, _ = resolve_experiment_input(case.path, case.repo)
    assert value["monitor_paths"] == [relative + "/STATUS.md"]
    assert (value["role_ref"], value["decision_ref"]) == (relative + "/protocol.md", relative + "/decision.md")
    policy = json.loads((case.repo / "research_memory/policies" / (policy_id + "-fresh.1.json")).read_bytes())
    assert policy["status"] == "candidate" and policy["approval"]["approved_by"] is None
    assert policy["created_from_source_digest"] == hashlib.sha256((folder / "protocol.md").read_bytes()).hexdigest()
