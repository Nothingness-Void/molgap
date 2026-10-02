"""Review-only checks for selected-family source and import closure."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

pytest_plugins = ["test_experiment_workflow"]

from molgap import experiment_execution as execution
from molgap import kaggle_workflow
from molgap.experiment_package import build_experiment_source_package
from molgap.experiment_source_inventory import registered_source_files
from molgap.experiment_spec import ExperimentSpec
from molgap.experiment_workflow import prepare_workflow
from molgap.experiment_execution import validate_execution_plan
from molgap.experiment_preflight import check_release_inputs
from molgap.experiment_launch import canonical_json
from test_experiment_lifecycle_integration import (
    _prepare_reference_inputs,
    _rewrite_case,
    _git,
)
from test_experiment_launch_config import _trajectory
from molgap.research_memory.trace import json_bytes


def _install_split_synthetic_owners(repo: Path, monkeypatch):
    """Register distinct temporary owners so selection cannot hide cross-imports."""
    modules = {}
    for module_name in ("synthetic_k1", "synthetic_gptrans"):
        path = repo / "src" / "molgap" / (module_name + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "def validate_screen_recipe(spec, arm_id, recipe):\n"
            "    if recipe.get('format') != 'synthetic-screen-recipe-v1':\n"
            "        raise ValueError('synthetic recipe format mismatch')\n"
            f"    return {{'owner': '{module_name}', 'arm_id': arm_id}}\n",
            encoding="utf-8",
        )
        qualified = "molgap." + module_name
        module_spec = importlib.util.spec_from_file_location(qualified, path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        module_spec.loader.exec_module(module)
        modules[module_name] = path
    _git(repo, "add", "src/molgap/synthetic_k1.py", "src/molgap/synthetic_gptrans.py")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Synthetic split owners")

    from molgap.experiment_execution import TrainingAdapter, TrainingAddon

    registry = {
        ("neural_atom_k1", "2"): TrainingAdapter(
            ("neural_atom_k1", "2"), "molgap.synthetic_k1", "k1-screen-v1",
            (TrainingAddon("k1_joint_aggregation", "1", "ssma"),),
            source_files=("src/molgap/synthetic_k1.py",),
        ),
        ("gptrans_t", "1"): TrainingAdapter(
            ("gptrans_t", "1"), "molgap.synthetic_gptrans", "gptrans-v1",
            (TrainingAddon("pair_prenorm", "1", "pair_prenorm"),),
            source_files=("src/molgap/synthetic_gptrans.py",),
        ),
    }
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", MappingProxyType(registry))
    return modules


def _select_one_arm(case: dict, arm_id: str):
    repo = case["repo"]
    declaration = case["spec"].to_dict()
    declaration["arms"] = [arm for arm in declaration["arms"] if arm["arm_id"] == arm_id]
    declaration["prospective"]["arms"] = [
        binding for binding in declaration["prospective"]["arms"] if binding["arm_id"] == arm_id
    ]
    spec = ExperimentSpec(declaration)
    recipe = {arm_id: case["recipes"][arm_id]}
    # Reuse the already repaired target-transform/reference artifacts from the
    # lifecycle fixture; only the plan identity and selected arm are narrowed.
    acceptance_path = repo / case["plan"]["acceptance_plan"]
    acceptance = json.loads(acceptance_path.read_bytes())
    acceptance["spec_identity"] = spec.identity
    acceptance["arms"] = [
        entry for entry in acceptance["arms"] if entry["arm_id"] == arm_id
    ]
    acceptance_path.write_bytes(json_bytes(acceptance))
    acceptance_plan = case["plan"]["acceptance_plan"]
    plan = copy.deepcopy(case["plan"])
    plan["spec_identity"] = spec.identity
    plan["acceptance_plan"] = acceptance_plan
    plan["arms"] = [item for item in plan["arms"] if item["arm_id"] == arm_id]
    return spec, plan, recipe


@pytest.mark.parametrize(
    ("arm_id", "selected_module", "unselected_module"),
    [
        ("k1_candidate", "molgap.synthetic_k1", "molgap.synthetic_gptrans"),
        ("gptrans_candidate", "molgap.synthetic_gptrans", "molgap.synthetic_k1"),
    ],
)
def test_selected_family_source_and_release_closure_isolated(
    tmp_path, workflow_case, monkeypatch, arm_id, selected_module, unselected_module,
):
    repo = workflow_case["repo"]
    _prepare_reference_inputs(repo)
    module_paths = _install_split_synthetic_owners(repo, monkeypatch)
    full_spec, _, expected, recipes = _rewrite_case(
        workflow_case, repo, module_paths["synthetic_k1"]
    )
    case = {
        "repo": repo,
        "spec": full_spec,
        "plan": workflow_case["plan"],
        "expected": expected,
        "recipes": recipes,
        "initial_states": workflow_case["initial_states"],
    }
    spec, plan, selected_recipes = _select_one_arm(case, arm_id)

    result = prepare_workflow(spec, repo, plan, tmp_path / arm_id)
    assert result["status"] == "PREPARED_FOR_PLATFORM", json.dumps(result, indent=2)
    prepared = tmp_path / arm_id
    package = prepared / "package"
    source_dataset = prepared / "source_dataset"
    release = json.loads((prepared / "release_report.json").read_bytes())
    assert release["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED"
    assert release["errors"] == []
    required = set(release["inputs"]["required_modules"])
    assert selected_module in required
    assert unselected_module not in required

    source_manifest = json.loads((package / "SOURCE_FILES.json").read_bytes())
    source_paths = {entry["path"] for entry in source_manifest["files"]}
    assert ("src/molgap/" + selected_module.rsplit(".", 1)[1] + ".py") in source_paths
    assert ("src/molgap/" + unselected_module.rsplit(".", 1)[1] + ".py") not in source_paths

    recheck = check_release_inputs(
        spec,
        package,
        expected_package_identity=result["package_identity"],
        recipe_files=selected_recipes,
        initial_states={
            arm_id: source_dataset / "initial_states" / (arm_id + ".pt")
        },
        required_modules=release["inputs"]["required_modules"],
        input_root=source_dataset,
        entry_script=prepared / "kernel" / "run.py",
        launch_config=source_dataset / "experiment_launch.json",
        kernel_metadata=prepared / "kernel" / "kernel-metadata.json",
    )
    assert recheck["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED", recheck
    assert recheck["errors"] == []
    assert "clean_import:package" in recheck["checks"]


@pytest.mark.parametrize(
    ("arm_id", "selected_owner", "unselected_owner"),
    [
        ("k1_candidate", "molgap.k1_screen_training", "molgap.gptrans_screen_workflow"),
        ("gptrans_candidate", "molgap.gptrans_screen_workflow", "molgap.k1_screen_training"),
    ],
)
def test_actual_registered_owner_clean_import_and_inventory_closure(
    tmp_path, workflow_case, monkeypatch, arm_id, selected_owner, unselected_owner,
):
    """Run the release gate against actual registered owner source bytes only."""
    repo = workflow_case["repo"]
    # Rebind the synthetic recipe contracts/state hashes without executing an
    # actual trainer, then restore the production registry for package closure.
    _prepare_reference_inputs(repo)
    actual_registry = execution.TRAINING_ADAPTERS
    module_paths = _install_split_synthetic_owners(repo, monkeypatch)
    full_spec, _, expected, recipes = _rewrite_case(
        workflow_case, repo, module_paths["synthetic_k1"]
    )
    monkeypatch.setattr(execution, "TRAINING_ADAPTERS", actual_registry)
    spec_declaration = full_spec.to_dict()
    spec_declaration["arms"] = [
        arm for arm in spec_declaration["arms"] if arm["arm_id"] == arm_id
    ]
    spec_declaration["prospective"]["arms"] = [
        binding for binding in spec_declaration["prospective"]["arms"]
        if binding["arm_id"] == arm_id
    ]
    spec = ExperimentSpec(spec_declaration)
    recipe = {arm_id: recipes[arm_id]}
    source_names = registered_source_files(spec, recipe.values())
    root = Path(__file__).resolve().parents[1]
    for name in source_names:
        source = root / name
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name not in recipe.values():
            shutil.copyfile(source, target)
    _git(repo, "add", "--", *source_names)
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Actual registered owner sources")

    package = tmp_path / (arm_id + "-package")
    manifest = build_experiment_source_package(spec, repo, source_names, package)
    jobs = [{
        "arm_id": arm_id,
        "device": 0,
        "recipe": recipe[arm_id],
        "initial_state": "initial_states/" + arm_id + ".pt",
        "trajectory_id": next(
            item["trajectory_id"] for item in spec.to_dict()["prospective"]["arms"]
            if item["arm_id"] == arm_id
        ),
    }]
    selected_job = validate_execution_plan(spec, jobs)[0]
    stage_root = tmp_path / (arm_id + "-stage")
    stage_root.mkdir()
    stage = kaggle_workflow.stage_inputs(
        repo_root=repo,
        output=stage_root,
        package=package,
        manifest=manifest,
        spec=spec,
        platform_plan=workflow_case["plan"]["kaggle"],
        metadata=kaggle_workflow.validate_plan(spec, workflow_case["plan"]["kaggle"]),
        initial_states={arm_id: workflow_case["initial_states"][arm_id]},
        jobs=jobs,
    )
    trajectory = tmp_path / (arm_id + "-trajectory.json")
    trajectory.write_bytes(canonical_json(_trajectory(
        spec,
        spec.to_dict()["arms"][0],
        manifest["source_commit"],
        spec.to_dict()["prospective"]["arms"][0]["trajectory_id"],
    )).encode("utf-8"))
    kaggle_workflow.freeze_inputs(stage, {arm_id: trajectory})
    release = check_release_inputs(
        spec,
        package,
        expected_package_identity=manifest["package_identity"],
        recipe_files=recipe,
        initial_states={arm_id: stage["input_root"] / "initial_states" / (arm_id + ".pt")},
        required_modules=selected_job["required_modules"],
        input_root=stage["input_root"],
        entry_script=stage["entry_path"],
        launch_config=stage["launch_path"],
        kernel_metadata=stage["metadata_path"],
    )
    assert release["status"] == "LOCAL_RELEASE_INPUTS_VERIFIED", json.dumps(release, indent=2)
    assert release["errors"] == []
    origins = release["checks"]["clean_import:package"]
    assert selected_owner in origins
    assert unselected_owner not in origins

    paths = {
        entry["path"] for entry in json.loads((package / "SOURCE_FILES.json").read_bytes())["files"]
    }
    assert "src/molgap/" + selected_owner.rsplit(".", 1)[1] + ".py" in paths
    assert "src/molgap/" + unselected_owner.rsplit(".", 1)[1] + ".py" not in paths
    if arm_id == "gptrans_candidate":
        assert "src/molgap/gptrans_variants.py" in paths
