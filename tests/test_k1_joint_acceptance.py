"""Synthetic metadata checks for the objective adapter; no model execution."""
import ast
import copy
from pathlib import Path

import pytest

from molgap.k1_joint_acceptance import verify_objective_preflight
from molgap.k1_joint_objective import objective_config, objective_fingerprint


def test_objective_preflight_requires_actual_rng_and_gradient_checks():
    recipe = "k1_corrupt_gap_atom_aux"
    keys = ("accepted", "corruption_replayable", "selected_rows_are_other_categories",
        "finite_component_losses", "finite_gradients", "finite_optimizer", "clean_eval_finite",
        "clean_eval_no_corruption", "resume_counter_replayable", "head_initialization_rng_unchanged",
        "same_encoder_initialization", "auxiliary_head_gradient_ok")
    checks = {k: True for k in keys}
    checks.update(objective_config=objective_config(recipe), objective_fingerprint=objective_fingerprint(recipe), physical_batch_size=128)
    verify_objective_preflight({"objective_training": checks}, recipe)
    for key in keys:
        broken = copy.deepcopy(checks)
        broken[key] = False
        with pytest.raises(ValueError, match="Missing objective preflight"):
            verify_objective_preflight({"objective_training": broken}, recipe)
    with pytest.raises(ValueError, match="Preflight objective changed"):
        verify_objective_preflight({"objective_training": checks}, "k1_corrupt_gap")


def test_acceptance_reuses_existing_artifact_adapter_not_model_factory():
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/molgap/k1_joint_acceptance.py").read_text()
    ast.parse(text)
    assert "shared._load_arm(root, MODE, arm_directory=arm)" in text
    assert "make_encoder(" not in text and "model.load_state_dict(" not in text
    assert "pending_controller_terminal_closure" in text
    assert 'field not in {"loss_fingerprint", "target_transform_fingerprint"}' in text
    assert '"Undeclared contract delta: {field}"' in text
    assert 'counts[RECIPES[0]] == counts[RECIPES[1]]' in text
    assert "NO_TRAIN" in text


def test_original_acceptance_directory_override_is_optional():
    from molgap.k1_joint_acceptance import _shared
    import inspect
    assert inspect.signature(_shared()._load_arm).parameters["arm_directory"].default is None
