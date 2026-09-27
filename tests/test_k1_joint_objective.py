"""Pure contract and synthetic-helper tests for the remote-only joint objective."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from molgap.k1_joint_objective import (
    ATOM_FEATURE_DIMS,
    AugmentationCounter,
    augmentation_seed,
    corrupt_atom_features,
    mean_atom_reconstruction_ce,
    objective_config,
    objective_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_SOURCE = ROOT / "src" / "molgap" / "k1_joint_objective.py"
RUNNER_SOURCE = ROOT / "src" / "molgap" / "pcqm_k1_variants_runner.py"


def _canonical_fingerprint(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_atom_features(rows: int = 10_000):
    torch = pytest.importorskip("torch")
    x = torch.empty((rows, len(ATOM_FEATURE_DIMS)), dtype=torch.long)
    for field, categories in enumerate(ATOM_FEATURE_DIMS):
        x[:, field] = torch.arange(rows, dtype=torch.long) % categories
    return x


def test_objective_module_is_lazy_and_has_no_local_model_entrypoint():
    tree = ast.parse(OBJECTIVE_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith(("torch", "torch_geometric")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("torch", "torch_geometric"))
        elif isinstance(node, ast.ClassDef):
            assert all(
                not (isinstance(base, ast.Name) and base.id.endswith("Module"))
                for base in node.bases
            )
    source = OBJECTIVE_SOURCE.read_text(encoding="utf-8")
    assert "torch.random.default_generator.manual_seed(SEED)" in source
    assert "torch.random.get_rng_state()" in source
    assert "torch.cuda.get_rng_state_all()" in source
    assert "auxiliary-head initialization changed CPU or CUDA RNG state" in source
    assert "torch.manual_seed(SEED)" not in source


def test_recipe_config_and_fingerprint_are_canonical_and_complete():
    gap = objective_config("k1_corrupt_gap")
    aux = objective_config("k1_corrupt_gap_atom_aux")
    assert gap["recipe_id"] == "k1_corrupt_gap"
    assert aux["recipe_id"] == "k1_corrupt_gap_atom_aux"
    assert gap["model_mode"] == aux["model_mode"] == "neural_atom_k1_v4"
    assert gap["physical_batch_size"] == aux["physical_batch_size"] == 128
    assert gap["optimizer_steps"] == aux["optimizer_steps"] == 31_240
    assert gap["corruption"]["feature_count"] == 9
    assert gap["corruption"]["category_dims"] == list(ATOM_FEATURE_DIMS)
    assert gap["corruption"]["edge_index_unchanged"] is True
    assert gap["corruption"]["edge_attr_unchanged"] is True
    assert gap["corruption"]["random_walk_pe_unchanged"] is True
    assert gap["rng"]["device"] == "cpu"
    assert gap["rng"]["global_rng_draws"] is False
    assert gap["auxiliary_loss"]["weight"] == 0.0
    assert aux["auxiliary_loss"]["weight"] == 0.1
    assert aux["auxiliary_heads"]["count"] == 9
    assert aux["auxiliary_heads"]["input_units"] == 192
    assert aux["auxiliary_heads"]["output_units"] == list(ATOM_FEATURE_DIMS)
    assert aux["gradient_clipping"] == {
        "backbone_max_norm": 1.0,
        "auxiliary_heads_max_norm": 1.0,
        "separate_parameter_groups": True,
    }
    assert gap["target_transform_asset"]["computed_train_statistics_must_equal_asset"] is True
    assert objective_fingerprint(gap) == _canonical_fingerprint(gap)
    assert objective_fingerprint("k1_corrupt_gap_atom_aux") == _canonical_fingerprint(aux)
    assert objective_config({"recipe_id": "k1_corrupt_gap"}) == gap


def test_corruption_is_counter_replayable_and_changes_all_selected_fields():
    torch = pytest.importorskip("torch")
    x = _valid_atom_features()
    first = corrupt_atom_features(x, epoch=3, step=17)
    second = corrupt_atom_features(x, epoch=3, step=17)
    assert first.generator_seed == second.generator_seed == augmentation_seed(42, 3, 17)
    assert first.selected_count > 0
    assert torch.equal(first.selected_mask, second.selected_mask)
    assert torch.equal(first.corrupted_x, second.corrupted_x)
    assert torch.equal(first.original_x, x)
    assert torch.equal(first.corrupted_x[~first.selected_mask], x[~first.selected_mask])
    for field in range(len(ATOM_FEATURE_DIMS)):
        assert torch.all(
            first.corrupted_x[first.selected_mask, field]
            != x[first.selected_mask, field]
        )
        assert int(first.corrupted_x[:, field].min()) >= 0
        assert int(first.corrupted_x[:, field].max()) < ATOM_FEATURE_DIMS[field]
    empty = corrupt_atom_features(x, epoch=3, step=17, rate=0.0)
    assert empty.selected_count == 0
    assert torch.equal(empty.corrupted_x, x)


def test_zero_selected_reconstruction_loss_is_differentiable_zero():
    torch = pytest.importorskip("torch")
    x = _valid_atom_features(rows=32)
    logits = [
        torch.randn((x.shape[0], categories), requires_grad=True)
        for categories in ATOM_FEATURE_DIMS
    ]
    selected = torch.zeros(x.shape[0], dtype=torch.bool)
    loss = mean_atom_reconstruction_ce(logits, x, selected)
    assert loss.requires_grad
    assert float(loss) == 0.0
    loss.backward()
    assert all(head.grad is not None and torch.equal(head.grad, torch.zeros_like(head.grad)) for head in logits)


def test_augmentation_counter_roundtrips_and_enforces_order():
    counter = AugmentationCounter()
    seed = counter.seed_for(0, 0)
    counter.advance(0, 0)
    restored = AugmentationCounter()
    restored.load_state_dict(counter.state_dict())
    assert restored.next_epoch == 0
    assert restored.next_step == 1
    assert restored.optimizer_steps == 1
    assert restored.last_generator_seed == seed
    with pytest.raises(RuntimeError):
        restored.advance(0, 0)


def test_runner_integration_is_scoped_and_preflight_forward_has_one_argument():
    tree = ast.parse(RUNNER_SOURCE.read_text(encoding="utf-8"))
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_arm")
    parameters = {arg.arg for arg in train.args.kwonlyargs}
    assert {"objective_recipe", "target_transform_asset"} <= parameters

    preflight_calls = [
        node
        for node in ast.walk(train)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_objective_preflight"
    ]
    assert len(preflight_calls) == 1
    forward_arg = preflight_calls[0].args[3]
    assert isinstance(forward_arg, ast.Lambda)
    assert len(forward_arg.args.args) == 1
    assert isinstance(forward_arg.body, ast.Call)
    assert isinstance(forward_arg.body.func, ast.Name)
    assert forward_arg.body.func.id == "_forward"
    assert len(forward_arg.body.args) == 2

    architecture = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_architecture_preflight"
    )
    assert any(isinstance(node, ast.Return) and node.value is not None for node in ast.walk(architecture))
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    assert "validate_target_transform_asset" in source
    assert "verify_frozen_train_targets(" in source
    assert '"mean_eV": float(transform["mean"])' in source
    assert '"sample_std_eV": float(transform["std"])' in source
    assert '"target_transform_file_sha256"' in source
    assert '"target_transform_asset_sha256"' in source
    assert "clip_joint_gradients(model, joint_objective)" in source
    assert '"trajectory_id": trajectory_id' in source
    assert '"physical_run_id": physical_run_id' in source
    assert "load_canonical_trace(output / \"canonical_trace.json\")" in source
    assert "Joint-objective resume copied" in source
    assert "total_atom_rows" in source
    assert "gradient_clip_group_count" in source
