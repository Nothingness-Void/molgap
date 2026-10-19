"""Synthetic post-load addon registration through the shared family kernel."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from molgap.experiment_execution import TRAINING_ADAPTERS, build_addon_declaration, build_family_recipe
from molgap.family_training_data import DatasetSelection
from molgap.family_training_kernel import FamilyKernel, load_family_model
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings
from molgap.training_reproducibility import atomic_torch_save
from molgap.v4_runtime import state_dict_sha256


@pytest.mark.parametrize("family,name,mode", [
    ("neural_atom_k1", "k1_post_layernorm", "post_layernorm"),
    ("gptrans_t", "gptrans_bond_local_resistance", "degree_bond_local_resistance"),
])
def test_registered_post_load_recipe_factory_and_optimizer(tmp_path, family, name, mode):
    root = Path(__file__).resolve().parents[1]
    interface = (family, "3" if family == "neural_atom_k1" else "2")
    addon = build_addon_declaration(interface, name, repo_root=root, config={})
    with torch.random.fork_rng(devices=[]):
        if family == "neural_atom_k1":
            from molgap.qm9_neural_atom import make_encoder
            base = make_encoder("neural_atom_k1")
        else:
            from molgap.gptrans_bond_local import make_bond_local_model
            base = make_bond_local_model()
        digest = state_dict_sha256(base.state_dict())
        initial = tmp_path / "base.pt"
        atomic_torch_save(initial, base.state_dict())
        recipe = FamilyRecipe(family, mode, 42, digest, DatasetSelection("synthetic.json", "2" * 64, False),
            128, 0, 1, .001, .01, ScheduleSpec("cosine_epoch", 1), "final-live", "torch-pass-randperm", 1,
            target_mean_eV=0., target_std_eV=1., addons=(addon,),
            torch_version=str(torch.__version__).split("+", 1)[0])
        assert FamilyRecipe.from_dict(recipe.to_dict()) == recipe
        assert TRAINING_ADAPTERS[interface].mode({"addons": [addon]}) == mode
        record = build_family_recipe(interface, addon=name, configuration=recipe, train_role_identity="synthetic-role")
        assert record["addons"] == [addon]
        reference = replace(recipe, addons=(), mode="reference" if family == "neural_atom_k1"
                            else "degree_bond_local_ema999")
        for field in ("objective", "sampler", "transform"):
            assert family_arm_bindings(recipe)[field] == family_arm_bindings(reference)[field]
        model, _ = load_family_model(recipe, initial)
        assert model._molgap_substrate.sealed
        assert len(model._molgap_substrate.bindings) == 1
        if family == "gptrans_t":
            assert len(model.state_dict()) == len(base.state_dict()) + 1
            assert sum(p.numel() for p in model.parameters()) == 5_871_329
        else:
            assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
        del model, base
        kernel = FamilyKernel(recipe, initial, device="cpu")
        expected = {id(p) for p in kernel.model.parameters() if p.requires_grad}
        actual = [id(p) for group in kernel.optimizer.param_groups for p in group["params"] if p.requires_grad]
        assert len(actual) == len(set(actual)) and set(actual) == expected
        from test_family_training_kernel import synthetic_batch
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            batch = synthetic_batch()
            batch.y = torch.full_like(batch.y, 20.)
            loss, absolute, rows = kernel.step(batch, check_finite=True)
            assert torch.isfinite(loss) and torch.isfinite(absolute) and rows == 2
            assert kernel.optimizer.state
            if family == "gptrans_t":
                parameter = kernel.model._molgap_addons["resistance"].projection.weight
                assert parameter in kernel.optimizer.state
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        finally:
            torch.use_deterministic_algorithms(deterministic)
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.set_num_threads(threads)
        kernel.close()
        with pytest.raises(ValueError, match="Policy-backed"):
            replace(recipe, mode="composed")
        if family == "gptrans_t":
            with pytest.raises(ValueError):
                replace(recipe, addons=())
            with pytest.raises(ValueError):
                replace(recipe, mode="reference")
