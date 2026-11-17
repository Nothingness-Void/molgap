"""Default-entry wiring and synthetic arithmetic; no real roles or accelerator."""
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap import pcqm_gptrans_v4 as native
from molgap.experiment_execution import build_family_recipe
from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward
from molgap.gptrans_execution_tuning import execution_policy
from molgap.gptrans_screen_workflow import _validate_recipe_fields


@pytest.mark.parametrize("phase", ["run_preflight", "run_training"])
def test_native_entry_automatically_enters_and_restores_accelerated_scope(
        tmp_path, monkeypatch, phase):
    original_loader, original_fields = native._training_loader, native._scientific_fields
    threads = torch.get_num_threads()
    def sentry(seed):
        assert native._accelerated_execution_active
        assert native._training_loader is not original_loader
        assert torch.get_num_threads() == 1
        assert native._scientific_fields()["execution_policy"] == execution_policy()
        raise RuntimeError("synthetic boundary; no input loaded")
    monkeypatch.setattr(native, "configure_fp32_determinism", sentry)
    kwargs = dict(dataset_root=tmp_path, manifest_path=tmp_path / "unused.json",
        source_archive=tmp_path / "unused.tar.gz", source_archive_sha256="a" * 64,
        source_commit="b" * 40, output=tmp_path / phase, platform_id="synthetic",
        initial_state_path=tmp_path / "unused.pt")
    if phase == "run_training":
        kwargs["preflight_path"] = tmp_path / "unused-preflight.json"
    with pytest.raises(RuntimeError, match="synthetic boundary"):
        getattr(native, phase)(**kwargs)
    assert native._training_loader is original_loader
    assert native._scientific_fields is original_fields
    assert not hasattr(native, "_accelerated_execution_active")
    assert torch.get_num_threads() == threads
    report = json.loads((kwargs["output"] / "execution_report.json").read_text())
    assert report["remote_qualified"] is False
    assert report["tuning"] == execution_policy()["tuning"]


@pytest.mark.parametrize("phase", ["run_preflight", "run_training"])
def test_unwrapped_native_body_cannot_select_uncached_training(tmp_path, phase):
    kwargs = dict(dataset_root=tmp_path, manifest_path=tmp_path / "unused.json",
        source_archive=tmp_path / "unused.tar.gz", source_archive_sha256="a" * 64,
        source_commit="b" * 40, output=tmp_path / phase, platform_id="synthetic",
        initial_state_path=tmp_path / "unused.pt")
    if phase == "run_training":
        kwargs["preflight_path"] = tmp_path / "unused-preflight.json"
    with pytest.raises(RuntimeError, match="registered accelerated entrypoint"):
        getattr(native, phase).__wrapped__(**kwargs)
    with pytest.raises(RuntimeError, match="registered accelerated entrypoint"):
        native._make_training_state(tmp_path / "unused.pt")


@pytest.mark.parametrize("addon", [None, "pair_prenorm", "centered_logits", "memory_value", "memory_message"])
def test_all_gptrans_scientific_recipes_bind_default_execution(addon):
    recipe = build_family_recipe(("gptrans_t", "1"), addon=addon,
        source_idx_sha256="a" * 64, target_sha256="b" * 64)
    assert recipe["execution_policy"] == execution_policy()
    _validate_recipe_fields(recipe)
    del recipe["execution_policy"]
    with pytest.raises(ValueError, match="execution_policy"):
        _validate_recipe_fields(recipe)


@pytest.mark.parametrize("key,value", [("adamw_foreach", 1), ("adamw_foreach", False),
    ("cpu_layout", 1), ("ema", "per-step-state-dict")])
def test_gptrans_policy_drift_or_bool_alias_fails_static_validation(key, value):
    recipe = build_family_recipe(("gptrans_t", "1"),
        source_idx_sha256="a" * 64, target_sha256="b" * 64)
    recipe["execution_policy"][key] = value
    with pytest.raises(ValueError, match="execution_policy"):
        _validate_recipe_fields(recipe)


def synthetic_batch():
    return prepare_layout(Batch.from_data_list([
        Data(x=torch.zeros(n, 9, dtype=torch.long),
             edge_index=torch.tensor([[0, 1], [1, 0]]) if n > 1 else torch.empty(2, 0, dtype=torch.long),
             edge_attr=torch.zeros(2 if n > 1 else 0, 3, dtype=torch.long))
        for n in (1, 2, 4)]))


def test_fast_input_assembly_preserves_outputs_gradients_and_rng(monkeypatch):
    from molgap import gptrans
    from molgap.gptrans import OGBGPTransTiny
    from torch_geometric import utils
    from molgap.training_reproducibility import configure_fp32_determinism
    configure_fp32_determinism(42)
    model = OGBGPTransTiny(node_channels=16, pair_channels=8, num_layers=2,
                          num_heads=4, dropout=.1)
    batch = synthetic_batch()
    initial_rng = torch.get_rng_state().clone()
    outputs = []
    for accelerated in (False, True):
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(initial_rng)
        if accelerated:
            bind_model_layout(model)
            assembly = model._dense_inputs
            assembly_calls = []

            def forbidden(*args, **kwargs):
                pytest.fail("Prepared input assembly synchronized or rebuilt categorical dense inputs")

            def observed_assembly(*args, **kwargs):
                assembly_calls.append(True)
                # Keep the observation inside input assembly; later model blocks
                # and the parity assertions retain their ordinary tensor APIs.
                with monkeypatch.context() as scoped:
                    scoped.setattr(torch.Tensor, "item", forbidden)
                    scoped.setattr(torch, "equal", forbidden)
                    scoped.setattr(utils, "to_dense_batch", forbidden)
                    scoped.setattr(gptrans, "to_dense_batch", forbidden)
                    return assembly(*args, **kwargs)

            model._dense_inputs = observed_assembly
            prediction = prepared_forward(model, batch)
            assert assembly_calls == [True]
        else:
            prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
        prediction.sum().backward()
        outputs.append((prediction.detach().clone(),
            {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None},
            torch.get_rng_state().clone()))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1].keys() == outputs[1][1].keys()
    assert all(torch.equal(outputs[0][1][k], v) for k, v in outputs[1][1].items())
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert not hasattr(model, "_execution_layout")


def test_unprepared_batch_fails_before_model_execution():
    with pytest.raises(ValueError, match="CPU-prepared"):
        prepared_forward(None, object())


def test_retired_k1_speed_runner_fails_before_reading_config():
    from molgap.k1_local_speed import run
    with pytest.raises(ValueError, match="Retired"):
        run(Path("does-not-exist"), None, None)


def test_cpu_smoke_works_when_bootstrap_has_no_package_context(tmp_path, monkeypatch):
    import importlib.util
    from molgap.gptrans import OGBGPTransTiny
    path = Path(__file__).resolve().parents[1] / "src/molgap/experiment_preflight.py"
    spec = importlib.util.spec_from_file_location("isolated_preflight_bootstrap", path)
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    assert not bootstrap.__package__
    model = OGBGPTransTiny(node_channels=16, pair_channels=8, num_layers=2, num_heads=4)
    monkeypatch.setattr(native, "_make_model", lambda **kwargs: model)
    batch = synthetic_batch()
    batch.y = torch.tensor([.1, .2, .3])
    arm = {"family": {"name": "gptrans_t", "version": "1"}, "addons": [],
           "initialization": {"kind": "random", "seed": 42,
                              "state_sha256": native._state_sha256(model)}}
    result = {key: False for key in (*bootstrap.MODEL_CHECKS, "initial_state_checked")}
    bootstrap._model_smoke(native, arm, batch, (0., 1.), tmp_path / "scratch.pt",
                           torch.Generator().manual_seed(42), result)
    assert all(result[key] for key in (*bootstrap.MODEL_CHECKS, "initial_state_checked"))
