"""Research mechanisms traverse the real shared update/output/inspection owners."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch

from molgap.experiment_execution import TRAINING_ADAPTERS, build_addon_declaration
from molgap.experiment_family_workflow import RunContext, inspect_output, tensor_digest
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import build_screen_recipe
from molgap.model_substrate import ReadoutInput
from molgap.modern2d_structure_addons import CountConditionedReadout
from molgap.modern2d_training import Modern2DFormalTrainingProfile
from molgap.modern2d_execution import prepared_kernel_execution
from molgap.research_memory.trace import file_digest
from molgap.shared_model_adapter import apply_reviewed_addons, resolve_model_factory
from modern2d_fixtures import batch as fixture_batch, graph, model_config
from test_modern2d_formal_training import case, actual_session
from test_family_training_session import assert_equal


RESEARCH_ADDON_CASES = (
    ("ppgt_small", ("ppgt_research_mean_readout",), "batched-v1"),
    ("ppgt_small", ("ppgt_research_residual_scale",), "batched-v1"),
    ("ppgt_small", ("ppgt_research_both",), "batched-v1"),
    ("ppgt_small", ("ppgt_content_pair",), "batched-v1"),
    ("ppgt_small", ("ppgt_research_mean_readout", "ppgt_content_pair"), "batched-v1"),
    ("ppgt_small", ("ppgt_content_pair",), "prepared-v2"),
    ("esa_small", ("esa_identity_residual",), "original"),
    ("esa_small", ("esa_count_readout",), "original"),
    ("esa_small", ("esa_count_readout",), "prepared-v2"),
)


@pytest.mark.parametrize(
    "addon_family,addon_names,execution_mode", RESEARCH_ADDON_CASES,
    ids=["+".join(names) for _, names, _ in RESEARCH_ADDON_CASES],
)
def test_research_mechanism_session_resume_and_complete_outputs(
        case, tmp_path, addon_family, addon_names, execution_mode):
    if case.recipe.family != addon_family:
        pytest.skip("addon belongs to the other Modern2D formal family")
    addons = tuple(build_addon_declaration(case.recipe.interface, name,
        repo_root=Path(__file__).resolve().parents[1]) for name in addon_names)
    case.recipe = replace(case.recipe, mode="composed", addons=addons,
        formal_training=Modern2DFormalTrainingProfile("configured-screen", horizon_passes=4,
            horizon_steps=4, execution_mode=execution_mode),
        schedule=ScheduleSpec("cosine_step", 4, reference_epochs=4))
    first, saved = actual_session(case, []), []
    first.checkpoint = lambda progress, state: saved.append(deepcopy(state))
    with (prepared_kernel_execution(first.kernel)
          if execution_mode == "prepared-v2" else nullcontext()):
        first.run(should_pause=lambda: True)
    resumed = actual_session(case, [])
    resumed.load_state_dict(deepcopy(saved[-1]))
    with (prepared_kernel_execution(resumed.kernel)
          if execution_mode == "prepared-v2" else nullcontext()):
        resumed.run()
    full = actual_session(case, [])
    recipe = case.recipe
    record = build_screen_recipe(recipe.mode, family=recipe.interface, configuration=recipe,
        train_role_identity="3" * 64, development_role_identity="4" * 64,
        source_idx_sha256=tensor_digest(torch.tensor(case.data.development_source_indices), role="source_idx"),
        target_sha256=tensor_digest(torch.cat([g.y for g in case.data.development]), role="target"))
    contract = tmp_path / "research-contract.json"
    contract.write_text(json.dumps(record), encoding="utf-8")
    context = RunContext("fixture", "research-fixture", "candidate", "a" * 64, "b" * 64,
        recipe.family, recipe.family_version, "c" * 40, "d" * 64, "e" * 64,
        file_digest(contract), "local", "fixture", "fixture/run", None)
    outputs = SessionOutputs(tmp_path / "output", context, contract=contract,
        adapter="modern2d-formal-session-v1", trajectory_id="research-fixture-trajectory",
        train_role_identity=record["train_role_identity"])
    full.checkpoint, full.selected, full.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    with (prepared_kernel_execution(full.kernel)
          if execution_mode == "prepared-v2" else nullcontext()):
        progress = full.run()
    assert_equal(full.state_dict(), resumed.state_dict())
    runtime = dict(platform="local", account="fixture", precision="fp32",
        source_commit=context.source_commit, source_archive_sha256=context.source_archive_sha256)
    costs = [dict(metric=key, unit="seconds", status="missing", value=None,
        semantics=semantics, hardware="synthetic-cpu") for key, semantics in
        (("wall_seconds", "process_wall"), ("device_seconds", "allocated_device"))]
    report = outputs.complete(progress, runtime=runtime, costs=costs)
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert inspect_output(outputs.root, context=context, expected=record["acceptance_requirements"])["status"] == "MECHANICALLY_VERIFIED"
    for file in ("selected_model.pt", "last_checkpoint.pt", "development_predictions.pt", "canonical_trace.json"):
        assert (outputs.root / file).is_file()
    observations = json.loads((outputs.root / "canonical_trace.json").read_bytes())["observations"]
    assert observations[-1]["event"] == "observation"
    assert observations[-1]["optimizer_step"] == 4
    for session in (first, resumed, full):
        session.close()


def _formal_model(family):
    interface = (family, "3" if family == "ppgt_small" else "4")
    factory = resolve_model_factory(TRAINING_ADAPTERS[interface])
    return factory({"model_config": model_config(family)})


def _apply_catalog_addons(model, family, names):
    interface = (family, "3" if family == "ppgt_small" else "4")
    declarations = [build_addon_declaration(interface, name,
        repo_root=Path(__file__).resolve().parents[1]) for name in names]
    model = apply_reviewed_addons(TRAINING_ADAPTERS[interface],
        {"addons": declarations}, model)
    owner = model._molgap_substrate
    assert owner.bindings
    owner.seal()
    return model


@pytest.mark.parametrize("family,addon_name", [
    ("ppgt_small", "ppgt_content_pair"),
    ("esa_small", "esa_count_readout"),
], ids=["ppgt-content-pair", "esa-count-readout"])
def test_new_structural_addon_factory_identity_gradients_and_two_step_optimizer(
        family, addon_name):
    torch.manual_seed(42)
    base = _formal_model(family).eval()
    candidate = _formal_model(family)
    candidate.load_state_dict(deepcopy(base.state_dict()), strict=True)
    candidate = _apply_catalog_addons(candidate, family, (addon_name,))
    data = fixture_batch()

    with torch.no_grad():
        expected = base(data)
        actual = candidate.eval()(data)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    candidate.train()
    addon_parameters = tuple(candidate._molgap_addons[addon_name].parameters())
    assert addon_parameters and all(parameter.requires_grad for parameter in addon_parameters)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = candidate(data)
        assert output.shape == (2,) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in candidate.parameters() if parameter.requires_grad)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in addon_parameters)
        optimizer.step()

    for parameter in candidate.parameters():
        if not parameter.requires_grad:
            continue
        assert parameter in optimizer.state
        state = optimizer.state[parameter]
        assert int(state["step"]) == 2
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()
    assert not candidate._molgap_substrate._active


def test_ppgt_mean_readout_composes_with_content_pair_at_exact_initialization():
    torch.manual_seed(42)
    mean = _apply_catalog_addons(_formal_model("ppgt_small"), "ppgt_small",
        ("ppgt_research_mean_readout",))
    composed = _formal_model("ppgt_small")
    composed.load_state_dict(deepcopy(mean.state_dict()), strict=True)
    composed = _apply_catalog_addons(composed, "ppgt_small",
        ("ppgt_research_mean_readout", "ppgt_content_pair"))
    data = fixture_batch()
    with torch.no_grad():
        torch.testing.assert_close(composed.eval()(data), mean.eval()(data), atol=0, rtol=0)


def _count_readout(addon, ptr, embedding):
    batch = SimpleNamespace(ptr=torch.tensor(ptr, dtype=torch.int64))
    addon.begin(None, (batch,), {})
    try:
        return addon(ReadoutInput(embedding)).embedding
    finally:
        addon.finish(None, (), {}, None)


def test_count_readout_falsifies_repeat_identical_tokens_by_graph_size():
    token = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    one_token = token.mean(0, keepdim=True)
    two_identical_tokens = token.repeat(2, 1).mean(0, keepdim=True)
    assert torch.equal(one_token, two_identical_tokens)

    addon = CountConditionedReadout(graph_dim=4)
    with torch.no_grad():
        addon.scale.weight.fill_(1.0)
        addon.shift.weight.zero_()
    output = _count_readout(addon, [0, 1, 3], torch.cat((one_token, two_identical_tokens)))
    assert torch.isfinite(output).all()
    assert not torch.equal(output[0], output[1])


def test_count_readout_token_permutation_preserves_output():
    model = _apply_catalog_addons(_formal_model("esa_small"), "esa_small",
        ("esa_count_readout",)).eval()
    original = graph(6)
    other = original.clone()
    atom_order = torch.tensor([5, 2, 0, 4, 1, 3])
    inverse = torch.argsort(atom_order)
    edge_order = torch.arange(original.edge_index.shape[1] - 1, -1, -1)
    other.x = original.x[atom_order]
    other.edge_index = inverse[original.edge_index[:, edge_order]]
    other.edge_attr = original.edge_attr[edge_order]
    with torch.no_grad():
        before = model(Batch.from_data_list([original]))
        after = model(Batch.from_data_list([other]))
    torch.testing.assert_close(before, after, atol=4e-6, rtol=2e-5)


def test_count_readout_invalid_ptr_fails_closed_with_clean_frames():
    model = _apply_catalog_addons(_formal_model("esa_small"), "esa_small",
        ("esa_count_readout",))
    addon = model._molgap_addons["esa_count_readout"]
    invalid = SimpleNamespace(ptr=torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="explicit graph boundaries"):
        model(invalid)
    assert addon._frames == []
