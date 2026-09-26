"""Synthetic EdgeState training-core tests; no dataset or platform access."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.edge_state_adapter import build_edge_state_model
from molgap.edge_state_training_core import (
    EdgeStateTrainingBinding,
    EpochPermutationBatchSampler,
    construct_bound_model,
    evaluate_development,
    normalized_gap_step,
    restore_checkpoint,
    save_checkpoint,
    training_target_statistics,
    validate_ogb_gap_batch,
)
from molgap.experiment_spec import ExperimentSpec
from molgap.v4_runtime import model_state_sha256


@pytest.fixture(scope="module")
def bound_spec():
    example = Path(__file__).resolve().parents[1] / "docs/operations/examples/edge_state_v1.json"
    payload = json.loads(example.read_text(encoding="utf-8"))
    payload["arms"] = [payload["arms"][1]]
    arm_id = payload["arms"][0]["arm_id"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial = build_edge_state_model(ExperimentSpec(payload), arm_id)
    payload["arms"][0]["initialization"]["state_sha256"] = model_state_sha256(initial)
    spec = ExperimentSpec(payload)
    binding = EdgeStateTrainingBinding.from_spec(
        spec, arm_id,
        source_commit="a" * 40,
        source_package_sha256="b" * 64,
        graph_manifest_sha256="c" * 64,
        training_contract_sha256="d" * 64,
        runtime_fingerprint="e" * 64,
        train_rows=10,
        physical_batch_size=2,
        target_mean_eV=6.5,
        target_std_eV=0.5,
    )
    return spec, binding


def _model(spec, binding):
    torch.manual_seed(42)
    return construct_bound_model(spec, binding)


def _batch(indices=(10, 11)):
    graphs = []
    for source_idx in indices:
        graphs.append(Data(
            x=torch.zeros((2, 9), dtype=torch.long),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.zeros((2, 3), dtype=torch.long),
            random_walk_pe=torch.zeros((2, 16), dtype=torch.float32),
            y=torch.tensor([6.0 + source_idx / 100], dtype=torch.float32),
            source_idx=torch.tensor([source_idx], dtype=torch.long),
        ))
    return Batch.from_data_list(graphs)


def test_binding_rejects_changed_spec_or_initial_state(bound_spec):
    spec, binding = bound_spec
    assert model_state_sha256(_model(spec, binding)) == binding.initialization_sha256
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        EdgeStateTrainingBinding.from_spec(
            spec, binding.arm_id, source_commit="a" * 40,
            source_package_sha256="invalid", graph_manifest_sha256="c" * 64,
            training_contract_sha256="d" * 64, runtime_fingerprint="e" * 64,
            train_rows=10, physical_batch_size=2,
            target_mean_eV=6.5, target_std_eV=0.5,
        )
    with pytest.raises(ValueError, match="does not match ExperimentSpec"):
        construct_bound_model(spec, replace(binding, graph_manifest_sha256="f" * 64,
                                            arm_identity="f" * 64))
    with pytest.raises(RuntimeError, match="Initial EdgeState model state differs"):
        construct_bound_model(spec, binding)


@pytest.mark.parametrize("field,mutate,match", [
    ("x", lambda b: b.x.__setitem__((0, 0), 119), "categorical range"),
    ("edge_attr", lambda b: b.edge_attr.__setitem__((0, 0), 5), "categorical range"),
    ("random_walk_pe", lambda b: b.random_walk_pe.__setitem__((0, 0), float("nan")), "RWSE16"),
    ("y", lambda b: b.y.__setitem__(0, float("nan")), "finite Gap target"),
    ("source_idx", lambda b: b.source_idx.__setitem__(1, 10), "unique nonnegative"),
    ("batch", lambda b: b.batch.__setitem__(0, 3), "Graph IDs"),
    ("edge_index", lambda b: b.edge_index.__setitem__((1, 0), 2), "crosses graph"),
])
def test_batch_contract_rejects_invalid_graphs(field, mutate, match):
    batch = _batch()
    assert validate_ogb_gap_batch(batch) == 2
    mutate(batch)
    with pytest.raises(ValueError, match=match):
        validate_ogb_gap_batch(batch)


def test_one_step_and_source_aligned_development(bound_spec):
    spec, binding = bound_spec
    with torch.random.fork_rng(devices=[]):
        model = _model(spec, binding)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial = model_state_sha256(model)
        result = normalized_gap_step(model, optimizer, _batch(), binding, clip_norm=1.0)
        assert result["rows"] == 2
        assert result["normalized_l1"] >= 0
        assert model_state_sha256(model) != initial
        output = evaluate_development(model, [_batch((11, 10))], binding,
                                      expected_source_idx=[10, 11])
        assert output["source_idx"].tolist() == [10, 11]
        assert output["prediction_eV"].shape == (2,)
        assert output["mae_eV"] >= 0
        assert model.training
        reordered = evaluate_development(model, [_batch()], binding,
                                         expected_source_idx=[11, 10])
        assert reordered["source_idx"].tolist() == [11, 10]
        with pytest.raises(ValueError, match="approved role"):
            evaluate_development(model, [_batch()], binding, expected_source_idx=[10, 12])
        with pytest.raises(ValueError, match="integer development"):
            evaluate_development(model, [_batch()], binding, expected_source_idx=[10.0, 11.0])
        non_fp32 = _batch()
        non_fp32.y = non_fp32.y.double()
        with pytest.raises(ValueError, match="FP32"):
            normalized_gap_step(model, optimizer, non_fp32, binding, clip_norm=1.0)
        with pytest.raises(ValueError, match="physical batch size"):
            normalized_gap_step(model, optimizer, _batch((10,)), binding, clip_norm=1.0)


def test_checkpoint_restores_training_state_and_rejects_wrong_identity(bound_spec, tmp_path):
    spec, binding = bound_spec
    path = tmp_path / "last.pt"
    with torch.random.fork_rng(devices=[]):
        model = _model(spec, binding)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        generator = torch.Generator().manual_seed(7)
        normalized_gap_step(model, optimizer, _batch(), binding, clip_norm=1.0)
        scheduler.step()
        sampler = EpochPermutationBatchSampler(10, 2, seed=42, epoch=0)
        receipt = save_checkpoint(
            path, binding, model, optimizer, scheduler,
            epoch=0, batch_offset=1, optimizer_steps=1, sample_presentations=2,
            sampler_state=sampler.state_for(1),
            loader_generator=generator,
        )
        assert len(receipt["sha256"]) == 64
        expected_generator_value = torch.rand(1, generator=generator)
        normalized_gap_step(model, optimizer, _batch(), binding, clip_norm=1.0)
        scheduler.step()
        expected_state = copy.deepcopy(model.state_dict())
        expected_lr = optimizer.param_groups[0]["lr"]

        restarted = _model(spec, binding)
        resumed_optimizer = torch.optim.AdamW(restarted.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(
            resumed_optimizer, step_size=1, gamma=0.8,
        )
        resumed_generator = torch.Generator().manual_seed(99)
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            restore_checkpoint(path, "0" * 64, binding, restarted,
                               resumed_optimizer, resumed_scheduler,
                               loader_generator=resumed_generator)
        with pytest.raises(ValueError, match="identity or format mismatch"):
            restore_checkpoint(path, receipt["sha256"],
                               replace(binding, graph_manifest_sha256="0" * 64),
                               restarted, resumed_optimizer, resumed_scheduler,
                               loader_generator=resumed_generator)
        with pytest.raises(ValueError, match="resume state is incomplete"):
            restore_checkpoint(path, receipt["sha256"], binding,
                               restarted, resumed_optimizer, resumed_scheduler)
        restored = restore_checkpoint(path, receipt["sha256"], binding,
                                      restarted, resumed_optimizer, resumed_scheduler,
                                      loader_generator=resumed_generator)
        assert restored["cursor"] == receipt["cursor"]
        assert restored["sampler_state"]["next_batch"] == 1
        resumed_sampler = EpochPermutationBatchSampler.from_state(
            restored["sampler_state"], dataset_size=10, batch_size=2, seed=42, epoch=0,
        )
        assert list(resumed_sampler) == list(sampler)[1:]
        assert torch.equal(torch.rand(1, generator=resumed_generator), expected_generator_value)
        normalized_gap_step(restarted, resumed_optimizer, _batch(), binding, clip_norm=1.0)
        resumed_scheduler.step()
        assert all(torch.equal(expected_state[key], restarted.state_dict()[key])
                   for key in expected_state)
        assert resumed_optimizer.param_groups[0]["lr"] == expected_lr


def test_mid_epoch_checkpoint_requires_sampler_state(bound_spec, tmp_path):
    spec, binding = bound_spec
    model = _model(spec, binding)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="sampler state"):
        save_checkpoint(tmp_path / "last.pt", binding, model, optimizer, None,
                        epoch=0, batch_offset=1, optimizer_steps=1,
                        sample_presentations=2, sampler_state={})
    sampler = EpochPermutationBatchSampler(10, 2, seed=42, epoch=0)
    with pytest.raises(ValueError, match="Sample presentations"):
        save_checkpoint(tmp_path / "last.pt", binding, model, optimizer, None,
                        epoch=0, batch_offset=1, optimizer_steps=1,
                        sample_presentations=3, sampler_state=sampler.state_for(1))


def test_sampler_cursor_is_processed_position_not_prefetch():
    sampler = EpochPermutationBatchSampler(9, 2, seed=42, epoch=3)
    observed = list(sampler)
    frozen_order = torch.randperm(9, generator=torch.Generator().manual_seed(45))[:8]
    assert [index for batch in observed for index in batch] == frozen_order.tolist()
    assert len(observed) == 4
    assert all(len(batch) == 2 for batch in observed)
    state = sampler.state_for(2)
    assert list(EpochPermutationBatchSampler.from_state(
        state, dataset_size=9, batch_size=2, seed=42, epoch=3,
    )) == observed[2:]
    with pytest.raises(ValueError, match="contract differs"):
        EpochPermutationBatchSampler.from_state(
            state, dataset_size=9, batch_size=2, seed=42, epoch=4,
        )
    with pytest.raises(ValueError, match="row order differs"):
        EpochPermutationBatchSampler.from_state(
            {**state, "order_sha256": "0" * 64},
            dataset_size=9, batch_size=2, seed=42, epoch=3,
        )


def test_training_stats_require_complete_finite_train_role():
    mean, std = training_target_statistics(
        [torch.tensor([5.0, 6.0]), torch.tensor([7.0])],
        expected_rows=3, minimum_std=1e-6,
    )
    assert mean == 6.0
    assert std == 1.0
    with pytest.raises(ValueError, match="row count"):
        training_target_statistics([torch.tensor([5.0])], expected_rows=3,
                                   minimum_std=1e-6)
    with pytest.raises(ValueError, match="finite"):
        training_target_statistics([torch.tensor([float("nan"), 6.0])],
                                   expected_rows=2, minimum_std=1e-6)


def test_development_rejects_denormalization_overflow(bound_spec):
    _, binding = bound_spec

    class LargeFiniteOutput(torch.nn.Module):
        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return torch.full((2, 1), 1e38)

    with pytest.raises(RuntimeError, match="Denormalized"):
        evaluate_development(LargeFiniteOutput(), [_batch()],
                             replace(binding, target_std_eV=1e38),
                             expected_source_idx=[10, 11])
