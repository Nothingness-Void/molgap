"""Focused synthetic checks for the registered K1 hop-shell addon."""
from copy import deepcopy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.family_training_data import DatasetSelection
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_policy import recipe_policy
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.k1_execution_backend import K1ExecutionConfig
from molgap.experiment_family_workflow import tensor_digest
from molgap.training_reproducibility import (
    atomic_torch_save, capture_rng_state, restore_rng_state, sha256_file,
)
from molgap.v4_runtime import state_dict_sha256
from test_family_training_kernel import synthetic_batch


INITIALIZATION = Path("D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt")


@pytest.fixture(scope="module")
def frozen_k1_initialization():
    if not INITIALIZATION.is_file():
        pytest.skip("The retained frozen K1 initialization artifact is unavailable")
    payload = torch.load(INITIALIZATION, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload)
    digest = payload.get("state_sha256", state_dict_sha256(state))
    assert state_dict_sha256(state) == digest
    return INITIALIZATION, digest


@pytest.fixture
def deterministic_cpu():
    rng = capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    threads = torch.get_num_threads()
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = False
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.set_num_threads(threads)
        restore_rng_state(rng)


def _graph(node_count, undirected_edges, *, source_idx=None):
    directed = [(a, b) for a, b in undirected_edges]
    directed += [(b, a) for a, b in undirected_edges]
    edge_index = (torch.tensor(directed, dtype=torch.long).t().contiguous()
                  if directed else torch.empty((2, 0), dtype=torch.long))
    values = {"x": torch.zeros(node_count, 9, dtype=torch.long), "edge_index": edge_index}
    if source_idx is not None:
        values.update(edge_attr=torch.zeros(edge_index.shape[1], 3, dtype=torch.long),
            random_walk_pe=torch.zeros(node_count, 16), y=torch.tensor([float(source_idx % 5)]),
            source_idx=torch.tensor([source_idx], dtype=torch.long))
    return Data(**values)


def _distance_pairs(node_count, undirected_edges, distance):
    adjacency = [set() for _ in range(node_count)]
    for left, right in undirected_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    pairs = set()
    for target in range(node_count):
        distances = {target: 0}
        frontier = [target]
        while frontier:
            current = frontier.pop(0)
            if distances[current] == distance:
                continue
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(neighbor)
        pairs.update((target, source) for source, value in distances.items() if value == distance)
    return pairs


def _recipe(initialization_sha256, ordered):
    from molgap.experiment_execution import build_addon_declaration

    config = {"ordered": ordered, "hops": 4, "width": 64,
        "layer_policy": "3-6-9", "seed": 42}
    addon = build_addon_declaration(("neural_atom_k1", "3"), "k1_hop_shell",
        repo_root=Path(__file__).resolve().parents[1], config=config)
    return FamilyRecipe("neural_atom_k1", "composed", 42, initialization_sha256,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 2,
        4e-4, 1e-5, ScheduleSpec("cosine_step", 2), "final-live",
        "torch-pass-randperm", 1, target_mean_eV=0., target_std_eV=1.,
        k1_execution=K1ExecutionConfig(), addons=(addon,),
        torch_version=str(torch.__version__).split("+", 1)[0])


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def _assert_rng_equal(actual, expected):
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    assert np.array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    assert torch.equal(actual["torch"], expected["torch"])
    assert len(actual["cuda"]) == len(expected["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(actual["cuda"], expected["cuda"]))


def test_exact_shortest_shells_permutation_and_total_content_invariant():
    from molgap.k1_hop_shell import shells_from_cpu_batch

    chain = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    cycle = chain + [(5, 0)]
    disconnected = [(0, 1), (1, 2), (3, 4)]
    permutation = [3, 1, 5, 0, 4, 2]
    permuted_chain = [(permutation[left], permutation[right]) for left, right in chain]
    cases = [(6, chain), (6, cycle), (6, disconnected), (6, permuted_chain)]
    batch = Batch.from_data_list([_graph(nodes, edges) for nodes, edges in cases])
    topology = shells_from_cpu_batch(batch, "cpu")

    assert len(topology.shells) == 4
    offsets = [0]
    for node_count, _ in cases:
        offsets.append(offsets[-1] + node_count)
    for hop, (targets, sources) in enumerate(topology.shells, start=1):
        observed = set(zip(targets.tolist(), sources.tolist()))
        expected = set()
        for graph_index, (node_count, edges) in enumerate(cases):
            offset = offsets[graph_index]
            expected |= {(target + offset, source + offset)
                for target, source in _distance_pairs(node_count, edges, hop)}
        assert observed == expected

    for hop, (targets, sources) in enumerate(topology.shells, start=1):
        observed = set(zip(targets.tolist(), sources.tolist()))
        expected_permutation = {(offsets[3] + permutation[target], offsets[3] + permutation[source])
            for target, source in _distance_pairs(6, chain, hop)}
        assert expected_permutation <= observed

    projected = torch.full((topology.nodes, 1), 12., dtype=torch.float64)
    denominator = topology.union_count.to(torch.float64).unsqueeze(-1)
    content = []
    for targets, sources in topology.shells:
        value = torch.zeros_like(projected)
        value.index_add_(0, targets, projected.index_select(0, sources))
        content.append(value / denominator)
    ordered_content = torch.stack(content)
    equal_mixture = (ordered_content.sum(dim=0, keepdim=True) / 4).expand_as(ordered_content)
    assert torch.equal(ordered_content.sum(dim=0), equal_mixture.sum(dim=0))
    expected_total = torch.full((topology.nodes,), 12., dtype=torch.float64)
    expected_total[offsets[2] + 5] = 0.  # Isolated vertex in the disconnected graph.
    assert torch.equal(ordered_content.sum(dim=0).squeeze(-1), expected_total)


def test_both_addon_branches_run_optimizer_and_restore_next_update(
        tmp_path, frozen_k1_initialization, deterministic_cpu):
    initialization_path, digest = frozen_k1_initialization
    batch = synthetic_batch()

    for ordered in (0, 1):
        recipe = _recipe(digest, ordered)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4200 + ordered)
            kernel = FamilyKernel(recipe, initialization_path, device="cpu")
        kernel.model.train()
        calls = []
        handles = [module.register_forward_hook(
            lambda module, args, output, name=name: calls.append(name))
            for name, module in kernel.model._molgap_addons.items()]
        try:
            first = kernel.step(batch, check_finite=True)
            assert torch.isfinite(first[0]) and torch.isfinite(first[1]) and first[2] == 2
            assert calls == ["hop_shell_3", "hop_shell_6", "hop_shell_9"]
            for module in kernel.model._molgap_addons.values():
                assert module._batch is None
                for parameter in module.parameters():
                    assert parameter.grad is not None
                    assert parameter in kernel.optimizer.state

            handles[0].remove()
            handles[1].remove()
            handles[2].remove()
            kernel.model.eval()
            with torch.no_grad():
                prediction = kernel.forward(batch)
            assert prediction.shape == (2,)
            assert not kernel.model.training
            assert all(module._batch is None for module in kernel.model._molgap_addons.values())
            kernel.model.train()

            checkpoint = {"model": {key: value.detach().clone()
                    for key, value in kernel.model.state_dict().items()},
                "optimizer": deepcopy(kernel.optimizer.state_dict()),
                "rng": capture_rng_state()}
            path = tmp_path / f"hop-shell-{ordered}.pt"
            atomic_torch_save(path, checkpoint)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            expected_update = kernel.step(batch, check_finite=True)
            expected_model = {key: value.detach().clone()
                for key, value in kernel.model.state_dict().items()}
            expected_optimizer = deepcopy(kernel.optimizer.state_dict())
            expected_rng = capture_rng_state()

            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(9000 + ordered)
                resumed = FamilyKernel(recipe, initialization_path, device="cpu")
            resumed.model.load_state_dict(checkpoint["model"], strict=True)
            resumed.optimizer.load_state_dict(checkpoint["optimizer"])
            resumed.model.train()
            restore_rng_state(checkpoint["rng"])
            actual_update = resumed.step(batch, check_finite=True)
            assert all(torch.equal(a, b) if torch.is_tensor(a) else a == b
                for a, b in zip(expected_update, actual_update))
            _assert_nested_equal(resumed.model.state_dict(), expected_model)
            _assert_nested_equal(resumed.optimizer.state_dict(), expected_optimizer)
            _assert_rng_equal(capture_rng_state(), expected_rng)
            assert all(module._batch is None for module in resumed.model._molgap_addons.values())
            resumed.close()
        finally:
            for handle in handles:
                handle.remove()
            kernel.close()


def test_composed_policy_and_fixed_train_probe_preserve_live_state(
        tmp_path, monkeypatch, frozen_k1_initialization, deterministic_cpu):
    from molgap import k1_hop_shell_observation as observation

    initialization_path, digest = frozen_k1_initialization
    recipe = _recipe(digest, ordered=1)
    policy = recipe_policy(recipe)
    assert callable(policy.qualify_resources)
    assert callable(policy.refresh_resources)
    assert callable(policy.observe_pass)
    resource = {"status": "accepted", "maximum_overhead_fraction": observation.MAXIMUM_OVERHEAD,
        "synchronized_step_overhead_fraction": .1}
    assert policy.refresh_resources(resource, 10) == {**resource, "resume_pass_index": 10}

    cohort_rows = 256
    monkeypatch.setattr(observation, "TRAIN_ROWS", cohort_rows)
    template = synthetic_batch().to_data_list()[0]
    train = []
    for source_idx in range(cohort_rows):
        graph = template.clone()
        graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
        graph.y = torch.tensor([float(source_idx % 5)])
        train.append(graph)
    loaders = SimpleNamespace(data=SimpleNamespace(train=train, train_rows=cohort_rows,
        train_source_start=0))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        kernel = FamilyKernel(recipe, initialization_path, device="cpu")
    kernel.model.train()
    state_before = {key: value.detach().clone() for key, value in kernel.model.state_dict().items()}
    buffers_before = {name: value.detach().clone() for name, value in kernel.model.named_buffers()}
    modes_before = {name: module.training for name, module in kernel.model.named_modules()}
    optimizer_before = deepcopy(kernel.optimizer.state_dict())
    rng_before = capture_rng_state()
    progress = SimpleNamespace(pass_index=10, optimizer_step=1280, sample_presentations=163840)
    try:
        policy.observe_pass(kernel, loaders, tmp_path, progress)
        report_path = tmp_path / "train_cohort" / "pass_10.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["role"] == "TRAIN" and report["weights"] == "live"
        assert report["metric"] == "MAE" and report["unit"] == "eV" and report["target"] == "Gap"
        assert report["model_mode"] == "eval" and report["rows"] == cohort_rows
        assert report["selection_used"] is False and report["formal_training_exposure"] == 0
        assert report["live_state_sha256"] == state_dict_sha256(state_before)
        assert report["source_idx_sha256"] == tensor_digest(
            torch.arange(cohort_rows, dtype=torch.long), role="source_idx")
        targets = torch.tensor([float(index % 5) for index in range(cohort_rows)])
        assert report["target_eV_sha256"] == tensor_digest(targets, role="target")
        assert len(report["prediction_eV_sha256"]) == 64
        assert np.isfinite(report["mae_eV"]) and report["mae_eV"] >= 0
        assert report["optimizer_step"] == progress.optimizer_step
        manifest = json.loads((report_path.parent / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["files"][report_path.name]["sha256"] == sha256_file(report_path)
        assert manifest["files"][report_path.name]["bytes"] == report_path.stat().st_size
        _assert_nested_equal(kernel.model.state_dict(), state_before)
        _assert_nested_equal(dict(kernel.model.named_buffers()), buffers_before)
        assert {name: module.training for name, module in kernel.model.named_modules()} == modes_before
        _assert_nested_equal(kernel.optimizer.state_dict(), optimizer_before)
        _assert_rng_equal(capture_rng_state(), rng_before)
        assert all(module._batch is None for module in kernel.model._molgap_addons.values())
    finally:
        kernel.close()
