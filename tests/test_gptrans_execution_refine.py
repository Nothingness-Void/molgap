"""Synthetic CPU equivalence and fail-closed producer checks, not native evidence."""
from copy import deepcopy
from types import SimpleNamespace
import os

import pytest
import torch

from molgap.family_training_core import LoopProgress
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_data import DatasetSelection
from molgap.gptrans_bond_local import BondLocalBlock
from molgap.training_reproducibility import configure_fp32_determinism, capture_rng_state, restore_rng_state
from test_family_training_kernel import initial_states, synthetic_batch
from test_family_training_runner import contract_file, context, PROFILES
from test_family_training_session import make_session, events, assert_equal


_DEVICES = ("cpu", "cuda") if os.environ.get("MOLGAP_EXECUTION_TEST_DEVICE") == "cuda" else ("cpu",)


@pytest.fixture
def _strict_algorithms(monkeypatch):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _synthetic_padding_batch():
    from torch_geometric.data import Batch, Data
    graphs = []
    for index, nodes in enumerate((1, 3, 7, 9)):
        source = torch.arange(nodes - 1, dtype=torch.long)
        target = source + 1
        edges = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
        graphs.append(Data(x=torch.zeros(nodes, 9, dtype=torch.long), edge_index=edges,
            edge_attr=torch.zeros(edges.shape[1], 3, dtype=torch.long),
            y=torch.tensor([float(index)]), source_idx=torch.tensor([index])))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["reference", "pair_prenorm", "centered_logits",
                                  "memory_value", "memory_message", "degree_bond_local_ema999"])
def test_gptrans_safety_fill_preserves_optimizer_gradients_ema_and_rng(initial_states, tmp_path, mode, device):
    from molgap.experiment_family_workflow import tensor_safe_rng_state
    path, digest = initial_states["gptrans_t"]
    if mode == "degree_bond_local_ema999":
        from molgap.gptrans_bond_local import make_bond_local_model
        from molgap.training_reproducibility import atomic_torch_save
        from molgap.v4_runtime import state_dict_sha256
        with torch.random.fork_rng(devices=[]):
            state = make_bond_local_model().state_dict()
            digest = state_dict_sha256(state)
            path = tmp_path / "synthetic-local-initial.pt"
            atomic_torch_save(path, {"model_state": state, "state_sha256": digest})
    recipe = FamilyRecipe("gptrans_t", mode, 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 2,
        .001, .01, ScheduleSpec("cosine_epoch", 2), "final-ema", "torch-pass-randperm", 1,
        target_mean_eV=0., target_std_eV=1., ema_decay=.999,
        torch_version=str(torch.__version__).split("+", 1)[0])
    rng, threads = capture_rng_state(), torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    records = []
    try:
        torch.set_num_threads(1)
        for reference in (True, False):
            configure_fp32_determinism(42)
            torch.utils.deterministic.fill_uninitialized_memory = True
            kernel = FamilyKernel(recipe, path, device=device)
            kernel._qualification_memory_fill = reference
            observations = []
            kernel._qualification_execution_observer = lambda boundary: observations.append((
                boundary, torch.utils.deterministic.fill_uninitialized_memory))
            losses = []
            for _ in range(2):
                losses.append(kernel.step(_synthetic_padding_batch(), check_finite=True))
                assert torch.utils.deterministic.fill_uninitialized_memory is True
                assert torch.are_deterministic_algorithms_enabled()
            assert observations == [(boundary, reference) for boundary in (
                "before_update", "after_update", "before_update", "after_update")]
            records.append(deepcopy({"losses": losses, "model": kernel.model.state_dict(),
                "optimizer": kernel.optimizer.state_dict(), "ema": kernel.ema.state_dict(),
                "gradients": {name: value.grad for name, value in kernel.model.named_parameters()},
                "rng": tensor_safe_rng_state(capture_rng_state())}))
            kernel.close()
        assert_equal(*records)
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.set_num_threads(threads)
        restore_rng_state(rng)


class _EchoBlock(torch.nn.Module):
    def forward(self, node, pair, padding):
        return node, pair


def test_gptrans_fill_scope_restores_policy_on_transfer_failure(initial_states, monkeypatch):
    path, digest = initial_states["gptrans_t"]
    recipe = FamilyRecipe("gptrans_t", "reference", 42, digest,
        DatasetSelection("synthetic.json", "2" * 64, False), 128, 0, 1,
        .001, .01, ScheduleSpec("cosine_epoch", 1), "final-live", "torch-pass-randperm", 1,
        target_mean_eV=0., target_std_eV=1., torch_version=str(torch.__version__).split("+", 1)[0])
    enabled = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        kernel = FamilyKernel(recipe, path, device="cpu")
        def fail(*args, **kwargs):
            assert torch.utils.deterministic.fill_uninitialized_memory is False
            raise RuntimeError("synthetic transfer failure")
        monkeypatch.setattr(kernel, "_gptrans_batch", fail)
        with pytest.raises(RuntimeError, match="synthetic transfer"):
            kernel.step(synthetic_batch(), check_finite=True)
        assert torch.utils.deterministic.fill_uninitialized_memory is True
        assert torch.are_deterministic_algorithms_enabled()
        kernel.close()
    finally:
        torch.use_deterministic_algorithms(enabled)
        torch.utils.deterministic.fill_uninitialized_memory = fill


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("empty", [True, False])
def test_local_in_place_sum_matches_functional_values_and_all_gradients(empty, device, _strict_algorithms):
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
        torch.manual_seed(12)
        actual = BondLocalBlock(_EchoBlock(), channels=8, pair_channels=3).to(device)
        torch.nn.init.normal_(actual.output.weight, std=.1)
        torch.nn.init.normal_(actual.output.bias, std=.1)
        expected = deepcopy(actual)
        node = torch.randn(2, 6, 8, requires_grad=True, device=device)
        pair = torch.randn(2, 3, 6, 6, requires_grad=True, device=device)
        reference_node, reference_pair = node.detach().clone().requires_grad_(), pair.detach().clone().requires_grad_()
        graph = torch.tensor([] if empty else [0, 0, 0, 1, 1], dtype=torch.long, device=device)
        source = torch.tensor([] if empty else [1, 2, 3, 1, 2], dtype=torch.long, device=device)
        target = torch.tensor([] if empty else [2, 2, 2, 2, 3], dtype=torch.long, device=device)
        padding = torch.zeros(2, 1, 1, 6, dtype=torch.bool, device=device)
        output, output_pair = actual(node, pair, padding, (graph, source, target))
        clean = expected.norm(reference_node)
        messages = expected.output(expected.message(torch.cat((clean[graph, source],
            clean[graph, target], reference_pair[graph, :, source, target]), -1)))
        update = torch.index_add(reference_node.new_zeros(12, 8), 0, graph * 6 + target, messages)
        reference_output = reference_node + update.view_as(reference_node)
        (output.square().sum() + output_pair.square().sum()).backward()
        (reference_output.square().sum() + reference_pair.square().sum()).backward()
        assert_equal(output, reference_output)
        assert_equal(node.grad, reference_node.grad)
        assert_equal(pair.grad, reference_pair.grad)
        assert_equal({name: value.grad for name, value in actual.named_parameters()},
                     {name: value.grad for name, value in expected.named_parameters()})


def _output_sink(tmp_path, session):
    from molgap.family_training_outputs import SessionOutputs
    path, record = contract_file(tmp_path, session)
    return SessionOutputs(tmp_path / "output", context(session.recipe.family, path), contract=path,
        adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-refine",
        train_role_identity=record["train_role_identity"])


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_registered_checkpoint_sink_validates_complete_payload_once(tmp_path, monkeypatch, family):
    from molgap import experiment_family_workflow as workflow
    session = make_session(family, development=False, schedule_kind="cosine_step", events=events())
    sink = _output_sink(tmp_path, session)
    session.checkpoint_builder = sink.checkpoint_from_builder
    original, observed = workflow._finite_tree, []
    def finite(state, label):
        observed.append((label, set(state)))
        return original(state, label)
    monkeypatch.setattr(workflow, "_finite_tree", finite)
    progress = session.run(should_pause=lambda: True)
    assert progress.status == "paused"
    assert len(observed) == 1 and observed[0][0] == "resume"
    assert {"model", "optimizer", "ema", "scheduler", "rng_state", "cursor"} <= observed[0][1]
    retained = torch.load(sink.root / "last_checkpoint.pt", map_location="cpu", weights_only=True)
    assert retained["optimizer_step"] == 1


@pytest.mark.parametrize("part", ["model", "optimizer", "ema", "scheduler", "cursor"])
def test_checkpoint_builder_cannot_bypass_nonfinite_state(tmp_path, part):
    from molgap.training_reproducibility import sha256_file
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    sink = _output_sink(tmp_path, session)
    session.checkpoint_builder = sink.checkpoint_from_builder
    progress = session.run(should_pause=lambda: True)
    checkpoint = sink.root / "last_checkpoint.pt"
    digest = sha256_file(checkpoint)
    state = deepcopy(session.state_dict())
    if part in {"model", "ema"}:
        next(iter(state[part].values())).view(-1)[0] = float("nan")
    elif part == "optimizer":
        next(iter(state[part]["state"].values()))["exp_avg"].view(-1)[0] = float("inf")
    else:
        state[part]["unexpected_nonfinite"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite checkpoint"):
        sink.checkpoint_from_builder(progress, lambda: state)
    assert sha256_file(checkpoint) == digest


def test_selection_borrows_one_ema_view_without_changing_state_or_metrics():
    results, counts = [], []
    for reuse in (False, True):
        log = events()
        session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=log)
        original_evaluate, original_use = session.kernel.evaluate, session.kernel.weight_view.use
        swaps = []
        def use(*args, **kwargs):
            swaps.append(1)
            return original_use(*args, **kwargs)
        session.kernel.weight_view.use = use
        if not reuse:
            def previous_selection_view(loaders, *, on_evaluation=None):
                result = original_evaluate(loaders)
                with session.kernel.weight_view.use(session.kernel.model, session.kernel.ema, "ema"):
                    on_evaluation(result)
                return result
            session.kernel.evaluate = previous_selection_view
        session.run()
        results.append((deepcopy(session.state_dict()), log))
        counts.append(len(swaps))
    assert_equal(*results)
    assert counts == [4, 2]


def test_profiler_still_observes_lazy_checkpoint_state_and_publication(tmp_path):
    from molgap.family_execution_profile import FamilyExecutionProfiler, ProfileConfig
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    sink = _output_sink(tmp_path, session)
    session.checkpoint_builder = sink.checkpoint_from_builder
    with FamilyExecutionProfiler(session, tmp_path / "profile", ProfileConfig(warmup_steps=0, active_steps=2)) as profile:
        session.run()
    phases = {row["name"]: row for row in profile.report["phases_by_host_self_time"]}
    assert phases["checkpoint_state"]["calls"] >= 1
    assert phases["checkpoint_publish"]["calls"] >= 1
    assert phases["finite_validation"]["calls"] >= 1
    assert profile.report["status"] == "COMPLETE"


def test_evaluation_observer_failure_restores_live_weights_and_view():
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    before = deepcopy(session.kernel.model.state_dict())
    for value in session.kernel.ema.state.values():
        if value.is_floating_point():
            value.add_(1.)
    def failed(result):
        assert session.kernel.weight_view.busy
        raise OSError("synthetic publication failure")
    try:
        with pytest.raises(OSError, match="synthetic publication"):
            session.kernel.evaluate(session.loaders, on_evaluation=failed)
        assert not session.kernel.weight_view.busy
        assert_equal(before, session.kernel.model.state_dict())
    finally:
        session.close()


def test_legacy_evaluation_adapter_fails_before_any_optimizer_update():
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    session.kernel.evaluate = lambda loaders: None
    before = deepcopy(session.kernel.model.state_dict())
    with pytest.raises(TypeError, match="on_evaluation"):
        session.run()
    assert session.progress.optimizer_step == 0 and session.closed
    assert not session.kernel.optimizer.state
    assert_equal(before, session.kernel.model.state_dict())


def test_evaluation_adapter_cannot_silently_discard_the_selection_observer():
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    original = session.kernel.evaluate
    session.kernel.evaluate = lambda loaders, **kwargs: original(loaders)
    with pytest.raises(RuntimeError, match="observed result"):
        session.run()
    assert session.best is None and session.selected_step is None


def _memory_fill_fixture(tmp_path, monkeypatch):
    from molgap import family_training_session as sessions
    from molgap import family_training_runner as runner
    from molgap.training_reproducibility import atomic_torch_save
    # Explicitly mocked unit metadata: these bytes are never published as evidence.
    state = {"payload": torch.ones(2), "rng_state": {"cuda": [torch.ones(1, dtype=torch.uint8)]}}
    probe = SimpleNamespace(identity="1" * 64, optimizer_steps=2)
    provenance = {"context": {"source_commit": "2" * 40}, "runtime_fingerprint": "3" * 64}
    envelope = {"format": "molgap-gptrans-memory-fill-reference-v1", "context": provenance["context"],
        "recipe_identity": probe.identity, "runtime_fingerprint": provenance["runtime_fingerprint"],
        "execution_role": "fill-enabled-reference", "state": state,
        "observations": [{"boundary": boundary, "memory_fill_enabled": True,
            "deterministic_algorithms_enabled": True, "warn_only": False} for boundary in (
                "before_update", "after_update", "before_update", "after_update")]}
    path = tmp_path / "fill-enabled.pt"
    atomic_torch_save(path, envelope)
    fingerprint = runner._state_fingerprint(state)
    proof = {"path": path.name, "sha256": runner.normalized_file_sha(path),
        "state_fingerprint": fingerprint}
    monkeypatch.setattr(sessions, "validate_session_resume", lambda *args: LoopProgress(2, 256, 0, 2, "complete"))
    return state, envelope, proof, fingerprint, probe, provenance


@pytest.mark.parametrize("alteration", ["missing", "boolean", "flag", "hash", "fingerprint", "extra",
    "partial", "cpu", "copied_repeat", "context", "runtime", "recipe", "role", "observations",
    "fill_disabled", "nondeterministic", "warn_only", "boundary"])
def test_memory_fill_qualification_requires_matching_retained_native_bytes(tmp_path, monkeypatch, alteration):
    from molgap import family_training_session as sessions
    from molgap import family_training_runner as runner
    from molgap.training_reproducibility import atomic_torch_save
    state, envelope, proof, fingerprint, probe, provenance = _memory_fill_fixture(tmp_path, monkeypatch)
    path = tmp_path / proof["path"]
    status = "paused" if alteration == "partial" else "complete"
    monkeypatch.setattr(sessions, "validate_session_resume", lambda *args: LoopProgress(2, 256, 0, 2, status))
    if alteration == "missing":
        proof = None
    elif alteration == "boolean":
        proof = True
    elif alteration == "flag":
        proof["safety_fill_enabled"] = 1
    elif alteration == "hash":
        proof["sha256"] = "0" * 64
    elif alteration == "fingerprint":
        proof["state_fingerprint"] = "0" * 64
    elif alteration == "extra":
        proof["accepted"] = True
    elif alteration == "cpu":
        state["rng_state"]["cuda"] = []
        atomic_torch_save(path, envelope)
        proof["sha256"] = runner.normalized_file_sha(path)
    elif alteration == "copied_repeat":
        atomic_torch_save(path, state)
        proof["sha256"] = runner.normalized_file_sha(path)
    elif alteration in {"context", "runtime", "recipe", "role", "observations", "fill_disabled",
                         "nondeterministic", "warn_only", "boundary"}:
        if alteration == "context":
            envelope["context"] = {"source_commit": "9" * 40}
        elif alteration == "runtime":
            envelope["runtime_fingerprint"] = "9" * 64
        elif alteration == "recipe":
            envelope["recipe_identity"] = "9" * 64
        elif alteration == "role":
            envelope["execution_role"] = "repeat-a"
        elif alteration == "observations":
            envelope["observations"].pop()
        elif alteration == "fill_disabled":
            envelope["observations"][0]["memory_fill_enabled"] = False
        elif alteration == "nondeterministic":
            envelope["observations"][0]["deterministic_algorithms_enabled"] = False
        elif alteration == "warn_only":
            envelope["observations"][0]["warn_only"] = True
        else:
            envelope["observations"][0]["boundary"] = "after_update"
        atomic_torch_save(path, envelope)
        proof["sha256"] = runner.normalized_file_sha(path)
    with pytest.raises(ValueError):
        runner._verify_memory_fill_proof(tmp_path, probe, proof, fingerprint, provenance)


def test_memory_fill_proof_verifies_content_not_only_declared_fingerprint(tmp_path, monkeypatch):
    from molgap import family_training_runner as runner
    from molgap.training_reproducibility import atomic_torch_save
    state, envelope, proof, expected, probe, provenance = _memory_fill_fixture(tmp_path, monkeypatch)
    path = tmp_path / "fill-enabled.pt"
    runner._verify_memory_fill_proof(tmp_path, probe, proof, expected, provenance)
    state["payload"][0] = 2.
    atomic_torch_save(path, envelope)
    proof["sha256"] = runner.normalized_file_sha(path)
    with pytest.raises(ValueError, match="differs"):
        runner._verify_memory_fill_proof(tmp_path, probe, proof, expected, provenance)
