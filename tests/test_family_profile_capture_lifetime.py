"""Synthetic observer lifetimes; no datasets, optimizer updates or native models."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import family_execution_profile as owner
from molgap import ppgt_author, ppgt_author_features, training_reproducibility, modern2d_execution


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Identity()
        self.FFN = torch.nn.Identity()

    def forward(self, value):
        return self.FFN(self.attention(value))


class ToyPPGT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_mlp = torch.nn.Identity()
        self.stem = torch.nn.ModuleList([torch.nn.Identity()])
        self.blocks = torch.nn.ModuleList([ToyBlock()])
        self.head = torch.nn.Identity()

    def graph_forward(self, value):
        value = self.pe_mlp(value)
        for layer in (*self.stem, *self.blocks):
            value = layer(value)
        return self.head(value)

    def forward(self, value):
        return self.graph_forward(value)


def no_op(*args, **kwargs):
    return None


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(ppgt_author, "PPGTAuthor", ToyPPGT)
    model = ToyPPGT()
    recipe = SimpleNamespace(family="ppgt_small", identity="synthetic-profile", batch_size=128,
                             integration_preview=True)
    kernel = SimpleNamespace(model=model, recipe=recipe, device=torch.device("cpu"),
        evaluate=no_op, _model_batch=no_op, step=lambda value: value)
    session = SimpleNamespace(kernel=kernel, recipe=recipe, closed=False,
        progress=SimpleNamespace(optimizer_step=17), loaders=SimpleNamespace(train=lambda: [1, 2]),
        schedule=SimpleNamespace(before_step=no_op, after_step=no_op),
        _flush_metrics=no_op, selected=no_op, pass_finished=no_op, _finish_pass=no_op,
        state_dict=no_op, checkpoint=no_op)

    class FakeProfiler:
        def __init__(self, **kwargs):
            self.stops = 0
            self.fault = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stops += 1
            if self.fault == "stop":
                raise OSError("synthetic stop fault")

        def events(self):
            if self.fault == "events":
                raise OSError("synthetic events fault")
            return []

        def export_chrome_trace(self, path):
            Path(path).write_text("{}", encoding="utf-8")
            if self.fault == "export":
                raise OSError("synthetic export fault")

    monkeypatch.setattr(torch.profiler, "profile", FakeProfiler)
    return session


def hooks(model):
    return [(dict(module._forward_pre_hooks), dict(module._forward_hooks),
             dict(module._forward_hooks_always_called)) for module in model.modules()]


def globals_snapshot():
    return (ppgt_author.validated_topology_views, ppgt_author.positional_inputs,
            ppgt_author.sinusoidal_pe, ppgt_author_features.rrwp,
            torch.autograd.backward, torch.nn.utils.clip_grad_norm_,
            ppgt_author_features.batched_positional_inputs, modern2d_execution.prepare_topology)


def test_capture_detaches_before_next_batch_and_retains_accounting(tmp_path, synthetic):
    session, model = synthetic, synthetic.kernel.model
    # Non-profiler hooks, including ones added during capture, must survive.
    external_pre = model.head.register_forward_pre_hook(no_op)
    external_post = model.head.register_forward_hook(no_op, always_call=True)
    original_hooks, original_globals = hooks(model), globals_snapshot()
    original_step, original_loader = session.kernel.step, session.loaders.train
    original_forward = model.forward.__func__
    with training_reproducibility.observe_finite_validation(lambda label: nullcontext()):
        outer_observer = training_reproducibility._FINITE_VALIDATION_OBSERVER.get()
        with owner._ExecutionCapture(session, tmp_path / "capture",
                owner.ProfileConfig(0, 1, ppgt_phases=True, module_paths=("head",))) as profile:
            assert globals_snapshot()[:4] != original_globals[:4]
            assert globals_snapshot()[4:6] == original_globals[4:6]
            late_hook = model.head.register_forward_hook(no_op)
            profile._begin_batch()
            assert model(7) == 7
            assert not profile._stack
            session.kernel.step(1)
            # A checkpoint after the last captured return still belongs to capture.
            session.checkpoint()
            assert profile._phases["checkpoint_publish"]["calls"] == 1
            profile._begin_batch()
            assert profile._finished and profile._prof.stops == 1
            assert model.forward.__func__ is original_forward
            assert "graph_forward" not in vars(model) and "forward" not in vars(model)
            assert globals_snapshot() == original_globals
            assert session.kernel.step is not original_step
            assert session.loaders.train is not original_loader
            assert "update_observer" not in vars(session.kernel)
            assert session.checkpoint is no_op and session._flush_metrics is no_op
            assert session.schedule.before_step is no_op and session.kernel._model_batch is no_op
            assert training_reproducibility._FINITE_VALIDATION_OBSERVER.get() is outer_observer
            assert late_hook.id in model.head._forward_hooks
            late_hook.remove()
            assert hooks(model) == original_hooks
            phases = deepcopy(profile._phases)
            assert model(8) == 8
            for value in session.loaders.train():
                assert session.kernel.step(value) == value
            assert profile._phases == phases
            assert profile._updates == 3 and profile._captured == 1
            # Keep exclusivity until all observers and global ownership are released.
            with pytest.raises(RuntimeError, match="Only one"):
                with owner._ExecutionCapture(session, tmp_path / "overlap"):
                    pytest.fail("Must retain process ownership after capture")
        assert training_reproducibility._FINITE_VALIDATION_OBSERVER.get() is outer_observer
    assert session.kernel.step is original_step and session.loaders.train is original_loader
    assert not owner._OWNER.locked()
    assert hooks(model) == original_hooks
    report = profile.report
    assert report["observed_updates"] == 3 and report["captured_updates"] == 1
    assert report["starting_optimizer_step"] == 17 and report["status"] == "COMPLETE"
    assert report["runtime"]["device"] == "cpu"
    assert "context_peak_allocated_bytes" in report and "context_peak_reserved_bytes" in report
    span, = report["capture_off_training_spans"]
    assert span["optimizer_updates"] == 2 and span["sample_presentations"] == 256
    assert span["wall_seconds"] >= 0 and span["start_relative_update"] == 1
    saved = json.loads((tmp_path / "capture/training_spans.json").read_text())
    assert saved["spans"] == [span]
    external_pre.remove()
    external_post.remove()


@pytest.mark.parametrize("fault", ["stop", "events", "export"])
def test_capture_failure_detaches_immediately_and_exit_releases_lock(tmp_path, synthetic, fault):
    original_globals, original_hooks = globals_snapshot(), hooks(synthetic.kernel.model)
    original_step = synthetic.kernel.step
    root = tmp_path / fault
    with pytest.raises(OSError, match="synthetic " + fault + " fault"):
        with owner._ExecutionCapture(synthetic, root,
                owner.ProfileConfig(0, 1, ppgt_phases=True)) as profile:
            profile._begin_batch()
            synthetic.kernel.step(1)
            profile._prof.fault = fault
            try:
                profile._begin_batch()
            except OSError:
                assert globals_snapshot() == original_globals
                assert hooks(synthetic.kernel.model) == original_hooks
                assert "update_observer" not in vars(synthetic.kernel)
                assert owner._OWNER.locked()
                raise
    assert profile._prof.stops == 1
    assert profile.report["status"] == "ERROR"
    assert profile.report["trace_sha256"] is None
    assert synthetic.kernel.step is original_step
    assert not owner._OWNER.locked()
    assert not (root / "trace.json.tmp").exists()
    with owner._ExecutionCapture(synthetic, tmp_path / (fault + "-retry")):
        pass


@pytest.mark.parametrize("updates", [0, 1, 2])
def test_exit_cleans_capture_hooks_without_next_batch(tmp_path, synthetic, updates):
    original_globals, original_hooks = globals_snapshot(), hooks(synthetic.kernel.model)
    with owner._ExecutionCapture(synthetic, tmp_path / "exit",
            owner.ProfileConfig(1, 1, ppgt_phases=True)) as profile:
        for _ in range(updates):
            profile._begin_batch()
            synthetic.kernel.step(1)
    assert profile.report["observed_updates"] == updates
    assert profile.report["captured_updates"] == max(0, updates - 1)
    assert globals_snapshot() == original_globals
    assert hooks(synthetic.kernel.model) == original_hooks
    assert "update_observer" not in vars(synthetic.kernel)
    assert not owner._OWNER.locked()


def test_partial_install_failure_cleans_both_lifetimes(tmp_path, synthetic, monkeypatch):
    original_globals, original_hooks = globals_snapshot(), hooks(synthetic.kernel.model)
    original_step, original_forward = synthetic.kernel.step, synthetic.kernel.model.forward.__func__

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic hook installation fault")

    monkeypatch.setattr(synthetic.kernel.model.head, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="hook installation fault"):
        with owner._ExecutionCapture(synthetic, tmp_path / "install",
                owner.ProfileConfig(ppgt_phases=True)):
            pytest.fail("Must reject during installation")
    assert globals_snapshot() == original_globals
    assert hooks(synthetic.kernel.model) == original_hooks
    assert synthetic.kernel.model.forward.__func__ is original_forward
    assert synthetic.kernel.step is original_step
    assert "update_observer" not in vars(synthetic.kernel)
    assert not owner._OWNER.locked()


def test_other_thread_and_failed_prehook_leave_phase_stack_balanced(tmp_path, synthetic):
    model = synthetic.kernel.model

    def fail(*args):
        raise RuntimeError("synthetic application prehook fault")

    application_hook = model.head.register_forward_pre_hook(fail)
    original_hooks = hooks(model)
    try:
        with owner._ExecutionCapture(synthetic, tmp_path / "threads",
                owner.ProfileConfig(0, 1, ppgt_phases=True)) as profile:
            profile._begin_batch()
            with pytest.raises(RuntimeError, match="application prehook fault"):
                model(1)
            assert not profile._stack
            application_hook.remove()
            phases = deepcopy(profile._phases)
            # Global wraps are shared, but attribution stays on the owning thread.
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(model, 2).result() == 2
            assert profile._phases == phases and not profile._stack
            synthetic.kernel.step(1)
            profile._begin_batch()
            assert all(not pre and not post and not always for pre, post, always in hooks(model))
    finally:
        application_hook.remove()
    assert any(pre for pre, _, _ in original_hooks)
    assert not owner._OWNER.locked()


def test_builder_and_explicit_empty_observer_are_restored_at_capture_end(tmp_path, synthetic):
    synthetic.kernel.update_observer = None
    synthetic.checkpoint_builder = no_op
    synthetic._checkpoint_state = no_op
    with owner._ExecutionCapture(synthetic, tmp_path / "builder", owner.ProfileConfig(0, 1)) as profile:
        assert synthetic.checkpoint is no_op
        assert synthetic.checkpoint_builder is not no_op
        assert synthetic._checkpoint_state is not no_op
        profile._begin_batch()
        synthetic.kernel.step(1)
        profile._begin_batch()
        assert synthetic.kernel.update_observer is None
        assert synthetic.checkpoint_builder is no_op and synthetic._checkpoint_state is no_op


def test_legacy_default_wrapper_lifetime_is_unchanged(tmp_path, synthetic):
    target = SimpleNamespace(call=no_op)

    class LegacyCapture(owner._ExecutionCapture):
        def _install(self):
            self._wrap(target, "call", "legacy_call")

    with LegacyCapture(synthetic, tmp_path / "legacy", owner.ProfileConfig(0, 1)) as profile:
        profile._begin_batch()
        profile._after_update()
        profile._begin_batch()
        assert profile._finished and target.call is not no_op
    assert target.call is no_op
