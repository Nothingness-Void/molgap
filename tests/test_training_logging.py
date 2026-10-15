"""Synthetic CPU regressions for shared-family telemetry and worker relays."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molgap.experiment_execution import build_family_recipe
from molgap.family_training_core import LoopProgress
from molgap.family_training_recipe import ScheduleSpec
from molgap.family_training_runner import INTERFACES, _bootstrap, read_recipe
from molgap.training_logging import (
    LOG_FORMAT,
    LOG_PREFIX,
    TrainingLogPolicy,
    TrainingLogger,
    forward_training_logs,
)
from molgap.training_reproducibility import capture_rng_state
from test_family_training_session import events, make_session


class FakeClock:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def family_record(recipe, *, logging=None):
    return build_family_recipe(
        INTERFACES[recipe.family],
        configuration=recipe,
        train_role_identity="3" * 64,
        development_role_identity="4" * 64 if recipe.dataset.development else None,
        source_idx_sha256="5" * 64 if recipe.dataset.development else None,
        target_sha256="6" * 64 if recipe.dataset.development else None,
        logging=logging,
    )


def make_logger(root, recipe, *, policy=None, observations=(), clock=None, emitted=None, arm_id="arm-a"):
    emitted = [] if emitted is None else emitted
    logger = TrainingLogger(
        root,
        SimpleNamespace(arm_id=arm_id),
        recipe,
        TrainingLogPolicy() if policy is None else policy,
        observations=observations,
        clock=clock,
        emit=emitted.append,
    )
    return logger, emitted


def wire_logger(session, logger):
    checkpoint = session.checkpoint
    pass_finished = session.pass_finished

    def publish_checkpoint(progress, state):
        checkpoint(progress, state)
        logger.checkpoint_finished(progress)

    def report_pass(progress, metrics):
        pass_finished(progress, metrics)
        logger.pass_finished(progress, metrics)

    session.checkpoint = publish_checkpoint
    session.pass_finished = report_pass
    session.batch_finished = logger.progress
    logger.start(session.progress)


def log_records(lines):
    assert all(line.startswith(LOG_PREFIX) for line in lines)
    return [json.loads(line[len(LOG_PREFIX):]) for line in lines]


def assert_rng_equal(left, right):
    assert left["python"] == right["python"]
    assert left["numpy"][0] == right["numpy"][0]
    assert np.array_equal(left["numpy"][1], right["numpy"][1])
    assert left["numpy"][2:] == right["numpy"][2:]
    assert torch.equal(left["torch"], right["torch"])
    assert len(left["cuda"]) == len(right["cuda"])
    for before, after in zip(left["cuda"], right["cuda"]):
        assert torch.equal(before, after)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for before, after in zip(left, right):
            assert_tree_equal(before, after)
    else:
        assert left == right


def test_logging_builder_defaults_to_epoch_policy_and_stable_record_schema(tmp_path):
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    try:
        record = family_record(session.recipe)
        assert record["logging"] == {"mode": "epoch", "interval": None}
        assert TrainingLogPolicy.from_dict(record["logging"]).to_dict() == record["logging"]
        legacy_record = dict(record)
        legacy_record.pop("logging")
        legacy_path = tmp_path / "legacy-contract.json"
        legacy_path.write_text(json.dumps(legacy_record), encoding="utf-8")
        _, legacy_recipe = read_recipe(legacy_path)
        assert legacy_recipe.identity == session.recipe.identity

        # Exercise a checkpoint at every batch; the default still reports complete
        # passes plus the terminal partial pass, rather than checkpoint cadence.
        recipe = replace(session.recipe, checkpoint_every=1)
        session.recipe = session.kernel.recipe = session.loaders.recipe = recipe
        logger, lines = make_logger(tmp_path, recipe)
        checkpoints = []
        checkpoint = session.checkpoint

        def count_checkpoint(progress, state):
            checkpoints.append(progress.optimizer_step)
            checkpoint(progress, state)
            logger.checkpoint_finished(progress)

        pass_finished = session.pass_finished

        def report_pass(progress, metrics):
            pass_finished(progress, metrics)
            logger.pass_finished(progress, metrics)

        session.checkpoint = count_checkpoint
        session.pass_finished = report_pass
        session.batch_finished = logger.progress
        logger.start(session.progress)
        progress = session.run()

        assert progress.status == "complete"
        assert checkpoints == [1, 2, 3, 4, 5]
        observed = log_records(lines)
        assert [(row["optimizer_steps"], row["event"]) for row in observed] == [
            (2, "epoch"), (4, "epoch"), (5, "terminal")
        ]
        expected_fields = {
            "format", "event", "arm_id", "family", "weight_state", "status",
            "completed_passes", "planned_passes", "next_batch", "optimizer_steps",
            "planned_optimizer_steps", "sample_presentations", "train_live_mae_eV",
            "development_mae_eV", "best_development_mae_eV", "metric_pass",
            "best_metric_pass", "learning_rates", "elapsed_seconds",
            "updates_per_second", "eta_seconds", "pass_wall_seconds",
            "pass_timing_scope", "checkpoint_optimizer_steps",
        }
        assert all(set(row) == expected_fields for row in observed)
        assert all(row["format"] == LOG_FORMAT for row in observed)
        assert json.loads((tmp_path / "training_progress.json").read_text(encoding="utf-8")) == observed[-1]
    finally:
        session.close()


@pytest.mark.parametrize("policy", [
    TrainingLogPolicy(),
    TrainingLogPolicy("batches", 3),
    TrainingLogPolicy("seconds", 2.5),
])
def test_logging_policy_accepts_only_exact_supported_shapes(policy):
    assert TrainingLogPolicy.from_dict(policy.to_dict()) == policy


@pytest.mark.parametrize("value", [
    {"mode": "epoch", "interval": 1},
    {"mode": "batches", "interval": True},
    {"mode": "batches", "interval": 1.5},
    {"mode": "batches", "interval": 0},
    {"mode": "seconds", "interval": True},
    {"mode": "seconds", "interval": 0},
    {"mode": "seconds", "interval": float("nan")},
    {"mode": "seconds", "interval": float("inf")},
    {"mode": "seconds", "interval": "2"},
    {"mode": "unknown", "interval": 1},
    {"mode": "epoch"},
    {"mode": "epoch", "interval": None, "extra": False},
    ["epoch", None],
])
def test_explicit_logging_policy_rejects_malformed_values(value):
    with pytest.raises((TypeError, ValueError)):
        TrainingLogPolicy.from_dict(value)


@pytest.mark.parametrize("logging", [
    {"mode": "batches", "interval": True},
    None,
])
def test_malformed_frozen_policy_blocks_recipe_read_and_runner_before_workers(tmp_path, monkeypatch, logging):
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    try:
        record = family_record(session.recipe)
    finally:
        session.close()
    record["logging"] = logging
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="logging|Logging"):
        read_recipe(path)

    # Bootstrap must reject the contract before validation can proceed to data,
    # frozen weights, accelerator discovery, or construction of a worker session.
    import molgap.family_training_runner as runner
    monkeypatch.setattr(runner, "validate_screen_recipe",
        lambda *args, **kwargs: pytest.fail("malformed logging reached post-read preparation"))
    with pytest.raises(ValueError, match="logging|Logging"):
        _bootstrap(
            spec=None,
            package_dir=tmp_path,
            expected_package_identity="unused",
            arm_id="arm-a",
            mode="reference",
            recipe_path=path,
            initial_state_path=tmp_path / "unused.pt",
            input_root=tmp_path,
            account="synthetic",
            run_reference="synthetic",
            trajectory_id="synthetic",
        )


def test_progress_metrics_convert_normalized_train_mae_and_keep_best_pass(tmp_path):
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    try:
        recipe = session.recipe
        logger, lines = make_logger(tmp_path, recipe, policy=TrainingLogPolicy("batches", 1))
        logger.start(LoopProgress(0, 0, 0, 0, "running"))

        before_first_pass = LoopProgress(1, 128, 0, 1, "running")
        logger.progress(before_first_pass, learning_rates=[0.001])
        initial = log_records(lines)[0]
        assert initial["event"] == "progress"
        assert initial["train_live_mae_eV"] is None
        assert initial["development_mae_eV"] is None
        assert initial["best_development_mae_eV"] is None
        assert initial["metric_pass"] is None

        first_pass = LoopProgress(2, 256, 1, 0, "running")
        logger.pass_finished(first_pass, {
            "normalized_train_gap_mae": 0.25,
            "development": {"mae_eV": 0.6},
        })
        logger.progress(first_pass, learning_rates=[0.0009])
        current = log_records(lines)[-1]
        assert current["event"] == "epoch"
        assert current["train_live_mae_eV"] == pytest.approx(0.5)
        assert current["development_mae_eV"] == pytest.approx(0.6)
        assert current["best_development_mae_eV"] == pytest.approx(0.6)
        assert current["metric_pass"] == current["best_metric_pass"] == 1

        periodic = LoopProgress(3, 384, 1, 1, "running")
        logger.progress(periodic, learning_rates=[0.0008])
        retained = log_records(lines)[-1]
        assert retained["event"] == "progress"
        assert retained["train_live_mae_eV"] == pytest.approx(0.5)
        assert retained["development_mae_eV"] == pytest.approx(0.6)
        assert retained["metric_pass"] == 1

        second_pass = LoopProgress(4, 512, 2, 0, "complete")
        logger.pass_finished(second_pass, {
            "normalized_train_gap_mae": 0.4,
            "development": {"mae_eV": 0.8},
        })
        logger.progress(second_pass, learning_rates=[0.0007])
        terminal_epoch = log_records(lines)[-1]
        assert terminal_epoch["event"] == "epoch"
        assert terminal_epoch["train_live_mae_eV"] == pytest.approx(0.8)
        assert terminal_epoch["development_mae_eV"] == pytest.approx(0.8)
        assert terminal_epoch["best_development_mae_eV"] == pytest.approx(0.6)
        assert terminal_epoch["metric_pass"] == 2
        assert terminal_epoch["best_metric_pass"] == 1
    finally:
        session.close()


def test_seconds_policy_emits_only_after_elapsed_interval(tmp_path):
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    try:
        clock = FakeClock(12.0)
        logger, lines = make_logger(
            tmp_path, session.recipe, policy=TrainingLogPolicy("seconds", 2.5), clock=clock
        )
        logger.start(LoopProgress(0, 0, 0, 0, "running"))
        clock.advance(2.49)
        logger.progress(LoopProgress(1, 128, 0, 1, "running"), learning_rates=[0.001])
        assert lines == []
        clock.advance(0.01)
        logger.progress(LoopProgress(2, 256, 1, 0, "running"), learning_rates=[0.0009])
        assert len(lines) == 1
        assert log_records(lines)[0]["event"] == "progress"
    finally:
        session.close()


def test_resumed_trace_restores_best_and_uses_invocation_timing(tmp_path):
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    try:
        recipe = replace(session.recipe, optimizer_steps=6, schedule=ScheduleSpec("cosine_step", 6))
        observations = [
            {"event": "observation", "epoch_or_pass": 1, "live_train_metric": 0.3,
             "ema_dev_metric": 0.2},
            {"event": "observation", "epoch_or_pass": 2, "live_train_metric": 0.5,
             "ema_dev_metric": 0.4},
        ]
        clock = FakeClock(100.0)
        logger, lines = make_logger(
            tmp_path, recipe, policy=TrainingLogPolicy("seconds", 2), observations=observations, clock=clock
        )
        resumed = LoopProgress(5, 640, 2, 1, "running")
        clock.advance(500.0)
        logger.start(resumed)
        clock.advance(2.0)
        logger.progress(resumed, learning_rates=[0.0005])

        restored = log_records(lines)[0]
        assert restored["metric_pass"] == 2
        assert restored["train_live_mae_eV"] == pytest.approx(1.0)
        assert restored["development_mae_eV"] == pytest.approx(0.4)
        assert restored["best_development_mae_eV"] == pytest.approx(0.2)
        assert restored["best_metric_pass"] == 1
        assert restored["elapsed_seconds"] == pytest.approx(2.0)

        final_pass = LoopProgress(6, 768, 3, 0, "complete")
        logger.pass_finished(final_pass, {
            "normalized_train_gap_mae": 0.2,
            "development": {"mae_eV": 0.1},
        })
        clock.advance(3.0)
        logger.progress(final_pass, learning_rates=[0.0004])
        invocation = log_records(lines)[-1]
        assert invocation["event"] == "epoch"
        assert invocation["pass_timing_scope"] == "invocation_partial_pass"
        assert invocation["elapsed_seconds"] == pytest.approx(5.0)
        assert invocation["pass_wall_seconds"] == pytest.approx(5.0)
        assert invocation["best_development_mae_eV"] == pytest.approx(0.1)
        assert invocation["best_metric_pass"] == 3
    finally:
        session.close()


@pytest.mark.parametrize(("selection", "clean_signal", "expected"), [
    ("final-live", None, "live"),
    ("final-ema", None, "ema"),
    ("final-ema", {"calibrated": True}, "clean_ema"),
])
def test_weight_state_labels_live_ema_and_forward_compatible_clean_ema(
        tmp_path, selection, clean_signal, expected):
    session = make_session("gptrans_t", development=False, schedule_kind="cosine_step", events=events())
    try:
        attributes = {**vars(session.recipe), "selection": selection}
        attributes["loop_plan"] = session.recipe.loop_plan
        if clean_signal is not None:
            # The frozen base FamilyRecipe has no Clean-BN field; this proxy
            # exercises the logger's forward-compatible reviewed-clean signal.
            attributes["clean_bn"] = clean_signal
        recipe = SimpleNamespace(**attributes)
        logger, lines = make_logger(tmp_path / expected, recipe)
        logger.start(LoopProgress(0, 0, 0, 0, "running"))
        logger.progress(LoopProgress(1, 128, 0, 1, "paused"), learning_rates=[0.001])
        assert log_records(lines)[0]["weight_state"] == expected
    finally:
        session.close()


def test_epoch_periodic_terminal_and_checkpoint_triggers_emit_once_per_step(tmp_path):
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    try:
        logger, lines = make_logger(tmp_path, session.recipe, policy=TrainingLogPolicy("batches", 2))
        logger.start(LoopProgress(0, 0, 0, 0, "running"))
        epoch = LoopProgress(2, 256, 1, 0, "complete")
        logger.pass_finished(epoch, {
            "normalized_train_gap_mae": 0.25,
            "development": {"mae_eV": 0.6},
        })
        logger.checkpoint_finished(epoch)
        logger.progress(epoch, learning_rates=[0.0009])
        logger.progress(epoch, learning_rates=[0.0009])
        assert len(lines) == 1
        row = log_records(lines)[0]
        assert row["event"] == "epoch"
        assert row["optimizer_steps"] == row["checkpoint_optimizer_steps"] == 2
    finally:
        session.close()


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
def test_logging_matches_training_state_rng_and_evaluation_count(tmp_path, family):
    baseline = make_session(family, development=True, schedule_kind="cosine_step", events=events())
    baseline_calls = [0]
    baseline_evaluate = baseline.kernel.evaluate

    def count_baseline(loaders, **kwargs):
        baseline_calls[0] += 1
        return baseline_evaluate(loaders, **kwargs)

    baseline.kernel.evaluate = count_baseline
    try:
        baseline.run()
        expected_state = baseline.state_dict()
        expected_rng = capture_rng_state()
    finally:
        baseline.close()

    observed = make_session(family, development=True, schedule_kind="cosine_step", events=events())
    observed_calls = [0]
    observed_evaluate = observed.kernel.evaluate

    def count_observed(loaders, **kwargs):
        observed_calls[0] += 1
        return observed_evaluate(loaders, **kwargs)

    observed.kernel.evaluate = count_observed
    try:
        logger, lines = make_logger(
            tmp_path, observed.recipe, policy=TrainingLogPolicy("batches", 1)
        )
        wire_logger(observed, logger)
        observed.run()
        actual_state = observed.state_dict()
        actual_rng = capture_rng_state()

        assert_tree_equal(expected_state, actual_state)
        assert_rng_equal(expected_rng, actual_rng)
        assert baseline_calls[0] == observed_calls[0] == 2
        assert len(lines) == observed.recipe.optimizer_steps
    finally:
        observed.close()


def test_checkpoint_failure_does_not_publish_successful_epoch_log(tmp_path):
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    try:
        logger, lines = make_logger(tmp_path, session.recipe)
        wire_logger(session, logger)

        def failed_checkpoint(progress, state):
            if progress.optimizer_step == 2:
                raise OSError("durable checkpoint unavailable")

        session.checkpoint = failed_checkpoint
        with pytest.raises(OSError, match="durable checkpoint"):
            session.run()
        assert session.progress.status == "persistence_failed"
        assert lines == []
        assert not (tmp_path / "training_progress.json").exists()
    finally:
        session.close()


def test_worker_line_relay_is_arm_scoped_deduplicated_and_retries_partial_lines(tmp_path):
    emitted = {"arm-a": [], "arm-b": []}
    offsets = {}
    paths = {}
    final_partial = {}
    for arm_id in ("arm-a", "arm-b"):
        first = {"format": LOG_FORMAT, "event": "progress", "arm_id": arm_id, "optimizer_steps": 1}
        last = {"format": LOG_FORMAT, "event": "terminal", "arm_id": arm_id, "optimizer_steps": 2}
        first_line = LOG_PREFIX + json.dumps(first, sort_keys=True, separators=(",", ":"))
        last_line = LOG_PREFIX + json.dumps(last, sort_keys=True, separators=(",", ":"))
        path = tmp_path / f"{arm_id}.log"
        path.write_bytes(b"ordinary worker diagnostic\n" + first_line.encode("utf-8") + b"\n" +
                         last_line.encode("utf-8"))
        paths[arm_id], final_partial[arm_id] = path, last_line
        offsets[arm_id] = forward_training_logs(
            path, 0, arm_id=arm_id, emit=emitted[arm_id].append
        )
        assert emitted[arm_id] == [first_line]

    for arm_id in ("arm-a", "arm-b"):
        with paths[arm_id].open("ab") as stream:
            stream.write(b"\n")
        offsets[arm_id] = forward_training_logs(
            paths[arm_id], offsets[arm_id], arm_id=arm_id, emit=emitted[arm_id].append
        )
        offsets[arm_id] = forward_training_logs(
            paths[arm_id], offsets[arm_id], arm_id=arm_id, emit=emitted[arm_id].append
        )
        assert emitted[arm_id] == [emitted[arm_id][0], final_partial[arm_id]]
        assert [json.loads(line[len(LOG_PREFIX):])["arm_id"] for line in emitted[arm_id]] == [arm_id, arm_id]
