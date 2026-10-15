"""Server adapters of desktop telemetry/refinement; elementary CPU fixtures only."""
import json
from types import SimpleNamespace

import pytest

from molgap.family_training_core import LoopProgress
from molgap.training_logging import LOG_PREFIX, TrainingLogPolicy, TrainingLogger
from test_family_server_recovery import MODES, prepared
from test_family_training_session import events, make_session


@pytest.mark.parametrize("unit,stored", [("eV", .5), ("normalized-Gap", .25)])
def test_retained_trace_unit_is_not_guessed_or_scaled_twice(tmp_path, unit, stored):
    session = make_session("gptrans_t", development=True, schedule_kind="cosine_step", events=events())
    try:
        lines = []
        logger = TrainingLogger(tmp_path, SimpleNamespace(arm_id="reference"), session.recipe,
            TrainingLogPolicy("batches", 1), emit=lines.append,
            observation_train_unit=unit,
            observations=[{"event": "observation", "epoch_or_pass": 1,
                "live_train_metric": stored, "ema_dev_metric": .4}])
        logger.start(LoopProgress(2, 256, 1, 0, "paused"))
        logger.progress(LoopProgress(3, 384, 1, 1, "running"), learning_rates=[.001])
        value = json.loads(lines[-1][len(LOG_PREFIX):])
        assert value["train_live_mae_eV"] == .5
        assert value["development_mae_eV"] == .4
    finally:
        session.close()


@pytest.mark.parametrize("mode", MODES)
def test_bounded_registered_sink_logs_committed_dual_curve_and_resumes_units(tmp_path, mode):
    session, out, reopen, _ = prepared(tmp_path, mode)
    session.checkpoint_builder = out.checkpoint_from_builder
    session.batch_finished = out.batch_finished
    lines = []
    out.logger.emit = lines.append
    out.logger.start(session.progress)
    progress = session.run(should_pause=lambda: session.progress.pass_index == 1)
    assert progress.optimizer_step == 2
    value = json.loads(lines[-1][len(LOG_PREFIX):])
    row = out.stage.recorder.record["observations"][-1]
    assert value["optimizer_steps"] == value["checkpoint_optimizer_steps"] == 2
    assert value["train_live_mae_eV"] == row["live_train_metric"]
    assert value["development_mae_eV"] == row["ema_dev_metric"]
    assert row["live_dev_metric"] is not None
    assert not (out.root / "session_commit.json").exists()
    recovered = reopen()
    assert recovered.logger.train_mae == row["live_train_metric"]
    assert recovered.retained_state()["optimizer_step"] == 2


@pytest.mark.parametrize("mode", MODES)
def test_failed_bounded_publication_does_not_emit_epoch_log(tmp_path, monkeypatch, mode):
    session, out, _, _ = prepared(tmp_path, mode)
    lines = []
    out.logger.emit = lines.append
    out.logger.start(session.progress)
    session.batch_finished = out.batch_finished
    session.checkpoint_builder = out.checkpoint_from_builder
    def unavailable(*args, **kwargs):
        raise OSError("synthetic checkpoint storage failure")
    monkeypatch.setattr(out, "checkpoint", unavailable)
    with pytest.raises(OSError, match="synthetic checkpoint"):
        session.run()
    assert session.progress.status == "persistence_failed"
    assert lines == [] and not (out.root / "training_progress.json").exists()


def test_campaign_reference_parser_reuses_optional_metadata_validation():
    import ast
    from pathlib import Path
    from molgap import family_pair_acceptance
    source = Path(family_pair_acceptance.__file__).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == "_validate_campaign_reference")
    text = ast.get_source_segment(source, node)
    assert "_recipe_configuration(record)" in text
    assert "FamilyRecipe.from_dict" not in text


def test_pair_runtime_relays_final_structured_lines_once_per_arm(tmp_path, monkeypatch, capsys):
    import test_experiment_workflow as fixtures
    from molgap import experiment_package, kaggle_pair_runtime as runtime
    spec = fixtures._candidate_pair_spec()
    jobs = fixtures._jobs(spec)
    source, package, inputs, output = [tmp_path / name for name in ("source", "package", "inputs", "output")]
    for directory in (source, package, inputs):
        directory.mkdir()
    (package / "experiment_spec.json").write_text(spec.to_json(), encoding="utf-8")
    launch = inputs / "experiment_launch.json"
    launch.write_text(json.dumps({"expected_package_identity": fixtures.HEX_A,
        "expected_source_archive_sha256": fixtures.HEX_B, "jobs": jobs}), encoding="utf-8")
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "Tesla T4\nTesla T4\n")
    monkeypatch.setattr(experiment_package, "verify_experiment_source_package", lambda *args: {
        "package_identity": fixtures.HEX_A, "spec_identity": spec.identity,
        "archive_sha256": fixtures.HEX_B, "source_commit": "c" * 40})
    class Finished:
        returncode = 0
        def poll(self):
            return self.returncode
    def complete(command, **kwargs):
        if command[command.index("--phase") + 1] == "train":
            arm = command[command.index("--arm") + 1]
            kwargs["stdout"].write(LOG_PREFIX + json.dumps({"format": "molgap-training-log-v1",
                "event": "terminal", "arm_id": arm, "optimizer_steps": 2}) + "\n")
            kwargs["stdout"].flush()
        return Finished()
    monkeypatch.setattr(runtime.subprocess, "Popen", complete)
    report = runtime.run_two_phase_pair(source_root=source, package_dir=package, input_root=inputs,
        launch_path=launch, output=output)
    assert report["status"] == "complete"
    lines = [json.loads(line[len(LOG_PREFIX):]) for line in capsys.readouterr().out.splitlines()
             if line.startswith(LOG_PREFIX)]
    assert [line["arm_id"] for line in lines] == [job["arm_id"] for job in jobs]
    assert all(line["event"] == "terminal" for line in lines)
