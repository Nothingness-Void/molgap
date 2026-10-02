"""Synthetic review regressions; no benchmark data or accelerator execution."""
import json
from dataclasses import replace

import pytest

import test_experiment_workflow as fixtures
from molgap import gptrans_screen_workflow as gptrans
from molgap import kaggle_pair_runtime as runtime
from molgap.experiment_family_workflow import (
    _metric_semantics, build_incomplete_terminal_descriptor, incomplete_observations_from_execution,
)


def test_gptrans_recipe_mode_must_match_executable_arm():
    recipe = gptrans.build_screen_recipe("centered_logits", source_idx_sha256="2" * 64,
                                        target_sha256="3" * 64)
    spec = fixtures._process_spec("gptrans_t", "1", "reference", gptrans, recipe)
    with pytest.raises(ValueError, match="mode"):
        gptrans.validate_screen_recipe(spec, "recipe_candidate", recipe)


def test_ema_only_recipe_still_pins_exact_development_semantics():
    recipe = gptrans.build_screen_recipe("reference", source_idx_sha256="2" * 64,
                                        target_sha256="3" * 64)
    assert _metric_semantics(recipe)["live_dev_metric"] is None
    recipe["development_role_identity"] = "wrong-development-role"
    with pytest.raises(ValueError, match="role identity"):
        _metric_semantics(recipe)


def test_incomplete_attempt_uses_frozen_plan_instead_of_platform_version(tmp_path):
    spec = fixtures._candidate_pair_spec()
    contexts = {arm: replace(context, platform_version="17")
                for arm, context in fixtures._contexts(spec).items()}
    locations = fixtures._incomplete_locations(spec)
    observations = {arm: {"status": "failed", "exit_reason": "synthetic_fault"}
                    for arm in contexts}
    for arm, where in locations.items():
        path = tmp_path / where["trajectory"]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"trajectory_id": where["trajectory_id"],
            "actions": [{"run_ids": [where["run_id"]], "attempt_ids": ["A-frozen-" + arm]}]}))
    result = build_incomplete_terminal_descriptor(spec, contexts=contexts, locations=locations,
                                                 observations=observations, repo_root=tmp_path)
    for entry in result.to_dict()["arms"]:
        assert entry["observed"]["identity"]["attempt_id"]["value"] == "A-frozen-" + entry["arm_id"]
    unknown = build_incomplete_terminal_descriptor(spec, contexts=contexts, locations=locations,
                                                  observations=observations)
    assert all(entry["observed"]["identity"]["attempt_id"]["value"] is None
               for entry in unknown.to_dict()["arms"])


def test_training_spawn_failure_retains_terminal_state(tmp_path, monkeypatch):
    spec = fixtures._candidate_pair_spec()
    jobs = fixtures._jobs(spec)
    source, package, inputs, output = [tmp_path / name for name in
                                       ("source", "package", "inputs", "output")]
    for path in (source, package, inputs):
        path.mkdir()
    (package / "experiment_spec.json").write_text(spec.to_json(), encoding="utf-8")
    launch = inputs / "experiment_launch.json"
    launch.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **k: "Tesla T4\nTesla T4\n")

    class Process:
        def __init__(self, phase):
            self.returncode = 0 if phase == "preflight" else None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, timeout=None):
            return self.returncode
        def kill(self):
            self.returncode = -9

    def spawn(command, **kwargs):
        arm = command[command.index("--arm") + 1]
        phase = command[command.index("--phase") + 1]
        if phase == "train" and arm == jobs[1]["arm_id"]:
            raise OSError("synthetic spawn failure")
        return Process(phase)

    monkeypatch.setattr(runtime.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="synthetic spawn failure"):
        runtime.run_two_phase_pair(source_root=source, package_dir=package, input_root=inputs,
                                  launch_path=launch, output=output)
    state = json.loads((output / "pair_state.json").read_text())
    assert state["arms"][jobs[1]["arm_id"]]["terminal_status"] == "failed"
    assert incomplete_observations_from_execution(spec, state)
