"""Synthetic account rebinding without checkpoints, models or remote services."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from molgap import pcqm_500k_preparation as preparation


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("k1_pretrained_mean2", "k1_pretrained_consistency")
RUN = "molgap-k1-consistency-500k-pair-s42-v1"
GRAPH = "pcqm4mv2-ogb-fixed-500k-scnet-v1"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    repo, previous, resume = tmp_path / "repo", tmp_path / "v2", tmp_path / "resume"
    experiment = repo / "experiments" / "pair"
    output = tmp_path / "v3"
    spec_value = {"arms": [dict(arm_id=a, data=dict(manifest="frozen-data")) for a in ARMS]}
    fake_spec = SimpleNamespace(identity="frozen-spec", to_dict=lambda: deepcopy(spec_value))
    monkeypatch.setattr(preparation, "ExperimentSpec", SimpleNamespace(from_json=lambda text: fake_spec))
    old = dict(archive_sha256="a" * 64, package_identity="old-package", relative_allowlist=["source.py"])
    monkeypatch.setattr(preparation, "verify_experiment_source_package", lambda path: old)
    put(previous / "package/experiment_spec.json", spec_value)
    put(experiment / "experiment_spec.json", spec_value)
    launcher = repo / "platforms/kaggle/run_legacy_500k_pair.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes((ROOT / "platforms/kaggle/run_legacy_500k_pair.py").read_bytes())
    source = previous / "source_dataset"
    launch = dict(account="nothingnessvoid", run_reference="nothingnessvoid/" + RUN,
                  dataset_sources=["nothingnessvoid/old-source", "nothingnessvoid/" + GRAPH,
                                   "nothingnessvoid/old-resume"],
                  source_mount="old-source", graph_mount=GRAPH, max_stage_seconds=32400,
                  stage_epochs=60, prospective_sha256={a: "prospect-" + a for a in ARMS},
                  required_modules=["synthetic-module"], arms=[])
    states = {}
    for device, aid in enumerate(ARMS):
        recipe = "recipes/" + aid + ".json"
        put(source / recipe, dict(epochs=60, coefficient=device * .1))
        put(source / "prospective" / aid / "trajectory.json", dict(trajectory_id=aid))
        launch["arms"].append(dict(arm_id=aid, device=device, recipe=recipe,
                                    initial_state="inputs/" + aid + ".pt",
                                    resume=dict(mount="old-resume")))
        prior = resume / aid
        contract = dict(epochs=60, coefficient=device * .1)
        runtime = dict(installed_distributions=["torch==2.4.1", "numpy==1.26.4"],
                       installed_distributions_sha256="runtime-pin")
        artifacts = {}
        for name in ("last_checkpoint.pt", "best_model.pt", "best_predictions.pt", "initial_state.pt",
                     "runtime.json", "trace.json", "scientific_contract.json", "data_manifest.json",
                     "runtime_certificate.json"):
            path = prior / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "runtime.json":
                put(path, runtime)
            else:
                path.write_bytes(b"synthetic artifact, not a tensor")
            artifacts[name] = preparation.sha256_file(path)
        artifacts["predictions_epoch_45.pt"] = "not-downloaded"
        put(prior / "stage_manifest.json", dict(arm=aid, source_sha256=old["archive_sha256"],
                                                next_epoch=46, artifacts=artifacts, contract=contract))
        states[aid] = dict(arm=aid, source_sha256=old["archive_sha256"], next_epoch=46,
                           trace=[{}] * 46, next_batch_index=0, global_step=3906 * 46,
                           next_schedule_epoch=46, runtime_software="runtime-pin", contract=contract)
    put(source / "legacy_500k_launch.json", launch)
    put(source / "dataset-metadata.json", dict(id="nothingnessvoid/old-source", isPrivate=True))
    put(previous / "kernel/kernel-metadata.json", dict(id="nothingnessvoid/" + RUN,
                                                       title="MolGap K1 Consistency 500K Pair S42 V1"))
    new = dict(source_commit="b" * 40, archive_sha256="b" * 64, package_identity="new-package")
    def build(spec, root, allowlist, destination):
        destination.mkdir()
        put(destination / "experiment_spec.json", spec.to_dict())
        (destination / "source.tar.gz").write_bytes(b"synthetic source archive")
        return new
    monkeypatch.setattr(preparation, "build_experiment_source_package", build)
    load = Mock(side_effect=lambda path, **kw: deepcopy(states[Path(path).parent.name]))
    monkeypatch.setattr(torch, "load", load)
    check = Mock(return_value=dict(status="PASSED", errors=[]))
    monkeypatch.setattr(preparation, "check_release_inputs", check)
    return SimpleNamespace(repo=repo, previous=previous, resume=resume, experiment=experiment,
                           output=output, source=source, states=states, load=load, check=check,
                           spec_value=spec_value, launch=launch)


def prepare(fixture, **overrides):
    options = dict(account="nvoid912", source_dataset="nvoid912/fresh-source",
                   graph_dataset="nvoid912/" + GRAPH, platform_id="kaggle3-t4x2",
                   run_reference="nvoid912/" + RUN, max_stage_seconds=30000)
    checkpoint = overrides.pop("checkpoint_dataset", "nvoid912/fresh-resume")
    options.update(overrides)
    preparation.prepare_continuation(fixture.previous, fixture.resume, fixture.output,
        fixture.repo / "synthetic-unused-pickle", checkpoint,
        repo_root=fixture.repo, experiment_dir=fixture.experiment, **options)


def test_kaggle3_rebind_preserves_science_and_resume(fixture):
    before = {p: p.read_bytes() for root in (fixture.previous, fixture.resume, fixture.experiment)
              for p in root.rglob("*") if p.is_file()}
    prepare(fixture)
    source = fixture.output / "source_dataset"
    launch = preparation.read(source / "legacy_500k_launch.json")
    assert launch["account"] == "nvoid912"
    assert launch["run_reference"] == "nvoid912/" + RUN
    assert launch["platform_id"] == "kaggle3-t4x2"
    assert launch["max_stage_seconds"] == 30000
    assert launch["stage_epochs"] == 60
    assert launch["dataset_sources"] == ["nvoid912/fresh-source", "nvoid912/" + GRAPH, "nvoid912/fresh-resume"]
    assert launch["prospective_sha256"] == fixture.launch["prospective_sha256"]
    assert launch["runtime_distributions"] == ["torch==2.4.1", "numpy==1.26.4"]
    for arm in launch["arms"]:
        assert arm["resume"]["next_epoch"] == 46
        assert arm["resume"]["retention"] == "selected-and-resume-v1"
        assert (source / arm["recipe"]).read_bytes() == (fixture.source / arm["recipe"]).read_bytes()
        assert (source / "prospective" / arm["arm_id"] / "trajectory.json").read_bytes() == (
            fixture.source / "prospective" / arm["arm_id"] / "trajectory.json").read_bytes()
    metadata = preparation.read(fixture.output / "kernel/kernel-metadata.json")
    assert metadata["id"] == launch["run_reference"]
    assert metadata["dataset_sources"] == launch["dataset_sources"]
    assert metadata["title"] == "MolGap K1 Consistency 500K Pair S42 V1"
    assert metadata["is_private"] is True
    assert preparation.read(source / "dataset-metadata.json")["isPrivate"] is True
    assert preparation.read(fixture.output / "checkpoint_dataset/dataset-metadata.json")["isPrivate"] is True
    binding = preparation.read(fixture.output / "continuation_binding.json")
    assert binding["prior_account"] == "nothingnessvoid"
    assert binding["account"] == "nvoid912"
    assert binding["prior_run_reference"] == "nothingnessvoid/" + RUN
    assert binding["run_reference"] == "nvoid912/" + RUN
    assert binding["original_data_identity"] == {a["arm_id"]: a["data"] for a in fixture.spec_value["arms"]}
    fixture.check.assert_called_once()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not list((fixture.output / "checkpoint_dataset").rglob("predictions_epoch_*.pt"))
    for call in fixture.load.call_args_list:
        assert call.kwargs == dict(map_location="cpu", weights_only=False)


@pytest.mark.parametrize("override", [
    dict(checkpoint_dataset="nothingnessvoid/fresh-resume"),
    dict(source_dataset="nothingnessvoid/fresh-source"),
    dict(graph_dataset="nothingnessvoid/" + GRAPH),
    dict(graph_dataset="nvoid912/other-graph"),
    dict(source_dataset="nvoid912/../source"),
    dict(source_dataset=""),
    dict(source_dataset=None),
    dict(account="unsupported"),
    dict(platform_id="kaggle1-t4x2"),
    dict(run_reference="nothingnessvoid/" + RUN),
    dict(run_reference="nvoid912/different-title-slug"),
    dict(checkpoint_dataset="nvoid912/fresh-source"),
    dict(max_stage_seconds=32401),
    dict(max_stage_seconds=0),
    dict(max_stage_seconds=True),
])
def test_invalid_account_arguments_fail_before_staging(fixture, override):
    with pytest.raises(ValueError):
        prepare(fixture, **override)
    assert not fixture.output.exists()
    fixture.load.assert_not_called()
    fixture.check.assert_not_called()


def test_backward_account_defaults_remain_kaggle1(fixture):
    preparation.prepare_continuation(fixture.previous, fixture.resume, fixture.output,
        fixture.repo / "synthetic-unused-pickle", "nothingnessvoid/new-resume",
        repo_root=fixture.repo, experiment_dir=fixture.experiment,
        source_dataset="nothingnessvoid/new-source")
    launch = preparation.read(fixture.output / "source_dataset/legacy_500k_launch.json")
    assert launch["account"] == "nothingnessvoid"
    assert launch["platform_id"] == "kaggle1-t4x2"
    assert launch["max_stage_seconds"] == 32400


@pytest.mark.parametrize("field,value", [
    ("next_batch_index", 1), ("global_step", 1), ("next_schedule_epoch", 45),
    ("runtime_software", "different"), ("contract", {"epochs": 59}),
])
def test_strict_resume_checks_not_relaxed(fixture, field, value):
    fixture.states[ARMS[0]][field] = value
    with pytest.raises(ValueError, match="cursor/scientific/runtime"):
        prepare(fixture)
    fixture.check.assert_not_called()


def test_mismatched_pair_runtime_still_rejected(fixture):
    path = fixture.resume / ARMS[1] / "runtime.json"
    runtime = preparation.read(path)
    runtime["installed_distributions"] = ["different==1"]
    put(path, runtime)
    manifest_path = path.parent / "stage_manifest.json"
    manifest = preparation.read(manifest_path)
    manifest["artifacts"]["runtime.json"] = preparation.sha256_file(path)
    put(manifest_path, manifest)
    with pytest.raises(ValueError, match="runtimes differ"):
        prepare(fixture)


def test_changed_resume_bytes_still_rejected(fixture):
    (fixture.resume / ARMS[0] / "last_checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare(fixture)
    fixture.load.assert_not_called()


def test_failed_release_check_is_not_reported_as_prepared(fixture):
    fixture.check.return_value = dict(status="FAILED", errors=["synthetic release blocker"])
    with pytest.raises(RuntimeError, match="local release failed"):
        prepare(fixture)
    assert preparation.read(fixture.output / "release_report.json")["errors"] == ["synthetic release blocker"]


def test_bootstrap_platform_label_uses_config_with_backward_default():
    tree = ast.parse((ROOT / "platforms/kaggle/run_legacy_500k_pair.py").read_text(encoding="utf-8"))
    expressions = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                   and node.args and isinstance(node.args[0], ast.Constant)
                   and node.args[0].value == "platform_id"]
    assert len(expressions) == 1
    code = compile(ast.Expression(expressions[0]), "synthetic-platform-label", "eval")
    assert eval(code, {"config": {}}) == "kaggle1-t4x2"
    assert eval(code, {"config": {"platform_id": "kaggle3-t4x2"}}) == "kaggle3-t4x2"


def test_thin_cli_forwards_explicit_continuation_options(monkeypatch, tmp_path):
    path = ROOT / "experiments/pcqm_k1_consistency_ablation_500k/prepare.py"
    spec = importlib.util.spec_from_file_location("synthetic_500k_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    call = Mock()
    monkeypatch.setattr(module, "prepare_continuation", call)
    args = [str(path), "--continuation-from", str(tmp_path / "v2"), "--resume-root", str(tmp_path / "resume"),
            "--output", str(tmp_path / "v3"), "--checkpoint-dataset", "nvoid912/fresh-resume",
            "--source-dataset", "nvoid912/fresh-source", "--account", "nvoid912",
            "--run-reference", "nvoid912/" + RUN, "--graph-dataset", "nvoid912/" + GRAPH,
            "--platform-id", "kaggle3-t4x2", "--max-stage-seconds", "30000"]
    monkeypatch.setattr("sys.argv", args)
    module.main()
    call.assert_called_once()
    assert call.call_args.kwargs["account"] == "nvoid912"
    assert call.call_args.kwargs["platform_id"] == "kaggle3-t4x2"
    assert call.call_args.kwargs["max_stage_seconds"] == 30000
    assert call.call_args.kwargs["run_reference"] == "nvoid912/" + RUN
    monkeypatch.setattr("sys.argv", [str(path), "--account", "nvoid912"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    call.assert_called_once()
