"""CPU-only contract checks for chemical profiling and family CLI wiring."""
from __future__ import annotations

import importlib.util
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import experiment_family_workflow, gptrans_chemical_training
from molgap.gptrans_chemical_training import ChemicalTrainingAddon
from molgap import pcqm_gptrans_v4 as core
from molgap import training_reproducibility
from molgap.experiment_spec import ExperimentSpec
from molgap.gptrans_objective import GPTransObjectiveConfig
from molgap.screen_policy import canonical_fingerprint

PROFILE_PARAMETER_NAMES = frozenset(inspect.signature(
    gptrans_chemical_training.profile_training_overhead
).parameters)


class _Batch:
    def to(self, device, **_kwargs):
        assert device == "cuda"
        return self


class _ProfileModel:
    def __init__(self, arm, objective=None):
        self.arm = arm
        if objective is not None:
            self._training_objective = objective
        self.training = False
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self):
        self.training = False
        self.eval_calls += 1

    def train(self):
        self.training = True
        self.train_calls += 1


class _Scheduler:
    def step(self, epoch):
        assert epoch == 0


def _profile_dependencies(monkeypatch, *, candidate_step_seconds=0.0104,
                          outputs_equal=True):
    config = SimpleNamespace(identity="b" * 64)
    addon = SimpleNamespace(
        config=config,
        validate_graphs=lambda graphs: assert_graphs(graphs),
        attach=lambda batch: batch,
    )
    graphs = [object()]
    assets = SimpleNamespace(train_paths=("synthetic-train",))
    monkeypatch.setattr(core, "configure_fp32_determinism", lambda seed: assert_seed(seed))
    monkeypatch.setattr(core, "validate_fixed_assets", lambda *a, **k: assets)
    monkeypatch.setattr(core, "_load_datasets", lambda _paths: (graphs, ["shard"]))
    monkeypatch.setattr(core, "_target_stats", lambda _shards: (0.0, 1.0))
    monkeypatch.setattr(core, "_training_loader", lambda _graphs, _epoch: [_Batch()])

    factory_calls = []
    models = []

    def make_state(initial_state_path, *args, **kwargs):
        factory_calls.append((initial_state_path, args, kwargs))
        arm = "candidate" if "objective_config" in kwargs else "baseline"
        model = _ProfileModel(arm, kwargs.get("objective_config"))
        models.append(model)
        return model, object(), _Scheduler(), object()

    monkeypatch.setattr(core, "_make_training_state", make_state)
    forward_calls = []

    def forward(model, _batch):
        forward_calls.append(model.arm)
        if model.arm == "candidate" and not outputs_equal:
            return torch.tensor([2.0])
        return torch.tensor([1.0])

    monkeypatch.setattr(core, "_forward", forward)
    clock = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    step_calls = []

    def optimizer_step(model, _optimizer, _ema, _batch, _mean, _std, *, objective, **_kwargs):
        step_calls.append((model.arm, objective))
        clock[0] += 0.01 if model.arm == "baseline" else candidate_step_seconds
        return torch.tensor(1.0)

    monkeypatch.setattr(core, "_optimizer_step", optimizer_step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index=0: "synthetic-device")
    return addon, factory_calls, models, forward_calls, step_calls


def assert_graphs(graphs):
    assert graphs == [graphs[0]]


def assert_seed(seed):
    assert seed == 42


@pytest.mark.parametrize(
    ("candidate_seconds", "accepted", "expected_ratio"),
    [(0.0104, True, 1.04), (0.011, False, 1.1)],
)
def test_profile_compares_equivalent_initial_states_and_enforces_step_gate(
    monkeypatch, tmp_path, candidate_seconds, accepted, expected_ratio
):
    addon, factory_calls, models, forward_calls, step_calls = _profile_dependencies(
        monkeypatch, candidate_step_seconds=candidate_seconds
    )
    initial_state = tmp_path / "initial.pt"

    profile = gptrans_chemical_training.profile_training_overhead(
        addon=addon,
        dataset_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        initial_state_path=initial_state,
        warmup=1,
        repeats=2,
    )

    signature = inspect.signature(gptrans_chemical_training.profile_training_overhead)
    assert {"addon", "dataset_root", "manifest_path", "initial_state_path"} <= set(signature.parameters)
    assert factory_calls == [
        (initial_state, (), {}),
        (initial_state, (), {"objective_config": addon.config}),
    ]
    assert all(model.eval_calls == 1 and model.train_calls == 3 for model in models)
    assert forward_calls == ["baseline", "candidate"]
    assert len(step_calls) == 6
    assert all(objective is None for arm, objective in step_calls if arm == "baseline")
    assert all(objective is addon.config for arm, objective in step_calls if arm == "candidate")
    assert profile["initial_gap_equivalence"] is True
    assert profile["warmup"] == 1 and profile["repeats"] == 2
    assert profile["baseline_step_seconds"] == pytest.approx([0.01, 0.01])
    assert profile["candidate_step_seconds"] == pytest.approx([candidate_seconds] * 2)
    assert profile["median_step_ratio"] == pytest.approx(expected_ratio)
    assert profile["accepted"] is accepted
    assert profile["scope"] == "gpu-resident-optimizer-step-only"
    assert profile["end_to_end_wall_overhead_qualified"] is False


def test_profile_rejects_different_initial_gap_predictions(monkeypatch, tmp_path):
    addon, _calls, _models, _forward_calls, _step_calls = _profile_dependencies(
        monkeypatch, outputs_equal=False
    )

    with pytest.raises(AssertionError):
        gptrans_chemical_training.profile_training_overhead(
            addon=addon,
            dataset_root=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            initial_state_path=tmp_path / "initial.pt",
            warmup=0,
            repeats=1,
        )


class _Cache:
    components = ("descriptors", "fingerprints")
    label_policy = {
        "components": ["descriptors", "fingerprints"],
        "parse_policy": "strict",
        "descriptor_missing_policy": "reject",
    }
    identity = {"fixture": "cache-identity", "label_policy": label_policy}


class _ComponentCache:
    def __init__(self, components, label_policy=None):
        self.components = tuple(components)
        self.label_policy = label_policy or {
            "components": list(self.components),
            "parse_policy": "strict",
            "descriptor_missing_policy": "reject",
        }
        self.identity = {
            "fixture": "cache", "components": self.components,
            "label_policy": self.label_policy,
        }


@pytest.mark.parametrize(
    ("descriptor_weight", "fingerprint_weight", "cache_components", "accepted"),
    [
        (0.1, 0.0, ("descriptors",), True),
        (0.0, 0.1, ("fingerprints",), True),
        (0.1, 0.1, ("descriptors", "fingerprints"), True),
        (0.1, 0.0, ("fingerprints",), False),
        (0.0, 0.1, ("descriptors",), False),
        (0.1, 0.1, ("descriptors",), False),
    ],
)
def test_addon_requires_cache_components_for_positive_weights(
    descriptor_weight, fingerprint_weight, cache_components, accepted
):
    config = GPTransObjectiveConfig(
        descriptor_weight=descriptor_weight,
        fingerprint_weight=fingerprint_weight,
    )
    cache = _ComponentCache(cache_components)

    if accepted:
        addon = ChemicalTrainingAddon(config, cache)
        assert addon.cache is cache
    else:
        with pytest.raises(ValueError):
            ChemicalTrainingAddon(config, cache)


def test_v1_default_label_policy_preserves_legacy_loss_fingerprint():
    config = GPTransObjectiveConfig(descriptor_weight=0.1)
    addon = ChemicalTrainingAddon(config, _Cache())
    base = {"loss_fingerprint": "normalized-gap-l1"}

    fields = addon.scientific_fields(base)

    assert fields["loss_fingerprint"] == f"normalized-gap-l1+chemical-aux:{config.identity}"


def test_nondefault_label_policy_is_bound_to_loss_fingerprint():
    config = GPTransObjectiveConfig(descriptor_weight=0.1)
    policy = {
        "components": ["descriptors"],
        "parse_policy": "strict",
        "descriptor_missing_policy": "reject",
    }
    addon = ChemicalTrainingAddon(
        config,
        _ComponentCache(("descriptors",), label_policy=policy),
    )
    base = {"loss_fingerprint": "normalized-gap-l1"}

    fields = addon.scientific_fields(base)

    assert fields["loss_fingerprint"] == (
        f"normalized-gap-l1+chemical-aux:{config.identity}:label-policy:"
        f"{canonical_fingerprint(policy)}"
    )


def _load_run_arm():
    path = Path(__file__).resolve().parents[1] / "experiments/pcqm_gptrans_chemical_aux/run_arm.py"
    spec = importlib.util.spec_from_file_location("_test_gptrans_chemical_run_arm", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _arm_args(tmp_path, *, phase, family=False):
    config = GPTransObjectiveConfig(descriptor_weight=0.1)
    config_path = tmp_path / "objective.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    args = [
        "--phase", phase,
        "--dataset-root", str(tmp_path / "dataset"),
        "--manifest-path", str(tmp_path / "manifest.json"),
        "--source-archive", str(tmp_path / "source.tar.gz"),
        "--output", str(tmp_path / "output"),
        "--initial-state-path", str(tmp_path / "initial.pt"),
        "--objective-config", str(config_path),
        "--source-archive-sha256", "a" * 64,
        "--source-commit", "b" * 40,
        "--platform-id", "synthetic-platform",
        "--preflight-path", str(tmp_path / "preflight.json"),
        "--cache-root", str(tmp_path / "cache"),
        "--cache-manifest-sha256", "c" * 64,
        "--cache-role-sha256", "d" * 64,
    ]
    if family:
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        contract = tmp_path / "contract.json"
        contract.write_text(json.dumps({"development_role_identity": "synthetic-development"}), encoding="utf-8")
        args.extend([
            "--spec", str(tmp_path / "spec.json"),
            "--package", str(tmp_path / "package"),
            "--contract", str(contract),
            "--expected-package-identity", "e" * 64,
            "--arm-id", "chemical_aux",
            "--trajectory-id", "synthetic-trajectory",
        ])
    return args


def _patch_cache(monkeypatch, run_arm):
    monkeypatch.setattr(
        run_arm,
        "ChemicalLabelCache",
        lambda *_args, **_kwargs: _Cache(),
    )


@pytest.mark.parametrize("profile_accepted", [True, False])
def test_preflight_wires_profile_signature_and_persists_gate_result(
    monkeypatch, tmp_path, capsys, profile_accepted
):
    run_arm = _load_run_arm()
    _patch_cache(monkeypatch, run_arm)
    preflights = []
    monkeypatch.setattr(run_arm, "run_preflight", lambda **kwargs: preflights.append(kwargs) or {"accepted": True})
    profile_calls = []
    profile = {"accepted": profile_accepted, "median_step_ratio": 1.02}

    def profile_stub(**kwargs):
        profile_calls.append(kwargs)
        return profile

    monkeypatch.setattr(gptrans_chemical_training, "profile_training_overhead", profile_stub)
    writes = []
    monkeypatch.setattr(training_reproducibility, "atomic_json", lambda path, value: writes.append((path, value)))
    monkeypatch.setattr("sys.argv", ["run_arm.py", *_arm_args(tmp_path, phase="preflight")])

    if profile_accepted:
        run_arm.main()
    else:
        with pytest.raises(RuntimeError, match="frozen 5% optimizer-step overhead gate"):
            run_arm.main()
    capsys.readouterr()

    assert len(preflights) == 1
    assert "preflight_path" not in preflights[0]
    assert set(preflights[0]) <= set(inspect.signature(core.run_preflight).parameters)
    assert preflights[0]["training_addon"].config.enabled
    assert len(profile_calls) == 1
    profile_kwargs = profile_calls[0]
    assert set(profile_kwargs) == {"addon", "dataset_root", "manifest_path", "initial_state_path"}
    assert set(profile_kwargs) <= PROFILE_PARAMETER_NAMES
    assert profile_kwargs["dataset_root"] == tmp_path / "dataset"
    assert profile_kwargs["manifest_path"] == tmp_path / "manifest.json"
    assert profile_kwargs["initial_state_path"] == tmp_path / "initial.pt"
    assert writes == [(tmp_path / "output" / "objective_profile.json", profile)]


def test_train_wires_session_into_supported_trainer_signature(monkeypatch, tmp_path, capsys):
    run_arm = _load_run_arm()
    _patch_cache(monkeypatch, run_arm)
    from molgap import experiment_family_workflow

    context = SimpleNamespace(
        platform="synthetic-platform",
        account="nothingnessvoid",
        source_commit="b" * 40,
        source_archive_sha256="a" * 64,
    )
    context_calls = []

    def make_context(*args, **kwargs):
        context_calls.append((args, kwargs))
        return context

    monkeypatch.setattr(
        experiment_family_workflow.RunContext,
        "for_training",
        classmethod(lambda cls, *args, **kwargs: make_context(*args, **kwargs)),
    )
    monkeypatch.setattr(ExperimentSpec, "from_json", classmethod(lambda cls, _text: object()))

    class Session:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs
            self.completion = None

        def complete(self, **kwargs):
            self.completion = kwargs
            return {"status": "MECHANICALLY_VERIFIED", "blockers": []}

    sessions = []

    def make_session(*args, **kwargs):
        session = Session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(experiment_family_workflow, "FamilyOutputSession", make_session)
    training_calls = []
    monkeypatch.setattr(run_arm, "run_training", lambda **kwargs: training_calls.append(kwargs) or {"complete": True})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index=0: "synthetic-device")
    writes = []
    monkeypatch.setattr(training_reproducibility, "atomic_json", lambda path, value: writes.append((path, value)))
    monkeypatch.setattr(
        "sys.argv",
        ["run_arm.py", *_arm_args(tmp_path, phase="train", family=True)],
    )

    run_arm.main()
    capsys.readouterr()

    assert "output_session" in inspect.signature(core.run_training).parameters
    assert set(training_calls[0]) <= set(inspect.signature(core.run_training).parameters)
    assert training_calls[0]["preflight_path"] == tmp_path / "preflight.json"
    assert training_calls[0]["output_session"] is sessions[0]
    assert training_calls[0]["training_addon"].config.enabled
    assert context_calls[0][1] == {
        "expected_package_identity": "e" * 64,
        "arm_id": "chemical_aux",
        "account": "nothingnessvoid",
        "run_reference": "nothingnessvoid/molgap-gptrans-chemical-aux-100k-s42-v1",
    }
    assert sessions[0].args[0] == tmp_path / "output" / "family_outputs"
    assert sessions[0].args[1] is context
    assert sessions[0].kwargs["adapter"] == "gptrans-v1"
    assert sessions[0].kwargs["trajectory_id"] == "synthetic-trajectory"
    assert sessions[0].kwargs["metric_semantics"] == {
        "live_train_metric": {
            "metric": "MAE", "unit": "eV", "target": "Gap",
            "role_identity": "official_train_prefix_0_100000", "weights": "live", "direction": "minimize",
        },
        "live_dev_metric": None,
        "ema_dev_metric": {
            "metric": "MAE", "unit": "eV", "target": "Gap",
            "role_identity": "synthetic-development", "weights": "ema", "direction": "minimize",
        },
    }
    assert sessions[0].completion == {
        "runtime": {
            "platform": "synthetic-platform", "account": "nothingnessvoid",
            "source_commit": "b" * 40, "source_archive_sha256": "a" * 64,
            "precision": "fp32",
        },
        "hardware": "synthetic-device",
    }
    assert writes == [(
        tmp_path / "output" / "family_acceptance.json",
        {"status": "MECHANICALLY_VERIFIED", "blockers": []},
    )]


def test_train_requires_family_bindings_before_calling_trainer(monkeypatch, tmp_path):
    run_arm = _load_run_arm()
    _patch_cache(monkeypatch, run_arm)
    training_calls = []
    monkeypatch.setattr(run_arm, "run_training", lambda **kwargs: training_calls.append(kwargs))
    monkeypatch.setattr("sys.argv", ["run_arm.py", *_arm_args(tmp_path, phase="train")])

    with pytest.raises(SystemExit) as error:
        run_arm.main()

    assert error.value.code == 2
    assert training_calls == []


@pytest.mark.parametrize(
    ("argument", "replacement", "context_field", "message_field"),
    [
        ("--source-commit", "f" * 40, "source_commit", "source_commit"),
        ("--source-archive-sha256", "f" * 64, "source_archive_sha256", "source_archive_sha256"),
    ],
)
def test_train_rejects_each_source_identity_mismatch_before_training(
    monkeypatch, tmp_path, argument, replacement, context_field, message_field
):
    run_arm = _load_run_arm()
    _patch_cache(monkeypatch, run_arm)
    from molgap import experiment_family_workflow

    context = SimpleNamespace(
        platform="synthetic-platform",
        account="nothingnessvoid",
        source_commit="b" * 40,
        source_archive_sha256="a" * 64,
    )
    monkeypatch.setattr(
        experiment_family_workflow.RunContext,
        "for_training",
        classmethod(lambda cls, *args, **kwargs: context),
    )
    monkeypatch.setattr(ExperimentSpec, "from_json", classmethod(lambda cls, _text: object()))
    sessions = []
    monkeypatch.setattr(
        experiment_family_workflow,
        "FamilyOutputSession",
        lambda *args, **kwargs: sessions.append((args, kwargs)),
    )
    training_calls = []
    monkeypatch.setattr(run_arm, "run_training", lambda **kwargs: training_calls.append(kwargs))

    argv = _arm_args(tmp_path, phase="train", family=True)
    index = argv.index(argument)
    argv[index + 1] = replacement
    monkeypatch.setattr("sys.argv", ["run_arm.py", *argv])

    with pytest.raises(
        ValueError,
        match=f"Trainer source differs from frozen family context: {message_field}",
    ):
        run_arm.main()

    assert getattr(context, context_field) != replacement
    assert sessions == []
    assert training_calls == []
