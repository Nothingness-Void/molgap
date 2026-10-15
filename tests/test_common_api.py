"""Common API contract: synthetic CPU/files only; no jobs or real credentials."""
import ast
from copy import deepcopy
import hashlib
from importlib.metadata import distribution, PackageNotFoundError
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from molgap.common.declarations import (
    ExecutionProfile, ExecutionQuestion, build_execution_proposal, comparison_base,
)
from molgap.common.identity import validate_execution_identity
from molgap.common import kaggle, transport


def _arguments():
    return dict(
        question=ExecutionQuestion("policy", "DIAGNOSTIC", "question", "gap", "mechanism",
                                   (), (), (), "retain", "reject"),
        arm={"arm_id": "candidate"}, trajectory_id="T-common-synthetic",
        source_commit="a" * 40, run="run", version="v1", contract_ref="contract.json",
        contract_refs=["contract.json"], authority_ref="authority.json", role_ref="roles.json",
        decision_ref="decision.md", timestamp="2026-10-15T00:00:00Z")


def _profile(**kwargs):
    return ExecutionProfile(**(dict(owner="desktop", track="B", platform="local",
        hardware="CPU", cheapest_falsifier="synthetic explicit diagnostic",
        estimated_device_hours=None, estimated_wall_hours=None) | kwargs))


@pytest.mark.parametrize("owner,track,platform,hardware", [
    ("desktop", "B", "kaggle1", "T4"),
    ("server", "C", "kaggle3", "T4"),
    ("desktop", "B", "ims", "A100"),
    ("server", "C", "scnet", "DCU"),
    ("desktop", "A", "local", "CPU"),
])
def test_proposal_has_no_host_or_experiment_defaults(owner, track, platform, hardware):
    profile = _profile(owner=owner, track=track, platform=platform, hardware=hardware)
    arguments = _arguments()
    before = deepcopy(arguments)
    result = build_execution_proposal(**arguments, profile=profile)
    t, c = result["trajectory"], result["costs"][0]
    assert (t["owner"], t["track"], c["platform"], c["hardware"]) == (owner, track, platform, hardware)
    assert t["hypothesis"]["cheapest_falsifier"] == profile.cheapest_falsifier
    assert t["record_mode"] == "prospective" and t["decision"]["outcome"] == "ACTIVE"
    assert c["measurement"]["device_hours"] == {"status": "measurement_missing", "value": None}
    assert c["measurement"]["wall_hours"] == {"status": "measurement_missing", "value": None}
    assert arguments == before
    assert "seed42" not in json.dumps(result) and "T4x2" not in json.dumps(result)
    with pytest.raises(TypeError):
        build_execution_proposal(**arguments)


@pytest.mark.parametrize("kwargs", [
    {"owner": "automatic"}, {"track": "unknown"}, {"platform": ""},
    {"hardware": " "}, {"cheapest_falsifier": ""},
    {"estimated_device_hours": -1}, {"estimated_wall_hours": float("inf")},
    {"estimated_device_hours": True}, {"estimated_wall_hours": float("nan")},
])
def test_profile_requires_explicit_valid_facts(kwargs):
    with pytest.raises(ValueError):
        _profile(**kwargs)


def test_estimates_remain_estimates_even_when_zero():
    result = build_execution_proposal(**_arguments(), profile=_profile(
        estimated_device_hours=0, estimated_wall_hours=1.25))
    assert result["costs"][0]["measurement"]["device_hours"] == {"status": "estimated", "value": 0}
    assert result["costs"][0]["measurement"]["wall_hours"] == {"status": "estimated", "value": 1.25}


@pytest.mark.parametrize("ids", [("solo",), ("baseline", "variant"), ("a", "b", "c")])
def test_storage_root_is_not_hardcoded_to_two_named_arms(ids):
    declaration = dict(experiment_id="example", arms=[dict(arm_id=a) for a in ids],
        prospective=dict(arms=[dict(arm_id=a, plan_spec_ref="experiments/example/"+a+"/plan_input.json")
                               for a in ids]))
    spec = NS(to_dict=lambda: declaration)
    assert comparison_base(spec, experiment_id="example", base="experiments/example",
                           arm_ids=ids) == "experiments/example"


def _identity(same_run=False):
    declaration = {"logical_run_id": "run", "prospective": {"same_run_replay": {} if not same_run else {"bound": True}}}
    trajectory = {"state_at_start": {"source_commit": "a" * 40}, "actions": [
        {"source_commit": "a" * 40, "run_ids": ["run:arm:downstream"],
         "attempt_ids": ["run-v1" if same_run else "arm-v1"]}]}
    return NS(to_dict=lambda: declaration), trajectory


@pytest.mark.parametrize("same_run", [False, True])
def test_identity_validation_is_pure_and_keeps_attempt_semantics(same_run):
    spec, trajectory = _identity(same_run)
    before = deepcopy(trajectory)
    assert validate_execution_identity(spec, "arm", trajectory, "a"*40) == (
        "run-v1" if same_run else "arm-v1")
    assert trajectory == before


@pytest.mark.parametrize("defect", ["state", "action", "run", "attempt", "duplicate"])
def test_common_identity_cannot_rewrite_away_a_mismatch(defect):
    spec, trajectory = _identity()
    if defect == "state":
        trajectory["state_at_start"]["source_commit"] = "b" * 40
    elif defect == "action":
        trajectory["actions"][0]["source_commit"] = "b" * 40
    elif defect == "run":
        trajectory["actions"][0]["run_ids"] = ["wrong"]
    elif defect == "attempt":
        trajectory["actions"][0]["attempt_ids"] = ["other-v1"]
    else:
        trajectory["actions"].append(deepcopy(trajectory["actions"][0]))
    before = deepcopy(trajectory)
    with pytest.raises(ValueError):
        validate_execution_identity(spec, "arm", trajectory, "a"*40)
    assert trajectory == before


def test_compatibility_paths_use_common_objects():
    from molgap import execution_comparison, kaggle_output_retrieval, kaggle_submission
    assert execution_comparison.ExecutionQuestion is ExecutionQuestion
    assert kaggle_output_retrieval._download_selected is transport.download_pinned_files
    assert kaggle_submission._dataset_status is kaggle.dataset_status
    assert kaggle_submission._dataset_details is kaggle.dataset_details
    assert kaggle_submission._kernel_source is kaggle.kernel_source


def test_runtime_aliases_do_not_relocate_checkpoint_types():
    from molgap.common import runtime
    from molgap import model_substrate, family_training_state, fast_sandbox, source_dependencies
    assert runtime.ModelSubstrate is model_substrate.ModelSubstrate
    assert runtime.FamilyTrainingState is family_training_state.FamilyTrainingState
    assert runtime.run_sandbox is fast_sandbox.run_sandbox
    assert runtime.discover_source_dependencies is source_dependencies.discover_source_dependencies
    assert runtime.FamilyTrainingState.__module__ == "molgap.family_training_state"
    from molgap import family_training_kernel, family_training_session
    assert runtime.FamilyKernel is family_training_kernel.FamilyKernel
    assert runtime.FamilyTrainingSession is family_training_session.FamilyTrainingSession


@pytest.mark.parametrize("value", ["ready", "READY", "processing"])
def test_sdk_status_uses_real_positional_signature(value):
    class Api:
        def dataset_status(self, ref, /):
            assert ref == "owner/source"
            return value
    assert kaggle.dataset_status(Api(), "owner/source") == {"status": value.lower()}


@pytest.mark.parametrize("value", [None, "", {}, {"status": "ready"}, 1])
def test_malformed_sdk_status_is_not_a_ready_source(value):
    with pytest.raises(ValueError):
        kaggle.dataset_status(NS(dataset_status=lambda _: value), "owner/source")


def test_unknown_sdk_state_never_leaks_a_signed_url_or_retries():
    calls = []
    def fail(ref):
        calls.append(ref)
        raise RuntimeError("secret signed-url")
    with pytest.raises(RuntimeError) as error:
        kaggle.dataset_status(NS(dataset_status=fail), "owner/source")
    assert "signed-url" not in str(error.value) and len(calls) == 1
    class Missing(Exception):
        status = 404
    def missing(_):
        raise Missing()
    assert kaggle.dataset_status(NS(dataset_status=missing), "owner/source") is None


def test_installed_sdk_status_signature_without_import_authentication():
    try:
        source = Path(distribution("kaggle").locate_file("kaggle/api/kaggle_api_extended.py"))
    except PackageNotFoundError:
        pytest.skip("Kaggle SDK is not installed")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "dataset_status")
    parameters = [a.arg for a in method.args.args]
    assert parameters in (["self", "dataset"], ["self", "dataset", "format"])
    if len(parameters) == 3:
        assert ast.literal_eval(method.args.defaults[-1]) is None


def test_explicit_account_client_ignores_global_oauth_without_environment_changes(tmp_path, monkeypatch):
    import os
    before = dict(os.environ)
    clients = []
    class Api:
        def _load_config(self, value):
            self.config = value
    def client(**kwargs):
        value = NS(credentials=kwargs, _http_client=NS(_session=NS(auth="global-oauth"), _signed_in=False))
        clients.append(value)
        return value
    api_module = ModuleType("kaggle.api.kaggle_api_extended")
    api_module.KaggleApi = Api
    sdk_module = ModuleType("kagglesdk")
    sdk_module.KaggleClient = client
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", api_module)
    monkeypatch.setitem(sys.modules, "kagglesdk", sdk_module)
    key = tmp_path / "synthetic-key.json"
    key.write_text(json.dumps({"username": "requested", "key": "synthetic-not-a-real-key"}))
    api = kaggle.api_for_credentials(key)
    for _ in range(2):
        c = api.build_kaggle_client()
        c._http_client._try_fill_auth()
        assert c._http_client._session.auth == ("requested", "synthetic-not-a-real-key")
        assert c._http_client._signed_in is True
    assert clients[0] is not clients[1] and dict(os.environ) == before


class Response:
    status_code = 200
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, size):
        yield b"synthetic bytes"


def _download(tmp_path, **overrides):
    arguments = dict(urls={"remote.json": "synthetic-url"},
        required={"remote.json": ("kept.json", hashlib.sha256(b"synthetic bytes").hexdigest())},
        destination=tmp_path / "retained", http_session=NS(get=lambda *a, **k: Response()))
    arguments.update(overrides)
    return transport.download_pinned_files(**arguments)


def test_transport_is_atomic_pinned_and_idempotent(tmp_path):
    first = _download(tmp_path)
    second = _download(tmp_path, http_session=NS(get=lambda *a, **k: pytest.fail("repeat download")))
    assert first[0]["state"] == "RETAINED" and second[0]["state"] == "ALREADY_RETAINED"
    assert (tmp_path / "retained/kept.json").read_bytes() == b"synthetic bytes"
    assert len(list((tmp_path / "retained").iterdir())) == 1


def test_transport_never_publishes_hash_mismatch_or_overwrites_conflict(tmp_path):
    with pytest.raises(ValueError, match="hash"):
        _download(tmp_path, required={"remote.json": ("kept.json", "a"*64)})
    assert list((tmp_path / "retained").iterdir()) == []
    (tmp_path / "retained/kept.json").write_bytes(b"existing")
    with pytest.raises(ValueError, match="conflicts"):
        _download(tmp_path)
    assert (tmp_path / "retained/kept.json").read_bytes() == b"existing"


@pytest.mark.parametrize("binding", [
    ("../outside.json", "a"*64), ("/absolute.json", "a"*64),
    ("C:/outside.json", "a"*64), ("CON", "a"*64), ("kept.json", "not-a-hash"),
])
def test_invalid_pins_fail_before_download_or_destination_creation(tmp_path, binding):
    with pytest.raises(ValueError):
        _download(tmp_path, required={"remote.json": binding})
    assert not (tmp_path / "retained").exists()


def test_transport_rejects_aliases_and_unselected_urls(tmp_path):
    with pytest.raises(ValueError):
        _download(tmp_path, required={"a.json": ("x.json", "a"*64), "b.json": ("X.json", "b"*64)})
    with pytest.raises(ValueError):
        _download(tmp_path, urls={"remote.json": "url", "extra.json": "url"})
    assert not (tmp_path / "retained").exists()


def test_package_closure_includes_declaration_identity_and_package_init():
    from molgap.source_dependencies import discover_source_dependencies
    root = Path(__file__).resolve().parents[1]
    closure = discover_source_dependencies(root, [
        "src/molgap/common/declarations.py", "src/molgap/common/identity.py"])
    expected = {"src/molgap/common/"+name+".py"
                for name in ("__init__", "declarations", "identity")}
    assert expected <= set(closure.source_files)
    from molgap.experiment_source_inventory import COMMON_SOURCE_FILES
    assert {"src/molgap/common/__init__.py", "src/molgap/common/declarations.py",
            "src/molgap/common/identity.py"} <= set(COMMON_SOURCE_FILES)
