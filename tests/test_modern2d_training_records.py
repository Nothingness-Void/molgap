"""Synthetic metadata only; production byte pins are replaced explicitly.

No source archive, tensor payload, remote command or training is needed.
Finalization tests operate exclusively inside pytest's temporary repository.
"""
import copy
import json
from pathlib import Path

import pytest

from molgap import experiment_launch as launch
from molgap import modern2d_training_records as records
from molgap.common.wire import canonical_fingerprint
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION_V2
from molgap.family_training_runner import read_recipe
from molgap.research_memory.finalize import finalize
from molgap.research_memory.roles import validate_observed_role_truth
from molgap.research_memory.terminal_wiring import inspect_trace_retention_evidence
from molgap.research_memory.trace import file_digest, json_bytes
from molgap.v5_common import validate_v5_evidence_envelope
from test_experiment_spec import payload


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        data = value.encode()
    elif path.name in {"launch_receipt.json", "execution_retention.json"}:
        data = launch.canonical_json(value).encode()
    else:
        data = json_bytes(value)
    path.write_bytes(data)
    return file_digest(path)


def test_real_formal_contract_family_recipe_json_wire_roundtrip():
    root = Path(__file__).resolve().parents[1]
    contracts = (
        ("ppgt_small", "3", root / "experiments/modern2d_100k_formal_screen/ppgt_author_v2/contract.json"),
        ("esa_small", "4", root / "experiments/modern2d_100k_formal_screen/esa_native_v3/contract.json"),
    )
    for family, version, path in contracts:
        record, recipe = read_recipe(path)
        assert (record["family"], record["family_version"]) == (family, version)
        wire = json.loads(json.dumps(recipe.to_dict(), sort_keys=True))
        rebuilt = type(recipe).from_dict(wire)
        assert canonical_fingerprint(rebuilt.to_dict()) == canonical_fingerprint(recipe.to_dict())
        assert json.loads(json.dumps(rebuilt.to_dict(), sort_keys=True)) == wire
        if family == "esa_small":
            assert isinstance(wire["model_config"]["layer_types"], list)
            assert isinstance(rebuilt.model_config.layer_types, tuple)


def repin(metadata, monkeypatch, scope, ref, value):
    path = metadata[scope] / ref
    digest = save(path, value)
    name = {"repo_root": "_REPO_PINS", "inspection_root": "_INPUT_PINS",
            "operations_root": "_OPERATION_PINS"}[scope]
    monkeypatch.setattr(records, name, {**getattr(records, name), ref: digest})


@pytest.fixture
def metadata(tmp_path, monkeypatch, payload):
    from test_terminal_trace_closure import setup_mock_repo

    root, inspection, operations = (tmp_path / name for name in ("repo", "inspection", "operations"))
    root.mkdir()
    setup_mock_repo(root)
    base = root / records.BASE
    payload.update(schema_version=SCHEMA_VERSION_V2, logical_run_id=records.RUN)
    payload["platform"].update(name="kaggle", accelerator="NvidiaTeslaT4", device_count=2)
    payload["prospective"] = {"arms": []}
    for a, arm, interface in zip(payload["arms"], records.ARMS, (("ppgt_small", "3"), ("esa_small", "4"))):
        contract = FAMILIES[interface]
        a.update(arm_id=arm, family=dict(zip(("name", "version"), interface)))
        a["data"]["feature_schema"] = contract.feature_schema
        for role in a["data"]["roles"]:
            role.update(membership_sha256="d" * 64, row_order_sha256="d" * 64)
        for field, name in (("recipe", contract.recipe), ("sampler", contract.sampler), ("transform", contract.transform)):
            a["training"][field]["name"] = name
        payload["prospective"]["arms"].append(dict(arm_id=arm,
            trajectory_id=f"TC-modern2d-formal-100k-{arm}-s42-v1",
            output=f"{records.BASE}/{arm}/rml", plan_spec_ref=f"{records.BASE}/{arm}/plan_input.json",
            plan_spec_sha256="a" * 64))
    spec = ExperimentSpec(payload)
    monkeypatch.setattr(records, "SPEC", spec.identity)
    save(base / "experiment_spec.json", spec.to_dict())
    save(base / "authority.json", {"purpose": "FORMAL_100K_TRAINING", "protected_role_access_authorized": False})
    save(base / "roles/train_100k.json", {"format": "molgap-pcqm4mv2-unified-roles-v2",
        "role": "train_100k", "rows": 100000, "protected": False, "status": "complete",
        "ordering": "ascending-source-index", "graph_chunks": [],
        "source_index_sha256": "d" * 64, "version_identity": "e" * 64})
    reference = json.loads((root / "experiments/ref_exp/trajectory.json").read_bytes())
    for a in spec.to_dict()["arms"]:
        arm = a["arm_id"]
        frozen = copy.deepcopy(reference)
        frozen.update(trajectory_id=f"TC-modern2d-formal-100k-{arm}-s42-v1",
                      owner="server", record_mode="prospective", track="C")
        frozen["hypothesis"].update(hypothesis_id=f"H-{arm}", supporting_evidence_ids=["ev-ref-1"],
            alternative_explanations=["infrastructure"], related_closed_family_ids=["F-ref"],
            observed_deficiency="unmeasured reference", changed_mechanism="formal screen",
            cheapest_falsifier="preflight", expected_native_cost_ref="cost-test-1",
            decision_changed_if_positive="evaluate", decision_changed_if_negative="close")
        frozen["state_at_start"].update(source_commit=records.SOURCE, source_config_identity=canonical_fingerprint(a),
            prior_evidence_ids=["ev-ref-1"], role_snapshot_refs=[f"{records.BASE}/authority.json"],
            budget_snapshot_ref=f"{records.BASE}/authority.json")
        frozen["actions"] = [dict(action_id="A001", type="training", source_commit=records.SOURCE,
            run_ids=[f"{records.RUN}:{arm}:downstream"], attempt_ids=[f"{arm}-v1"],
            cost_event_ids=["cost-test-1"], evidence_refs=["experiments/ref_exp/contract.json"])]
        frozen["result"] = dict(evidence_ids=[], evidence_refs=[])
        frozen["decision"].update(outcome="ACTIVE")
        save(base / arm / "rml/trajectory.json", frozen)
    binding = dict(spec_identity=spec.identity, spec_sha256=spec.identity, source_commit=records.SOURCE,
        package_identity=records.PACKAGE, source_archive_sha256=records.ARCHIVE,
        logical_run_id=records.RUN, experiment_id=payload["experiment_id"], requested_logical_platform="kaggle",
        arms=[dict(arm_id=a["arm_id"], arm_identity=canonical_fingerprint(a)) for a in spec.to_dict()["arms"]])
    fact = lambda value: dict(value=value, missing_reason=None)
    response = dict(format=launch.RESPONSE_FORMAT, version=1, mode="observed", outcome="accepted", conflict_kind=None,
        binding=binding, canonical_platform_reference=fact(records.KERNEL), platform_version=fact("1"),
        physical_runs=fact([dict(arm_ids=list(records.ARMS), canonical_reference=records.KERNEL,
            run_identity="kernel139364627/version1", platform_version=fact("1"))]),
        timestamp=fact("2026-10-18T05:51:28+00:00"), monitor_paths=fact([f"{records.BASE}/STATUS.md"]))
    # Generate an actual shared receipt from a synthetic saved package binding.
    with monkeypatch.context() as patch:
        patch.setattr(launch, "_binding", lambda *args: binding)
        receipt = launch.build_launch_receipt(spec, tmp_path / "unused-package",
            expected_package_identity=records.PACKAGE, response_json=launch.canonical_json(response))
    save(operations / "launch_receipt.json", receipt)
    save(operations / "source_verification.json", dict(kernel=records.KERNEL, kernel_id=139364627, version=1,
        entry_bytes_equal=True, private=True, machine_shape="NvidiaTeslaT4",
        entry_sha256="8cc7f173e150915de3902f7eacb956a91303d2846442991a21da71fe2e01a3b8",
        dataset_sources=["kaseichou/molgap-modern2d-100k-commondev-v1", "kaseichou/molgap-modern2d-formal-source-v1"]))
    save(inspection / "bootstrap_allocation.json", dict(status="failed", hardware=["Tesla T4"] * 2,
        allocation_device_count=2, wall_seconds=71.42726459599999, allocated_device_seconds=142.85452919199997,
        provisioning_before_python_seconds=None, queue_seconds=None))
    execution = dict(format="molgap-kaggle-two-phase-pair-v2", spec_identity=spec.identity,
        hardware=["Tesla T4"] * 2, status="failed", phases=[dict(phase="preflight", status="failed", workers=[])], arms={})
    for device, arm in enumerate(records.ARMS):
        execution["phases"][0]["workers"].append(dict(arm_id=arm, device=device, returncode=1))
        execution["arms"][arm] = dict(device=device, training_started=False, resumed_from_checkpoint=False,
            terminal_status="failed", worker_wall_seconds=8.5, exit_reason="preflight_worker_exit_1")
        save(inspection / f"worker_logs/experiment/{arm}/preflight.log",
            "Traceback (most recent call last):\n_bootstrap(training_only=True, **options)\n"
            "_assert_target_statistics(recipe, data)\nValueError: Decoded train target statistics differ from frozen recipe: "
            "observed=(5.338300266634226, 1.2750902063230574), expected=(5.338300266634226, 1.275090206323082)\n")
    report_sha = save(inspection / "experiment/execution_report.json", execution)
    missing = dict(value=None, status="measurement_missing")
    ledger = dict(format="molgap-allocation-ledger-v1", spec_identity=spec.identity,
        started_at="2026-10-18T05:51:45+00:00", observed_at="2026-10-18T05:52:56+00:00", status="failed",
        allocation_device_count=2, wall_seconds=71.0, allocated_device_seconds=142.0, unassigned_device_seconds=0.0,
        devices=[dict(device=i, hardware="Tesla T4", arm_id=arm, allocated_seconds=71.0) for i, arm in enumerate(records.ARMS)],
        scope="Python bootstrap/runtime observation window; includes idle allocated devices",
        provisioning_before_python_seconds=missing, queue_seconds=missing, prior_segments=[], prior_unobserved_intervals=False)
    ledger_sha = save(inspection / "experiment/allocation_ledger.json", ledger)
    save(inspection / "experiment/execution_retention.json", dict(format="molgap-execution-retention-v1", status="SEALED",
        spec_identity=spec.identity, package_identity=records.PACKAGE, source_commit=records.SOURCE, source_archive_sha256=records.ARCHIVE,
        execution_report=dict(path="execution_report.json", sha256=report_sha), allocation_ledger=dict(
            root=dict(path="allocation_ledger.json", sha256=ledger_sha),
            arms={arm: dict(path=f"{arm}/allocation_ledger.json", sha256=ledger_sha) for arm in records.ARMS})))
    for name, folder in (("_INPUT_PINS", inspection), ("_OPERATION_PINS", operations), ("_REPO_PINS", root)):
        monkeypatch.setattr(records, name, {ref: file_digest(folder / ref) for ref in getattr(records, name)})
    return dict(repo_root=root, inspection_root=inspection, operations_root=operations,
        output=f"{records.BASE}/results/preflight_failure_terminal", finalized_at="2026-10-18T06:00:00+00:00")


def test_prepares_idempotent_independently_finalizable_infrastructure_failure(metadata, monkeypatch):
    original_read = Path.read_bytes

    def metadata_only(path):
        assert path.suffix in {".json", ".log"}, f"Unexpected payload read: {path}"
        return original_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", metadata_only)
        prepared = records.prepare_preflight_failure_terminal(**metadata)
        assert records.prepare_preflight_failure_terminal(**metadata)["status"] == "ALREADY_PREPARED"
    root = metadata["repo_root"]
    assert prepared["status"] == "PREPARED" and set(prepared["arms"]) == set(records.ARMS)
    assert not list(root.rglob("rml_finalized"))
    for arm_id, arm in prepared["arms"].items():
        terminal = json.loads((root / arm["terminal"]).read_bytes())
        acceptance = json.loads((root / terminal["acceptance_ref"]).read_bytes())
        assert terminal["decision"]["outcome"] == "INFRASTRUCTURE_ONLY"
        assert terminal["evidence"]["outcome"]["scientific_status"] == "NOT_EVALUATED"
        assert terminal["evidence"]["outcome"]["execution_status"] == "failed"
        assert acceptance["noReplay"] and acceptance["training_started"] is False
        assert acceptance["optimizer_updates"] == acceptance["model_updates"] == 0
        assert acceptance["progress"] == dict(epoch=0, step=0, samples=0)
        assert acceptance["context"]["arm_id"] == arm_id and acceptance["context"]["platform_version"] == "1"
        window = acceptance["observed_python_window_lower_bound"]
        assert window["device_seconds"] == pytest.approx(71.427264596 * 2)
        assert window["additive_with_ledger"] is False
        assert len(terminal["costs"]) == 1 and terminal["costs"][0]["category"] == "infrastructure_failure"
        assert all(v == dict(status="measurement_missing", value=None) for v in terminal["costs"][0]["measurement"].values())
        assert {(r["role_name"], r["access_kind"], r["selection_used"]) for r in terminal["roles"]} == {
            ("train_100k", "labels_read", False), ("train_100k", "prediction_input", False)}
        assert all(terminal["role_use"][role] == "untouched" for role in
                   ("common_dev", "audit", "official_validation", "test_dev", "test_challenge"))
        validate_v5_evidence_envelope(terminal["evidence"], repo_root=root)
        validate_observed_role_truth([terminal["role_use"]], terminal["roles"])
        assert inspect_trace_retention_evidence(terminal["evidence"]) == (False, None)
        assert "trace_manifest" not in terminal and "action_replay" not in terminal
        for ref, sha in terminal["artifact_hashes"].items():
            assert not Path(ref).is_absolute() and file_digest(root / ref) == sha
        receipt = finalize(root, arm["trajectory"], arm["terminal"])
        assert receipt["trace_status"] == "unavailable" and receipt["outcome"] == "INFRASTRUCTURE_ONLY"
        assert finalize(root, arm["trajectory"], arm["terminal"])["status"] == "ALREADY_FINALIZED"
    assert not list(root.rglob("*.pt"))


@pytest.mark.parametrize("scope,pin_name", [("inspection_root", "_INPUT_PINS"),
    ("operations_root", "_OPERATION_PINS"), ("repo_root", "_REPO_PINS")])
def test_every_pinned_input_rejects_byte_tampering(metadata, scope, pin_name):
    for ref in getattr(records, pin_name):
        path = metadata[scope] / ref
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="SHA256"):
            records.prepare_preflight_failure_terminal(**metadata)
        assert not (metadata["repo_root"] / metadata["output"]).exists()
        path.write_bytes(original)


@pytest.mark.parametrize("scope,ref,keys,value,error", [
    ("operations_root", "source_verification.json", ["version"], 2, "source verification"),
    ("operations_root", "source_verification.json", ["version"], True, "source verification"),
    ("operations_root", "source_verification.json", ["entry_bytes_equal"], False, "source verification"),
    ("inspection_root", "experiment/execution_report.json", ["arms", "ppgt_author_v2", "training_started"], True, "training not started"),
    ("inspection_root", "experiment/execution_report.json", ["arms", "esa_native_v3", "resumed_from_checkpoint"], True, "training not started"),
    ("inspection_root", "experiment/execution_report.json", ["phases", 0, "workers", 1, "returncode"], 0, "preflight worker"),
    ("inspection_root", "experiment/execution_retention.json", ["package_identity"], "f" * 64, "retention identity"),
    ("inspection_root", "experiment/execution_retention.json", ["allocation_ledger", "arms", "esa_native_v3", "path"], "../model.pt", "ledger binding"),
    ("inspection_root", "experiment/execution_retention.json", ["allocation_ledger", "arms", "esa_native_v3", "sha256"], "f" * 64, "Retention"),
    ("repo_root", f"{records.BASE}/roles/train_100k.json", ["protected"], True, "train role"),
])
def test_rebound_metadata_still_requires_exact_failure(metadata, monkeypatch, scope, ref, keys, value, error):
    record = json.loads((metadata[scope] / ref).read_bytes())
    target = record
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    repin(metadata, monkeypatch, scope, ref, record)
    if ref == "experiment/execution_report.json":
        retention_ref = "experiment/execution_retention.json"
        retention = json.loads((metadata[scope] / retention_ref).read_bytes())
        retention["execution_report"]["sha256"] = file_digest(metadata[scope] / ref)
        repin(metadata, monkeypatch, scope, retention_ref, retention)
    with pytest.raises(ValueError, match=error):
        records.prepare_preflight_failure_terminal(**metadata)
    assert not (metadata["repo_root"] / metadata["output"]).exists()


@pytest.mark.parametrize("field", ["source_commit", "spec_identity", "package_identity", "arms", "platform_version"])
def test_rebound_receipt_identity_and_binding_checks(metadata, monkeypatch, field):
    ref = "launch_receipt.json"
    receipt = json.loads((metadata["operations_root"] / ref).read_bytes())
    if field == "platform_version":
        for source in (receipt, receipt["response"]):
            source[field]["value"] = "2"
            source["physical_runs"]["value"][0][field]["value"] = "2"
    else:
        for binding in (receipt["binding"], receipt["response"]["binding"]):
            if field == "arms":
                binding[field][0]["arm_identity"] = "f" * 64
            else:
                binding[field] = "f" * (40 if field == "source_commit" else 64)
    receipt["receipt_identity"] = canonical_fingerprint({k: v for k, v in receipt.items() if k != "receipt_identity"})
    repin(metadata, monkeypatch, "operations_root", ref, receipt)
    with pytest.raises(ValueError, match="binding|physical version"):
        records.prepare_preflight_failure_terminal(**metadata)


def test_rebound_wrong_failure_log_rejected(metadata, monkeypatch):
    ref = f"worker_logs/experiment/{records.ARMS[1]}/preflight.log"
    repin(metadata, monkeypatch, "inspection_root", ref, "different infrastructure failure")
    with pytest.raises(ValueError, match="failure log"):
        records.prepare_preflight_failure_terminal(**metadata)
    assert not (metadata["repo_root"] / metadata["output"]).exists()


def test_changed_spec_is_not_silently_adopted(metadata):
    path = metadata["repo_root"] / records.BASE / "experiment_spec.json"
    declaration = json.loads(path.read_bytes())
    declaration["arms"][0]["initialization"]["state_sha256"] = "f" * 64
    save(path, declaration)
    with pytest.raises(ValueError, match="Spec/arm identity"):
        records.prepare_preflight_failure_terminal(**metadata)
    assert not (metadata["repo_root"] / metadata["output"]).exists()


def test_no_clobber_or_unexpected_output_inventory(metadata):
    prepared = records.prepare_preflight_failure_terminal(**metadata)
    path = metadata["repo_root"] / prepared["arms"][records.ARMS[0]]["terminal"]
    original = path.read_bytes()
    path.write_bytes(b"user-owned change")
    with pytest.raises(ValueError, match="PREPARATION_CONFLICT"):
        records.prepare_preflight_failure_terminal(**metadata)
    assert path.read_bytes() == b"user-owned change"
    path.write_bytes(original)
    extra = path.parent / "user.json"
    extra.write_bytes(b"{}")
    with pytest.raises(ValueError, match="PREPARATION_CONFLICT"):
        records.prepare_preflight_failure_terminal(**metadata)
    assert extra.read_bytes() == b"{}"


@pytest.mark.parametrize("output", ["../escape", "experiments", f"{records.BASE}/results",
    f"{records.BASE}/ppgt_author_v2/rml", f"{records.BASE}/results/rml_finalized"])
def test_output_is_a_separate_confined_results_directory(metadata, output):
    with pytest.raises(ValueError):
        records.prepare_preflight_failure_terminal(**{**metadata, "output": output})


@pytest.mark.parametrize("timestamp", [None, "", "invalid", "2026-10-18T06:00:00", "2026-10-18T15:00:00+09:00"])
def test_explicit_utc_timestamp_required(metadata, timestamp):
    with pytest.raises(ValueError):
        records.prepare_preflight_failure_terminal(**{**metadata, "finalized_at": timestamp})
    assert not (metadata["repo_root"] / metadata["output"]).exists()
