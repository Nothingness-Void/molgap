"""CPU-only synthetic custody fixtures; no model, checkpoint, or data decoding."""
from copy import deepcopy
import hashlib
import struct

import pytest

import molgap.cuda_execution_closure as closure
from molgap.research_memory.finalize import verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file


CONTROLS = ("baseline", "non_blocking", "no_redundant_sync", "transfer_and_sync", "cpu_layout")
CASES = (*CONTROLS, "fused", "tf32")


@pytest.fixture
def custody(tmp_path, monkeypatch):
    root, commit = tmp_path, "a" * 40
    folder = root / "experiments/synthetic"
    output = folder / "results"
    source = root / "src/synthetic.json"
    atomic_json(source, {"synthetic_source": True})
    payload = source.read_bytes()
    inputs = dict(source_commit=commit, files={"src/synthetic.json": sha256_file(source)},
                  bindings={}, source_idx=[8, 3, 7, 4], worker_wall_limit_seconds=600)
    for name in ("train_probe", "initial", "normalization_checkpoint"):
        path = root / f"{name}.json"
        atomic_json(path, {"synthetic_binding": name})
        inputs["bindings"][name] = dict(path=str(path), sha256=sha256_file(path))
    atomic_json(folder / "inputs.json", inputs)
    for name in ("README.md", "terminal_decision.md", "attribution.md"):
        (folder / name).write_text("Synthetic NO_TRAIN scratch-only fixture.\n", encoding="ascii")
    rel = "experiments/synthetic"
    cost = dict(schema="molgap-cost-event-v1", cost_event_id="cost-estimate", trajectory_id="TB-fixture",
                action_id="A001", run_id="run-fixture", attempt_id="attempt-fixture", platform="local",
                hardware="synthetic", category="other", evidence_ref=f"{rel}/README.md",
                measurement={u: dict(value=None, status="measurement_missing") for u in
                             ("wall_hours", "cpu_hours", "device_hours", "queue_hours")})
    atomic_json(folder / "plan_input.json", dict(costs=[cost]))
    atomic_json(folder / "rml/costs/estimate.json", cost)
    prior = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id="ev-prior",
        track="B", scope="synthetic", legacy_contract="synthetic-v1",
        outcome={k: "not_evaluated" for k in ("execution_status", "artifact_status", "comparison_status",
                 "scientific_status", "transfer_status", "budget_decision", "full_handoff_status")},
        authority=dict(pointers=[f"{rel}/README.md"]),
        role_use={k: "untouched" for k in ("official_validation", "test_dev", "test_challenge")},
        artifacts=[dict(name="readme", locator=f"{rel}/README.md", sha256=sha256_file(folder / "README.md"), availability="local")],
        migration=dict(migrated_at="2026-10-10", verification_scope="Synthetic fixture", training_executed=False,
                       inference_executed=False, scientific_reinterpretation=False))
    atomic_json(root / "experiments/prior/v5_evidence.json", prior)
    hypothesis = dict(hypothesis_id="H-fixture", supporting_evidence_ids=["ev-prior"],
                      alternative_explanations=["Synthetic noise"], related_closed_family_ids=["synthetic"])
    hypothesis.update({k: "Synthetic fixture" for k in ("observed_deficiency", "changed_mechanism", "cheapest_falsifier",
                       "decision_changed_if_positive", "decision_changed_if_negative")}, expected_native_cost_ref="cost-estimate")
    trajectory = dict(schema="molgap-trajectory-v1", trajectory_id="TB-fixture", record_mode="prospective",
        track="B", owner="desktop", family_id="synthetic", question="Synthetic closure?", hypothesis=hypothesis,
        state_at_start=dict(source_commit=commit, source_config_identity=sha256_file(folder / "inputs.json"),
            contract_refs=[f"{rel}/inputs.json"], reference_ids=[], parent_trajectory_ids=[], prior_evidence_ids=["ev-prior"],
            role_snapshot_refs=[f"{rel}/README.md"], budget_snapshot_ref=f"{rel}/README.md"),
        actions=[dict(action_id="A001", type="execution_profile", source_commit=commit, run_ids=["run-fixture"],
                      attempt_ids=["attempt-fixture"], evidence_refs=[f"{rel}/inputs.json"], cost_event_ids=["cost-estimate"])],
        result=dict(evidence_ids=[], evidence_refs=[]),
        decision=dict(outcome="ACTIVE", decision_ref=f"{rel}/README.md", next_allowed_actions=[], reopen_conditions=[]))
    atomic_json(folder / "rml/trajectory.json", trajectory)
    runtime = dict(source_commit=commit, inputs_sha256=sha256_file(folder / "inputs.json"),
        prospective_sha256=sha256_file(folder / "rml/trajectory.json"), scientific_training=False,
        idle_observations=[dict(idle=True), dict(idle=True)], train_decoded_source_idx=inputs["source_idx"],
        train_executed_source_idx=inputs["source_idx"][:3])
    atomic_json(output / "runtime.json", runtime)
    cases = {}
    for name in CASES:
        exact, changed = name != "tf32", name == "fused"
        cases[name] = dict(status="qualified" if exact else "numerical_mismatch", optimizer_contract_change=changed,
            equivalence=dict(exact=exact, max_abs_error=0 if exact else 0.01), execution_only_admissible=exact and not changed,
            speed_candidate=False, median_seconds_per_step=0.1, timings=[dict(seconds=1, seconds_per_step=0.1) for _ in range(4)])
    atomic_json(output / "result.json", dict(status="complete", cases=cases, equivalence_updates=3, contract_promotion=False))
    atomic_json(output / "cost_observation.json", dict(worker_wall_seconds=12, process_cpu_seconds=2, gpu_busy_seconds=None))

    def git(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return commit + "\n"
        assert args == ["git", "show", f"{commit}:src/synthetic.json"]
        return payload

    monkeypatch.setattr(closure.subprocess, "check_output", git)
    _complete(folder)
    return root, folder, payload


def _complete(folder):
    output = folder / "results"
    names = {"result.json", "runtime.json", "cost_observation.json"}
    if (output / "contrasts.json").exists():
        names.add("contrasts.json")
    atomic_json(output / "completion.json", dict(status="complete", artifacts={n: sha256_file(output / n) for n in names}))


def _close(root, folder):
    return closure.close_cuda_execution_diagnostic(root, folder, evidence_id="ev-fixture", trajectory_id="TB-fixture",
        run_id="run-fixture", scope="synthetic_cuda_execution", legacy_contract="synthetic-v1",
        exact_controls=CONTROLS, expected_cases=CASES, executed_rows=3)


def test_custody_uses_real_finalizer_and_portable_source(custody):
    root, folder, payload = custody
    atomic_json(folder / "__pycache__/excluded.json", {})
    frozen = (folder / "rml/trajectory.json").read_bytes()
    _close(root, folder)
    receipt = verified_receipt(folder / "rml/rml_finalized")
    assert receipt["format"] == "molgap-rml-finalization-v1"
    assert (folder / "rml/trajectory.json").read_bytes() == frozen
    assert (folder / "executed_source/src/synthetic.json").read_bytes() == payload
    acceptance = closure.read(folder / "acceptance.json")
    assert acceptance["trajectory_decision"]["outcome"] == "NO_TRAIN"
    assert acceptance["trajectory_decision"]["next_allowed_actions"] == []
    assert acceptance["training_replay_ready"] is acceptance["generalization_accuracy_qualified"] is False
    cost = acceptance["costs"][0]["measurement"]
    assert cost["wall_hours"] == dict(value=12/3600, status="measured")
    assert cost["cpu_hours"] == dict(value=2/3600, status="measured")
    assert cost["device_hours"] == dict(value=None, status="measurement_missing")
    assert cost["queue_hours"] == dict(value=None, status="not_applicable")
    roles = acceptance["roles"]
    assert [(r["role_name"], r["access_kind"]) for r in roles] == [
        ("train_decoded", "labels_read"), ("train_executed", "prediction_input"), ("train_executed", "metric_computed")]
    assert roles[1]["row_manifest_hash"] == hashlib.sha256(b"".join(struct.pack("<q", i) for i in [8, 3, 7])).hexdigest()
    package = closure.read(folder / "terminal.json")
    assert all(not any(s in p for s in ("/rml/", "__pycache__", "closure_receipt", "terminal.json")) for p in package["artifact_hashes"])


def test_optional_contrasts_and_retry_do_not_rewrite(custody):
    root, folder, _ = custody
    atomic_json(folder / "results/contrasts.json", dict(contrasts={}, strict_fp32_admission=False))
    _complete(folder)
    _close(root, folder)
    hashes = closure.read(folder / "terminal.json")["artifact_hashes"]
    assert hashes["experiments/synthetic/results/contrasts.json"] == sha256_file(folder / "results/contrasts.json")
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert _close(root, folder)["status"] == "ALREADY_FINALIZED"
    assert before == {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}


@pytest.mark.parametrize("wrong_flags", [False, True])
def test_joint_shape_reuses_parent_qualifier(custody, monkeypatch, wrong_flags):
    from molgap import k1_precision_diagnostic as precision

    root, folder, _ = custody
    report = closure.read(folder / "results/result.json")
    cases = report["cases"]
    cases["fused_layout"] = cases.pop("fused")
    cases["fused_tf32_layout"] = cases.pop("tf32")
    for name in ("fused_layout", "fused_tf32_layout"):
        cases[name].update(optimizer_contract_change=True, execution_only_admissible=False,
                           declared_optimizer_flags=dict(foreach=False, fused=not wrong_flags))
    atomic_json(folder / "results/result.json", report)
    _complete(folder)
    called = []
    owner = precision.qualify_combined_report

    def qualified(value):
        called.append(True)
        return owner(value)

    monkeypatch.setattr(precision, "qualify_combined_report", qualified)
    kwargs = dict(evidence_id="ev-fixture", trajectory_id="TB-fixture", run_id="run-fixture",
                  scope="synthetic_cuda_execution", legacy_contract="synthetic-v1", exact_controls=CONTROLS,
                  expected_cases=(*CONTROLS, "fused_layout", "fused_tf32_layout"), executed_rows=3)
    if wrong_flags:
        with pytest.raises(ValueError, match="non-admissible"):
            closure.close_cuda_execution_diagnostic(root, folder, **kwargs)
        assert not (folder / "acceptance.json").exists()
    else:
        closure.close_cuda_execution_diagnostic(root, folder, **kwargs)
        verified_receipt(folder / "rml/rml_finalized")
    assert called == [True]


@pytest.mark.parametrize("mutation", ["sha", "missing_inventory", "listed_other", "non_json"])
def test_contrasts_inventory_is_explicit_and_hash_verified(custody, mutation):
    root, folder, _ = custody
    path = folder / "results/contrasts.json"
    atomic_json(path, dict(contrasts={}))
    _complete(folder)
    if mutation == "sha": atomic_json(path, dict(changed=True))
    if mutation == "non_json":
        path.write_bytes(b"not JSON")
        _complete(folder)
    if mutation in {"missing_inventory", "listed_other"}:
        completion = closure.read(folder / "results/completion.json")
        if mutation == "missing_inventory": del completion["artifacts"]["contrasts.json"]
        else:
            atomic_json(folder / "results/other.json", {})
            completion["artifacts"]["other.json"] = sha256_file(folder / "results/other.json")
        atomic_json(folder / "results/completion.json", completion)
    with pytest.raises(ValueError):
        _close(root, folder)
    assert not (folder / "acceptance.json").exists()


@pytest.mark.parametrize("mutation", ["missing_case", "failed_case", "missing_window", "nan_window", "inf_window", "zero_window",
    "updates", "control", "optimizer_admissible", "runtime_rows", "runtime_count", "empty_idle", "one_idle", "busy_idle",
    "runtime_hash", "cpu_nan", "cpu_negative", "wall_inf", "over_bound", "limit", "gpu_fabricated", "output_hash",
    "extra_output", "missing_inventory", "binding_hash", "source_hash", "git_bytes", "snapshot", "decision", "attribution"])
def test_fail_closed_before_publication(custody, monkeypatch, mutation):
    root, folder, payload = custody
    result, runtime = closure.read(folder / "results/result.json"), closure.read(folder / "results/runtime.json")
    observation = closure.read(folder / "results/cost_observation.json")
    case = result["cases"]["baseline"]
    if mutation == "missing_case": del result["cases"]["tf32"]
    if mutation == "failed_case": case["status"] = "failed_or_unsupported"
    if mutation == "missing_window": case["timings"].pop()
    if mutation in {"nan_window", "inf_window", "zero_window"}:
        case["timings"][0]["seconds"] = {"nan_window": float("nan"), "inf_window": float("inf"), "zero_window": 0}[mutation]
    if mutation == "updates": result["equivalence_updates"] = 2
    if mutation == "control": case.update(status="numerical_mismatch", equivalence=dict(exact=False, max_abs_error=0.1), execution_only_admissible=False)
    if mutation == "optimizer_admissible": result["cases"]["fused"]["execution_only_admissible"] = True
    if mutation == "runtime_rows": runtime["train_decoded_source_idx"] = [3, 8, 7, 4]
    if mutation == "runtime_count": runtime["train_executed_source_idx"].pop()
    if mutation == "empty_idle": runtime["idle_observations"] = []
    if mutation == "one_idle": runtime["idle_observations"].pop()
    if mutation == "busy_idle": runtime["idle_observations"][0]["idle"] = False
    if mutation == "runtime_hash": runtime["prospective_sha256"] = "f" * 64
    if mutation == "cpu_nan": observation["process_cpu_seconds"] = float("nan")
    if mutation == "cpu_negative": observation["process_cpu_seconds"] = -1
    if mutation == "wall_inf": observation["worker_wall_seconds"] = float("inf")
    if mutation == "over_bound": observation["worker_wall_seconds"] = 601
    if mutation == "gpu_fabricated": observation["gpu_busy_seconds"] = 0
    atomic_json(folder / "results/result.json", result)
    atomic_json(folder / "results/runtime.json", runtime)
    atomic_json(folder / "results/cost_observation.json", observation)
    _complete(folder)
    if mutation == "output_hash": atomic_json(folder / "results/runtime.json", dict(runtime, changed=True))
    if mutation == "extra_output": atomic_json(folder / "results/unlisted.json", {})
    if mutation == "missing_inventory":
        completion = closure.read(folder / "results/completion.json")
        del completion["artifacts"]["runtime.json"]
        atomic_json(folder / "results/completion.json", completion)
    if mutation == "binding_hash": atomic_json(root / "initial.json", dict(changed=True))
    if mutation == "source_hash": atomic_json(root / "src/synthetic.json", dict(changed=True))
    if mutation == "git_bytes":
        monkeypatch.setattr(closure.subprocess, "check_output", lambda args, **kw: "a" * 40 if args[1] == "rev-parse" else b"changed")
    if mutation == "snapshot": atomic_json(folder / "executed_source/src/synthetic.json", dict(changed=True))
    if mutation in {"decision", "attribution"}:
        (folder / ("terminal_decision.md" if mutation == "decision" else "attribution.md")).unlink()
    if mutation == "limit":
        inputs = closure.read(folder / "inputs.json")
        inputs["worker_wall_limit_seconds"] = 601
        atomic_json(folder / "inputs.json", inputs)
        trajectory = closure.read(folder / "rml/trajectory.json")
        trajectory["state_at_start"]["source_config_identity"] = sha256_file(folder / "inputs.json")
        atomic_json(folder / "rml/trajectory.json", trajectory)
        runtime.update(inputs_sha256=sha256_file(folder / "inputs.json"), prospective_sha256=sha256_file(folder / "rml/trajectory.json"))
        atomic_json(folder / "results/runtime.json", runtime)
        _complete(folder)
    before = deepcopy(closure.read(folder / "rml/trajectory.json"))
    with pytest.raises(ValueError):
        _close(root, folder)
    assert closure.read(folder / "rml/trajectory.json") == before
    assert not (folder / "acceptance.json").exists()
    assert not (folder / "terminal.json").exists()
