"""Thin local acceptance entry: explicit decisions and no hand-authored maps."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from molgap import experiment_cli as cli
from molgap import experiment_run as runs


def test_accept_workflow_cli_rejects_ambiguous_options_and_forwards_unexercised_ones(tmp_path, monkeypatch, capsys):
    # Per-arm outcome/decision, --retrieved-root, --role-audit and the execute=False
    # default go through the real CLI in test_run_acceptance_integration. The strict
    # owner runs there directly, so --strict forwarding is still checked here, together
    # with parser-level rejection and the options no real CLI flow passes.
    owner = Mock(return_value={"status": "MECHANICALLY_VERIFIED", "arms": {}, "executed": False})
    monkeypatch.setattr(runs, "accept_run_workflow", owner)
    arguments = ["accept-workflow", "--run-id", "test", "--attempt-id", "v1",
        "--repo-root", str(tmp_path), "--profile", str(tmp_path / "profile.json")]
    physical = tmp_path / "physical.json"
    physical.write_text('[{"platform":"kaggle","platform":"other"}]', encoding="utf-8")
    for extra in (["--outcome", "CLOSED", "--arm-outcome", "reference=CLOSED"],
                  ["--decision-ref", "decision.md", "--arm-decision-ref", "reference=decision.md"],
                  ["--arm-outcome", "reference=CLOSED", "--arm-outcome", "reference=INCONCLUSIVE"],
                  ["--arm-decision-ref", "reference=one.md", "--arm-decision-ref", "reference=two.md"],
                  ["--physical-runs", str(physical)]):
        assert cli.main([*arguments, *extra]) == 2
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "ERROR"
        owner.assert_not_called()
    assert "Duplicate JSON key" in output["error"]["message"]
    physical.write_text(json.dumps([{"physical_run": "bound-by-owner"}]), encoding="utf-8")
    receipt, recovery, authority = tmp_path / "v3.json", tmp_path / "recovery", tmp_path / "decision.md"
    observation = tmp_path / "allocation_entry_observation.json"
    assert cli.main([*arguments, "--continuation-receipt", str(receipt), "--physical-runs", str(physical),
        "--resume-prepared", str(recovery), "--resume-authority", str(authority), "--stage-terminal"]) == 0
    kwargs = owner.call_args.kwargs
    assert (kwargs["continuation_receipt"], kwargs["resume_prepared"], kwargs["resume_authority"]) == (
        receipt, recovery, authority)
    assert kwargs["physical_runs"] == [{"physical_run": "bound-by-owner"}]
    assert cli.main([*arguments, "--outcome", "INCONCLUSIVE", "--decision-ref", str(authority)]) == 0
    assert owner.call_args.kwargs["outcome"] == "INCONCLUSIVE"
    assert owner.call_args.kwargs["decision_ref"] == authority
    assert cli.main([*arguments, "--execution-root", str(tmp_path / "execution"),
                     "--physical-observation", str(observation)]) == 0
    assert owner.call_args.kwargs["physical_observation"] == observation
    capsys.readouterr()
    owner.return_value = {"status": "BLOCKED", "arms": {}, "executed": False,
        "strict_validation": {"status": "BLOCKED", "blockers": ["strict_reference_missing"],
            "arms": {"arm": {"native_qualification": {"status": "VERIFIED", "dependencies": ["large inventory"]}}}}}
    for extra, expected in [([], 2000), (["--bootstrap-samples", "0"], 0)]:
        assert cli.main([*arguments, "--strict", "--diagnose", *extra]) == 1
        assert owner.call_args.kwargs["n_bootstrap"] == expected
        assert owner.call_args.kwargs["strict_inputs"] == {}
        # Stdout keeps the strict verdict but not the large native-proof inventory.
        visible = json.loads(capsys.readouterr().out)["strict_validation"]
        assert visible["status"] == "BLOCKED"
        assert visible["arms"]["arm"]["native_qualification"] == {"status": "VERIFIED"}


@pytest.mark.parametrize("extra,stage,diagnose", [([], True, True), (["--locations", "reviewed.json"], False, True),
    (["--locations=reviewed.json"], False, True), (["--locations", "reviewed.json", "--stage-terminal"], True, True),
    (["--locations", "reviewed.json", "--strict"], False, False),
    (["--locations", "reviewed.json", "--strict-inputs=proofs.json"], False, False),
    (["--locations", "reviewed.json", "--strict", "--diagnose"], False, True),
    (["--locations", "reviewed.json", "--strict", "--stage-terminal"], True, True),
    (["--locations", "reviewed.json", "--strict", "--arm-outcome=reference=CLOSED",
      "--arm-decision-ref=reference=decision.md"], False, True)])
def test_acceptance_script_selects_checkout_and_analysis_defaults(monkeypatch, extra, stage, diagnose):
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location("local_accept_run", root / "platforms/local/accept_run.py")
    entry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entry)
    owner = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", owner)
    assert entry.main(["--run-id", "test", "--attempt-id", "v1", *extra]) == 0
    arguments = owner.call_args.args[0]
    assert arguments[0] == "accept-workflow"
    assert ("--diagnose" in arguments) is diagnose
    assert ("--stage-terminal" in arguments) is stage
    assert arguments[arguments.index("--repo-root") + 1] == str(root)
    assert "--execute" not in arguments


def test_closure_input_expands_to_per_arm_options_and_rejects_mixing(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location("local_accept_run_closure", root / "platforms/local/accept_run.py")
    entry = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entry)
    owner = Mock(return_value=0)
    monkeypatch.setattr(cli, "main", owner)
    closure = tmp_path / "closure.json"
    closure.write_text(json.dumps({"arms": {
        "reference": {"outcome": "CLOSED", "decision_ref": "experiments/q/reference.md"},
        "candidate": {"outcome": "NEGATIVE_UNDER_CONTRACT", "decision_ref": "experiments/q/candidate.md",
                      "facts_ref": "experiments/q/closure_facts_candidate.json"}},
        "role_audit": str(tmp_path / "roles.json")}), encoding="utf-8")
    (tmp_path / "roles.json").write_text("{}", encoding="utf-8")
    assert entry.main(["--run-id", "test", "--attempt-id", "v1", "--profile", str(tmp_path / "p.json"),
                       "--closure-input", str(closure)]) == 0
    arguments = owner.call_args.args[0]
    assert "--closure-input" not in arguments
    for flag, value in (("--arm-outcome", "reference=CLOSED"), ("--arm-outcome", "candidate=NEGATIVE_UNDER_CONTRACT"),
                        ("--arm-decision-ref", "candidate=experiments/q/candidate.md"),
                        ("--arm-facts-ref", "candidate=experiments/q/closure_facts_candidate.json"),
                        ("--role-audit", str(tmp_path / "roles.json"))):
        assert arguments[arguments.index(value) - 1] == flag
    assert arguments.count("--arm-facts-ref") == 1
    options = Mock(return_value={"status": "MECHANICALLY_VERIFIED"})
    monkeypatch.setattr(runs, "accept_run_workflow", options)
    assert cli._dispatch(cli._parser().parse_args(arguments))[1] == 0
    assert options.call_args.kwargs["facts_ref"] == {"candidate": "experiments/q/closure_facts_candidate.json"}
    assert "--stage-terminal" in arguments
    owner.reset_mock()
    closure.write_text(json.dumps({"arms": {"reference": {"outcome": "CLOSED"}}}), encoding="utf-8")
    for extra in ([], ["--outcome", "CLOSED"]):
        assert entry.main(["--run-id", "test", "--attempt-id", "v1", "--closure-input", str(closure), *extra]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "ERROR"
    owner.assert_not_called()


@pytest.mark.parametrize("extra", [["--control", "unused"], ["--threshold-ev", "0.1"]])
def test_run_diagnosis_rejects_ambiguous_or_unfrozen_policy(extra, tmp_path, capsys):
    assert cli.main(["diagnose", "--run-id", "test", "--attempt-id", "v1",
        "--profile", str(tmp_path / "machine.json"), *extra]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "ERROR"


def test_outcome_cannot_be_supplied_without_decision(tmp_path):
    with pytest.raises(ValueError, match="provided together"):
        runs.accept_run_workflow(None, tmp_path, "test", "v1", outcome="INCONCLUSIVE")


@pytest.mark.parametrize("read_only", [False, True])
def test_missing_strict_evidence_cannot_execute_terminal_closure(tmp_path, monkeypatch, read_only):
    from molgap import experiment_workflow as workflow
    from molgap import experiment_family_workflow as family
    spec = Mock(identity="f" * 64)
    spec.to_dict.return_value = {"prospective": {}}
    resolved = {"spec": spec, "options": {}, "outputs": {}, "contexts": {}, "locations": {}}
    monkeypatch.setattr("molgap.experiment_run_acceptance.resolve_run_acceptance_inputs", Mock(return_value=resolved))
    mechanical = Mock(return_value={"status": "MECHANICALLY_VERIFIED", "arms": {}})
    monkeypatch.setattr(workflow, "accept_workflow", mechanical)
    monkeypatch.setattr(runs, "inspect_run_strict_evidence", Mock(return_value={
        "status": "BLOCKED", "blockers": ["native_proof_dependencies_missing"]}))
    qualify = Mock(return_value={"locations": {}})
    monkeypatch.setattr(family, "qualify_same_run_terminal_packages", qualify)
    stage = Mock(return_value={"locations": {}, "blockers": []})
    monkeypatch.setattr("molgap.experiment_terminal_staging.stage_verified_terminal_packages",
        stage)
    result = runs.accept_run_workflow(None, tmp_path, "test", "v1", strict_inputs={},
        execute=not read_only, stage_terminal=not read_only, locations={} if read_only else None)
    assert result["status"] == "BLOCKED"
    assert result["executed"] is False
    assert mechanical.call_count == 1
    assert mechanical.call_args.kwargs["execute"] is False
    mechanical.reset_mock()
    with pytest.raises(ValueError, match="requires verified execution retention"):
        runs.accept_run_workflow(None, tmp_path, "test", "v1", locations={}, execute=True,
                                 physical_observation=tmp_path / "observation.json")
    mechanical.assert_not_called()
    with pytest.raises(ValueError, match="either physical_runs or one physical_observation"):
        runs.accept_run_workflow(None, tmp_path, "test", "v1", locations={}, execute=True,
                                 physical_runs=[], physical_observation=tmp_path / "observation.json")
    mechanical.assert_not_called()
    qualify.assert_not_called()
    if read_only:
        stage.assert_not_called()
        assert result["strict_mode"] == "CANONICAL_READ_ONLY"
        assert not list(tmp_path.iterdir())


def test_synchronous_native_cache_rechecks_bytes_without_reloading_proofs(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from molgap import family_training_runner as native
    from molgap.v4_runtime import normalized_source_sha256
    output = tmp_path / "output"
    output.mkdir()
    contract = output / "training_contract.json"
    contract.write_bytes(b"{\r\n  \"synthetic\": \"recipe identity\"\r\n}\r\n")
    proof = output / "proof.bin"
    proof.write_bytes(b"previously checked synthetic proof")
    context = SimpleNamespace(training_recipe_sha256=normalized_source_sha256(contract),
        to_dict=lambda: {"synthetic": "bound-context"})
    cached = {"status": "VERIFIED", "blockers": [], "dependencies": [{"local_path": str(proof),
        "verified_sha256": hashlib.sha256(proof.read_bytes()).hexdigest()}]}
    resolved = {"outputs": {"arm": {"output_dir": str(output)}}, "contexts": {"arm": context},
        "locations": {"arm": {"trajectory": "experiments/missing/trajectory.json", "trajectory_id": "T-missing"}}}
    verifier = Mock(side_effect=AssertionError("CPU tensor verification must be reused within the transaction"))
    monkeypatch.setattr(native, "inspect_retained_native_qualification", verifier)
    kwargs = {"_native_reports": {"arm": (cached, context.to_dict())}, "require_finalized": False}
    first = runs.inspect_run_strict_evidence(tmp_path, resolved, {}, **kwargs)
    assert first["arms"]["arm"]["native_qualification"]["status"] == "VERIFIED"
    proof.write_bytes(b"changed after verification")
    changed = runs.inspect_run_strict_evidence(tmp_path, resolved, {}, **kwargs)
    assert "native_qualification_invalid" in changed["arms"]["arm"]["blockers"]
    assert "dependency changed" in changed["arms"]["arm"]["native_qualification"]["error"]
    verifier.assert_not_called()
