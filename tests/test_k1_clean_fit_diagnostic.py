import pytest
import torch
import json
from molgap.k1_clean_fit_diagnostic import retained_buffers, validate_prediction
from molgap.v4_runtime import state_dict_sha256


def test_retained_bn_buffers_restore_parameters_and_buffers_on_error():
    model = torch.nn.BatchNorm1d(2).eval().requires_grad_(False)
    before = state_dict_sha256(model.state_dict())
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    buffers["running_mean"].fill_(3)
    with pytest.raises(RuntimeError, match="synthetic"):
        with retained_buffers(model, buffers, expected_sha256=state_dict_sha256(buffers)):
            assert torch.equal(model.running_mean, torch.full((2,), 3.0))
            raise RuntimeError("synthetic")
    assert state_dict_sha256(model.state_dict()) == before


def test_retained_state_hash_failclosed_and_non_bn_unchanged():
    model = torch.nn.BatchNorm1d(2).eval().requires_grad_(False)
    model.register_buffer("other", torch.ones(1))
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    buffers["other"].fill_(2)
    with pytest.raises(ValueError, match="matching BN-only"):
        with retained_buffers(model, buffers, expected_sha256=state_dict_sha256(buffers)):
            pytest.fail("changed non-BN buffer")


@pytest.mark.parametrize("change", [None, "rows", "target", "nonfinite", "shape"])
def test_prediction_identity(change):
    record = {"source_idx": torch.tensor([8, 3]), "target_eV": torch.zeros(2), "prediction_eV": torch.ones(2)}
    target = record["target_eV"].clone()
    if change == "rows":
        record["source_idx"] = torch.tensor([3, 8])
    elif change == "target":
        record["target_eV"][0] = 1
    elif change == "nonfinite":
        record["prediction_eV"][0] = float("nan")
    elif change == "shape":
        record["prediction_eV"] = torch.ones(2, 1)
    if change is None:
        validate_prediction(record, torch.tensor([8, 3]), target=target)
    else:
        with pytest.raises(ValueError):
            validate_prediction(record, torch.tensor([8, 3]), target=target)


@pytest.mark.parametrize("mutation", [None, "cost", "source", "roles"])
def test_clean_fit_metadata_closure_cost_and_roles(tmp_path, monkeypatch, mutation):
    from molgap import frozen_diagnostic_closure as closure
    from molgap.training_reproducibility import sha256_file

    experiment = tmp_path / "experiment"
    (experiment / "results").mkdir(parents=True)
    (experiment / "rml").mkdir()
    for relative in ("src/molgap/frozen_diagnostic_closure.py",
                     "research_memory/policies/synthetic.1.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    def write(name, value):
        (experiment / name).write_text(json.dumps(value), encoding="utf-8")
    write("inputs.json", {"cpu_threads": 4, "train_source_idx": [8, 3],
                          "development_source_idx": [500000],
                          "train_decoded_source_bounds": [0, 500000],
                          "executed_source_files": {"src/molgap/frozen_diagnostic_closure.py":
                              "0" * 64 if mutation == "source" else sha256_file(
                                  tmp_path / "src/molgap/frozen_diagnostic_closure.py")},
                          "ceiling_seconds": 1200})
    write("rml/trajectory.json", {"trajectory_id": "synthetic", "actions": [
        {"action_id": "diagnostic", "run_ids": ["synthetic-run"]}]})
    write("results/result.json", {"status": "complete", "checks": {"synthetic": True},
        "runtime": {"device": "cpu", "cpu_threads": 4},
        "optimizer_created": False, "gradients_computed": False, "training_executed": False,
        "inputs_sha256": sha256_file(experiment / "inputs.json"),
        "prospective_sha256": sha256_file(experiment / "rml/trajectory.json"),
        "format": "molgap-k1-clean-fit-diagnostic-v1", "wall_seconds": -1 if mutation == "cost" else 2,
        "process_cpu_seconds": 6})
    write("results/completion.json", {"complete": True,
        "inputs_sha256": sha256_file(experiment / "inputs.json"),
        "files": {"result.json": sha256_file(experiment / "results/result.json")}})
    source = tmp_path / "analysis_source.py"
    source.write_text("# synthetic metadata only\n", encoding="utf-8")
    write("analysis.json", {"analysis_source_sha256": {str(source):
        "0" * 64 if mutation == "source" else sha256_file(source)},
        "wall_seconds": -1 if mutation == "cost" else 1, "process_cpu_seconds": 3})
    for name in ("terminal_decision.md", "attribution.md"):
        (experiment / name).write_text("Synthetic NO_TRAIN\n", encoding="utf-8")
    roles = {"train_decoded": {"labels_read": list(range(500000))},
        "train_descriptive": {"prediction_input": [8, 3], "metric_computed": [8, 3]},
        "internal_development": {a: [500000] for a in
            ("labels_read", "prediction_input", "metric_computed", "selection_used")}}
    if mutation == "roles":
        roles["train_descriptive"]["metric_computed"] = [3, 8]
    monkeypatch.setattr(closure, "finalize", lambda *args: {"synthetic_finalizer": True})
    if mutation:
        with pytest.raises(ValueError):
            closure.close_local_diagnostic(tmp_path, experiment, evidence_id="synthetic",
                policy_id="synthetic", role_rows=roles)
    else:
        closure.close_local_diagnostic(tmp_path, experiment, evidence_id="synthetic",
            policy_id="synthetic", role_rows=roles)
        accepted = json.loads((experiment / "acceptance.json").read_text())
        assert len(accepted["costs"]) == 1
        assert accepted["costs"][0]["measurement"]["wall_hours"]["value"] == 2 / 3600
        assert accepted["costs"][0]["measurement"]["cpu_hours"]["value"] == 6 / 3600
        assert accepted["comparison_class"] == "CONTEXT_ONLY"
        assert accepted["trajectory_decision"]["outcome"] == "NO_TRAIN"
