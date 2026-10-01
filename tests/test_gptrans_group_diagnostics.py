from copy import deepcopy
import inspect
import pytest
from molgap.gptrans_endpoint_paths import parameter_groups_fingerprint
from molgap.gptrans_author_acceptance import _selected_completed_arms


def test_group_inventory_digest_accepts_actual_list_shape():
    inventory = [{"name": "embedding.weight", "shape": [8, 4], "weight_decay": .05},
                 {"name": "norm.bias", "shape": [4], "weight_decay": 0.0}]
    digest = parameter_groups_fingerprint(inventory)
    assert len(digest) == 64 and digest == parameter_groups_fingerprint(deepcopy(inventory))
    inventory[1]["weight_decay"] = .05
    assert parameter_groups_fingerprint(inventory) != digest


def test_trainer_and_acceptance_share_inventory_digest():
    from molgap.pcqm_gptrans_v4 import run_training
    from molgap.gptrans_author_acceptance import accept_training_outputs
    for function in (run_training, accept_training_outputs):
        source = inspect.getsource(function)
        assert "parameter_groups_fingerprint(" in source
        assert "canonical_fingerprint(inventory)" not in source


def test_partial_acceptance_does_not_adopt_failed_worker():
    outcomes = [{"variant": "failed", "complete": False}, {"variant": "done", "complete": True}]
    assert _selected_completed_arms({"failed", "done"}, outcomes, ["done"]) == {"done"}
    for selected in (None, ["failed"], ["not-declared"], []):
        with pytest.raises(ValueError):
            _selected_completed_arms({"failed", "done"}, outcomes, selected)


def test_duplicate_worker_outcomes_are_rejected():
    with pytest.raises(ValueError, match="identity"):
        _selected_completed_arms({"a", "b"}, [{"variant": "a", "complete": True}] * 2, ["a"])


def test_failure_closure_never_relabels_accepted_companion(tmp_path, monkeypatch):
    import json
    from molgap.gptrans_author_terminal import close_author_failed_arm
    base = tmp_path / "experiments/example/gpu"
    base.mkdir(parents=True)
    acceptance = base / "acceptance.json"
    acceptance.write_text(json.dumps({"accepted": True, "acceptance_scope": "selected_completed_arms", "arms": {"done": {}}}))
    (base / "screen_config.json").write_text(json.dumps({"arms": {"done": {}, "failed": {}}}))
    monkeypatch.setattr("molgap.research_memory.pipeline.finalize_rebuild_backtest",
                        lambda *a, **k: pytest.fail("Completed companion must not finalize as failed"))
    with pytest.raises(ValueError, match="partial run identity"):
        close_author_failed_arm(tmp_path, tmp_path, acceptance, experiment_ref="experiments/example", mode="done")
