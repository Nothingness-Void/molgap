"""Static dispatch completion capability; synthetic CPU files, no model roles."""
import json
from types import SimpleNamespace

import pytest
import torch

from molgap import experiment_execution as execution
from molgap import experiment_family_workflow as family
from molgap.training_reproducibility import sha256_file
from molgap.v4_runtime import state_dict_sha256, normalized_source_sha256


@pytest.mark.parametrize("family_id,version", [("neural_atom_k1", "2"), ("gptrans_t", "1"), ("neural_atom_k1", "3")])
@pytest.mark.parametrize("phase", ["train", "preflight"])
@pytest.mark.parametrize("separate_roots", [False, True])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_registered_completion_binding_is_used_once(tmp_path, monkeypatch, family_id, version, phase, separate_roots, newline):
    recipe = tmp_path / "recipe.json"
    source = json.dumps({"acceptance_requirements": {"synthetic": True}}, indent=2).encode() + b"\n"
    recipe.write_bytes(source.replace(b"\n", newline))
    if newline == b"\r\n":
        assert sha256_file(recipe) != normalized_source_sha256(recipe)
    staged = tmp_path / "staged" if separate_roots else tmp_path
    staged.mkdir(exist_ok=True)
    initial = staged / "initial.pt"
    state = {"weight": torch.ones(2)}
    torch.save(state, initial)
    arm = {"arm_id": "reference", "family": {"name": family_id, "version": version},
           "addons": [], "initialization": {"state_sha256": state_dict_sha256(state)}}
    spec = SimpleNamespace(to_dict=lambda: {"arms": [arm]})
    context = SimpleNamespace(source_commit="a" * 40, training_recipe_sha256=normalized_source_sha256(recipe))
    monkeypatch.setattr(family.RunContext, "for_training", lambda *args, **kwargs: context)
    monkeypatch.setattr(execution, "validate_staged_trajectory", lambda *args, **kwargs:
                        {"state_at_start": {"source_commit": context.source_commit}})
    calls, inspections = [], []
    result = {"status": "MECHANICALLY_VERIFIED" if phase == "train" else "accepted"}

    def runner(**kwargs):
        calls.append(kwargs)
        return result

    def inspect_output(path, **kwargs):
        inspections.append((path, kwargs))
        return result

    monkeypatch.setattr(execution, "import_module", lambda name:
                        SimpleNamespace(run_screen_preflight=runner, run_screen_arm=runner))
    monkeypatch.setattr(family, "inspect_output", inspect_output)
    binding = object()
    output = tmp_path / "output"
    options = dict(spec=spec,
        job={"arm_id": "reference", "recipe": "recipe.json", "initial_state": "initial.pt",
             "trajectory_id": "synthetic"}, phase=phase, source_root=tmp_path,
        package_dir=tmp_path, expected_package_identity="b" * 64, input_root=tmp_path,
        output=output, account="synthetic", run_reference="synthetic/kernel",
        staged_root=staged, prospective_sha256="c" * 64, target_identity=binding)
    if version == "3" and phase == "train":
        options["pause_after_step"] = 3
    if separate_roots and family_id == "gptrans_t":
        with pytest.raises(ValueError, match="Separate prospective/data roots"):
            execution.execute_training_phase(**options)
        assert calls == inspections == []
        return
    observed = execution.execute_training_phase(**options)
    assert observed is result and len(calls) == 1
    if version == "3" and phase == "train":
        assert calls[0]["pause_after_step"] == 3
    if separate_roots:
        assert calls[0]["prospective_root"] == staged
    if phase == "preflight":
        assert "target_identity" not in calls[0] and inspections == []
    elif family_id == "neural_atom_k1":
        assert calls[0]["target_identity"] is binding and inspections == []
    else:
        assert "target_identity" not in calls[0] and len(inspections) == 1
        assert inspections[0][0] == output / "normalized"
        assert inspections[0][1]["target_identity"] is binding
