"""Digest-bound output inspection hand-off tests."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

pytest_plugins = ["test_experiment_family_workflow"]

import molgap.experiment_family_workflow as family_workflow
from molgap.experiment_family_workflow import (
    build_verified_terminal_descriptor,
    capture_output_inspection,
    close_verified_outputs,
)
from test_experiment_family_workflow import _closure_descriptor


def _locations(descriptor):
    return {
        arm["arm_id"]: {
            "trajectory_id": arm["trajectory_id"],
            "run_id": arm["run_id"],
            "trajectory": arm["trajectory"],
            "terminal": arm["terminal"],
            "trace": arm["trace"],
        }
        for arm in descriptor.to_dict()["arms"]
    }


def _snapshots(outputs):
    return {
        arm_id: capture_output_inspection(
            item["output_dir"], context=item["context"], expected=item["expected"]
        )
        for arm_id, item in outputs.items()
    }


def test_descriptor_and_close_reuse_one_snapshot_without_tensor_reads(
    tmp_path, repo, launch_contexts, monkeypatch
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    locations = _locations(descriptor)
    snapshots = _snapshots(outputs)

    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("the owner inspector must not run during snapshot reuse")

    def unexpected_tensor_load(*args, **kwargs):
        raise AssertionError("snapshot reuse must not deserialize tensors")

    monkeypatch.setattr(family_workflow, "inspect_output", unexpected_inspection)
    monkeypatch.setattr(torch, "load", unexpected_tensor_load)

    rebuilt = build_verified_terminal_descriptor(
        repo, spec, outputs=outputs, locations=locations, inspections=snapshots
    )
    result = close_verified_outputs(
        repo, spec, rebuilt, outputs=outputs, inspections=snapshots
    )

    assert result["status"] == "MECHANICALLY_VERIFIED"
    assert result["executed"] is False
    assert all(report["status"] == "MECHANICALLY_VERIFIED"
               for report in result["arms"].values())


def test_capture_runs_owner_inspection_once_per_arm_and_closure_only_rehashes(
    tmp_path, repo, launch_contexts, monkeypatch
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    locations = _locations(descriptor)
    calls = {"inspect": 0, "load": 0}
    original_inspect = family_workflow.inspect_output
    original_load = torch.load

    def counted_inspect(*args, **kwargs):
        calls["inspect"] += 1
        return original_inspect(*args, **kwargs)

    def counted_load(*args, **kwargs):
        calls["load"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(family_workflow, "inspect_output", counted_inspect)
    monkeypatch.setattr(torch, "load", counted_load)
    snapshots = _snapshots(outputs)

    assert calls["inspect"] == len(outputs)
    # Each retained output has predictions, selected_model, and resume tensors.
    assert calls["load"] == len(outputs) * 3

    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("descriptor/closure must reuse the captured snapshot")

    def unexpected_tensor_load(*args, **kwargs):
        raise AssertionError("descriptor/closure must not deserialize tensors")

    monkeypatch.setattr(family_workflow, "inspect_output", unexpected_inspection)
    monkeypatch.setattr(torch, "load", unexpected_tensor_load)
    rebuilt = build_verified_terminal_descriptor(
        repo, spec, outputs=outputs, locations=locations, inspections=snapshots
    )
    result = close_verified_outputs(
        repo, spec, rebuilt, outputs=outputs, inspections=snapshots
    )
    assert result["status"] == "MECHANICALLY_VERIFIED"


def test_stale_snapshot_is_rejected_before_descriptor_translation(
    tmp_path, repo, launch_contexts
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    locations = _locations(descriptor)
    snapshots = _snapshots(outputs)
    target = outputs["gptrans_t"]["output_dir"] / "predictions.pt"
    with target.open("ab") as stream:
        stream.write(b"stale")

    with pytest.raises(ValueError, match="snapshot is stale: artifact hash mismatch: predictions"):
        build_verified_terminal_descriptor(
            repo, spec, outputs=outputs, locations=locations, inspections=snapshots
        )


def test_snapshot_context_and_expected_are_bound_to_original_inspection(
    tmp_path, repo, launch_contexts
):
    spec, descriptor, outputs = _closure_descriptor(tmp_path, repo, launch_contexts)
    locations = _locations(descriptor)
    snapshots = _snapshots(outputs)

    context_item = outputs["gptrans_t"]
    context_item["context"] = replace(
        context_item["context"], run_reference="synthetic-account/changed"
    )
    with pytest.raises(ValueError, match="snapshot/context mismatch"):
        build_verified_terminal_descriptor(
            repo, spec, outputs=outputs, locations=locations, inspections=snapshots
        )

    outputs = dict(outputs)
    outputs["gptrans_t"] = dict(outputs["gptrans_t"])
    outputs["gptrans_t"]["context"] = launch_contexts[4]["gptrans_t"]
    outputs["gptrans_t"]["expected"] = copy.deepcopy(outputs["gptrans_t"]["expected"])
    outputs["gptrans_t"]["expected"]["epochs"] = 1
    with pytest.raises(ValueError, match="snapshot/expected mismatch"):
        build_verified_terminal_descriptor(
            repo, spec, outputs=outputs, locations=locations, inspections=snapshots
        )
