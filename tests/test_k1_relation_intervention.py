"""Static and synthetic-tensor tests; never instantiate a MolGap model."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from molgap.k1_relation_intervention import (
    VARIANTS, check_variant, group_masks, intervention, joined_payload,
)


@pytest.mark.parametrize("mode", list(VARIANTS))
def test_frozen_variants(mode):
    assert VARIANTS[mode][:3] == ("full", "half", "off")
    for variant in VARIANTS[mode]:
        check_variant(mode, variant)


@pytest.mark.parametrize("mode,variant", [("fake", "off"), (next(iter(VARIANTS)), "rrwp_off"),
                                          (next(iter(VARIANTS)), "quarter")])
def test_no_undeclared_search(mode, variant):
    with pytest.raises(ValueError):
        check_variant(mode, variant)


def test_groups_are_fixed_exhaustive_disjoint():
    values = np.array([0, 0.27272728085517883, .5, 0.7333333492279053, 1])
    masks = group_masks(values)
    assert masks["low_conjugation"].tolist() == [True, False, False, False, False]
    assert masks["high_conjugation"].tolist() == [False, False, False, True, True]
    assert np.all(sum(masks[k].astype(int) for k in masks if k != "all") == 1)
    with pytest.raises(ValueError):
        group_masks([float("nan")])


def toy_addon():
    return SimpleNamespace(forward=lambda *args: "original",
        _triplet_update=lambda *args: "original_triplet",
        compute_update=lambda hidden, batch, edge: (torch.ones_like(hidden)*2, {}))


@pytest.mark.parametrize("variant,expected", [("off", 3.), ("half", 4.)])
def test_scale_only_and_exception_restore(variant, expected):
    addon = toy_addon()
    previous = addon.forward
    hidden = torch.ones(3, 2)*3
    with pytest.raises(RuntimeError):
        with intervention(addon, next(iter(VARIANTS)), variant, []):
            assert torch.equal(addon.forward(hidden, torch.zeros(3, dtype=torch.long)),
                               torch.full_like(hidden, expected))
            raise RuntimeError("synthetic interruption")
    assert addon.forward is previous


def test_common_return_preserves_mean_not_receiver_difference():
    addon = toy_addon()
    update = torch.tensor([[1.,2.], [3.,4.], [7.,9.]])
    addon.compute_update = lambda *args: (update.clone(), {})
    with intervention(addon, next(iter(VARIANTS)), "common_return", []):
        result = addon.forward(torch.zeros_like(update), torch.tensor([0,0,1]))
    assert torch.equal(result, torch.tensor([[2.,3.], [2.,3.], [7.,9.]]))


def test_triplet_removal_retains_normalizer_and_restores():
    addon = toy_addon()
    addon.triplet_norm = lambda x: x + 7
    original = addon._triplet_update
    with intervention(addon, "neural_atom_k1_triplet_aggregate", "triplet_off", []):
        result = addon._triplet_update(torch.ones(1,2,2,3), torch.tensor([[[True,False],[True,True]]]), None)
        assert result[0][0,0,0].tolist() == [8.,8.,8.]
        assert result[0][0,0,1].tolist() == [0.,0.,0.]
        assert result[1:] == (None,None,None)
    assert addon._triplet_update is original


def test_rrwp_hook_zeroes_contribution_and_removes_hook():
    addon = toy_addon()
    state = {}
    def register(callback):
        state["callback"] = callback
        return SimpleNamespace(remove=lambda: state.update(removed=True))
    addon.rrwp_projection = SimpleNamespace(register_forward_hook=register)
    with intervention(addon, "neural_atom_k1_rrwp_pair", "rrwp_off", []):
        assert torch.equal(state["callback"](None, None, torch.ones(2)), torch.zeros(2))
    assert state["removed"] is True


def test_payload_accepts_nested_metadata_and_rejects_tamper(tmp_path):
    chunks = []
    for index in range(10):
        path = tmp_path / f"chunk_{index:02d}.pt"
        row = {"source_idx": torch.arange(100000+5000*index,105000+5000*index),
            "target_eV": torch.zeros(5000), "prediction_eV": torch.ones(5000),
            "descriptors": {"conjugated_bond_fraction": torch.zeros(5000)},
            "diagnostics": {"update_rms": torch.ones(5000)}}
        torch.save(row, path)
        chunks.append({"file": path.name, "rows": 5000, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = joined_payload(tmp_path, chunks, 100000)
    assert result["descriptors"]["conjugated_bond_fraction"].shape == (50000,)
    assert result["diagnostics"]["update_rms"].sum() == 50000
    chunks[0]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="SHA"):
        joined_payload(tmp_path, chunks, 100000)


def test_no_training_and_private_two_gpu_metadata():
    root = Path(__file__).resolve().parents[1]
    for name in ("k1_relation_intervention.py", "k1_relation_intervention_records.py"):
        tree = ast.parse((root / "src/molgap" / name).read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
        assert not calls.intersection({"backward", "step", "AdamW", "kernels_push", "train"})
        if name.endswith("_records.py"):
            assert not calls.intersection({"_model", "_infer", "make_encoder", "run_worker"})
    metadata = json.loads((root / "experiments/pcqm_k1_relation_resolution_100k/kaggle_diagnostic/kernel-metadata.json").read_text())
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] is True
    assert not metadata["kernel_sources"] and not metadata["competition_sources"]
    assert all("full" not in item and "1m" not in item for item in metadata["dataset_sources"])


def test_submission_binding_preserves_actual_and_logical_identity(tmp_path, monkeypatch):
    from molgap import k1_relation_intervention_records as records
    base = tmp_path / records.BASE
    diagnostic = base / "diagnostic"
    store = tmp_path / "records"
    monkeypatch.setattr(records, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(records, "ROOT", diagnostic)
    monkeypatch.setattr(records, "RECORDS", store)
    monkeypatch.setattr(records, "REMOTE", store / "pcqm_k1_relation_diagnostic")
    actual = "kaseichou/title-derived-diagnostic"
    response = {"ref": "/code/"+actual, "kernelId": 123, "versionNumber": 1, "error": None}
    metadata = {"id": actual, "id_no": 123, "is_private": True,
                "dataset_sources": ["kaseichou/fixed-data"], "code_file": "remote.py"}
    package = base / "kaggle_diagnostic"
    package.mkdir(parents=True)
    entry = package / "run.py"
    entry.write_text("# synthetic entry\n")
    remote_entry = store / "remote_metadata/remote.py"
    remote_entry.parent.mkdir(parents=True)
    remote_entry.write_bytes(entry.read_bytes())
    records.save(diagnostic / "release.json", {"run_id": records.RUN, "training_authorized": False,
        "entry_sha256": records.file_digest(entry), "input_archive_sha256": "a"*64,
        "model_source_commit": "b"*40, "helper_commit": "c"*40, "source_archive_sha256": "d"*64,
        "max_allocated_device_seconds": 5400})
    records.save(store / "kernel_push.json", {"status": "PUSH_RETURNED", "response_text": json.dumps(response)})
    records.save(store / "confirmed_status.json", {"response": response, "kernel": actual,
        "kernel_id": 123, "version": 1, "status_text": '{"status":"RUNNING"}'})
    records.save(store / "remote_metadata/kernel-metadata.json", metadata)
    records.save(package / "kernel-metadata.json", metadata)
    records.save(base / "audit_monitor_binding.json", {"format": "molgap-server-multi-job-monitor-v1",
        "owner": "server", "working_directory": str(tmp_path), "python": "python", "credential_file": "unread.json",
        "credential_owner": "kaseichou", "controller_thread_id": "controller", "monitor_thread_id": "monitor",
        "healthy_action": "SILENT", "terminal_action": "handoff"})
    result = records.bind_submission()
    assert result["job"]["run_id"] == actual+":v1"
    assert result["job"]["experiment_run_id"] == records.RUN
    assert records.load(Path(result["receipt"]))["logical_run_id"] == records.RUN
    with pytest.raises(FileExistsError):
        records.bind_submission()


@pytest.mark.parametrize("bad_receipt", [False, True])
def test_terminal_reuses_no_train_translation_with_bound_receipt(tmp_path, monkeypatch, bad_receipt):
    from molgap import k1_relation_intervention_records as records
    from molgap import k1_relation_audit_records as shared
    root = tmp_path / records.REL
    monkeypatch.setattr(records, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(records, "ROOT", root)
    monkeypatch.setattr(records, "accept_and_analyze", lambda: {"accepted": True, "run_id": records.RUN})
    records.save(root / "release.json", {"run_id": records.RUN})
    records.save(root / "rml_plan/trajectory.json", {"trajectory_id": records.TID})
    records.save(root / "submission_receipt_v1.json", {"logical_run_id": records.RUN, "version": 1,
        "submission_confirmed": True, "release_sha256": "0"*64 if bad_receipt else records.file_digest(root / "release.json")})
    captured = {}
    def translate(**kwargs):
        captured.update(kwargs)
        return "translated"
    monkeypatch.setattr(shared, "prepare_no_train_terminal", translate)
    if bad_receipt:
        with pytest.raises(ValueError, match="receipt"):
            records.prepare_terminal("2026-09-27T08:30:00Z")
        assert not captured
    else:
        assert records.prepare_terminal("2026-09-27T08:30:00Z") == "translated"
        assert captured["run_id"] == records.RUN
        assert captured["evidence_id"] == records.EID
        assert captured["outcome"]["full_handoff_status"] == "not_authorized"
        assert captured["outcome"]["comparison_status"] == "paired_endpoint_diagnostic"
        assert f"{records.REL}/submission_receipt_v1.json" in captured["artifact_refs"]
