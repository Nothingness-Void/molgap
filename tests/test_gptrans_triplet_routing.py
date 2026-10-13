"""Synthetic tensor/metadata probes only; no molecular forward or training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from molgap import gptrans_triplet_routing as routing
from molgap.gptrans_triplet_communication import aggregate_direction


@pytest.mark.parametrize("outward", [False, True])
def test_supported_direction_matches_explicit_conditioning(outward):
    g = torch.Generator().manual_seed(17)
    value = torch.randn(1, 4, 4, 2, 8, generator=g, requires_grad=True)
    bias = torch.randn(1, 4, 4, 2, generator=g, requires_grad=True)
    gate = torch.randn(1, 4, 4, 2, generator=g, requires_grad=True)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    support = torch.eye(4, dtype=torch.bool)[None].clone()
    support[:, 0, :] = support[:, :, 0] = True
    actual = routing.bond_direction(value, bias, gate, valid, support, outward=outward)
    expected = torch.zeros_like(actual)
    for i in range(4):
        for j in range(4):
            logits, gates = (bias[0, :, i], gate[0, :, i]) if outward else (bias[0, i], gate[0, i])
            mask = support[0, :, j] if outward else support[0, j]
            weights = logits.masked_fill(~mask[:, None], -torch.inf).softmax(0) * gates.sigmoid()
            values = value[0, :, j] if outward else value[0, j]
            expected[0, i, j] = (weights[..., None] * values).sum(0)
    torch.testing.assert_close(actual, expected)
    full = routing.bond_direction(value, bias, gate, valid, valid, outward=outward)
    torch.testing.assert_close(full, aggregate_direction(value, bias, gate, valid, outward=outward))
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (value, bias, gate))
    permutation = torch.tensor([0, 2, 1, 3])
    perm = lambda t: t[:, permutation][:, :, permutation]
    torch.testing.assert_close(routing.bond_direction(perm(value), perm(bias), perm(gate), perm(valid), perm(support), outward=outward), perm(actual))
    empty = torch.zeros_like(valid)
    assert not routing.bond_direction(value, bias, gate, empty, empty, outward=outward).any()


def test_bond_support_real_self_virtual_and_padding():
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[:, 3] = valid[:, :, 3] = False
    edges = (torch.tensor([0, 0]), torch.tensor([1, 2]), torch.tensor([2, 1]))
    support = routing.bond_support(valid, edges)
    assert support[0, 1, 2] and support[0, 2, 1] and support[0, 0, 2]
    assert not support[:, 3].any() and not support[:, :, 3].any()


@pytest.mark.parametrize("mode", routing.MODES)
def test_registration_and_frozen_tensor_qualification(mode, tmp_path, monkeypatch):
    from molgap.experiment_spec import ADDONS
    from molgap.pcqm_gptrans_v4 import capacity_module, _ema_decay
    assert ADDONS[(mode, "1")].single_addon
    assert capacity_module(mode) is routing and _ema_decay(mode) == .999
    from molgap.constants import REPO_ROOT
    old = "degree_local_triplet_product_ema999" if mode == routing.MODES[0] else "degree_local_triplet_aggregate_ema999"
    if not (REPO_ROOT / "platforms/_records/kaggle/initializations" / (old + "_initial.pt")).exists():
        pytest.skip("Retained initial record is not installed on this checkout")
    path = tmp_path / "initial.pt"
    routing.freeze_initial(mode, None, path)
    monkeypatch.setattr(routing.product, "_extras", lambda *a: (_ for _ in ()).throw(AssertionError("Do not regenerate random tensors during qualification")))
    facts = routing.read_initial_facts(mode, path)
    assert all(facts["qualification_checks"].values()) and facts["parameters"] == routing.PARAMETERS[mode]
    payload = torch.load(path, weights_only=True)
    key = next(iter(payload["model_state"]))
    payload["model_state"][key].view(-1)[0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="tensor_pin"):
        routing.read_initial_facts(mode, path)


def test_freeze_has_no_model_execution():
    tree = ast.parse(Path(routing.__file__).read_text())
    freeze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "freeze_initial")
    assert "construct(" not in ast.unparse(freeze) and "load_initial(" not in ast.unparse(freeze)


def test_failed_preflight_prevents_all_formal_training(tmp_path, monkeypatch):
    from molgap import gptrans_author_screen as screen
    from molgap import gptrans_kaggle_runtime as scheduler
    source = tmp_path / "source"
    source.mkdir()
    config = {"arms": {m: {} for m in routing.MODES}, "dataset_manifest_sha256": "x",
        "target_transform_sha256": "y", "maximum_wall_seconds": 600,
        "all_arms_preflight_barrier": True}
    (source / "screen.json").write_text(json.dumps(config))
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(screen, "restore_source_package", lambda *a: {"package_identity": "p", "spec_identity": "s", "source_commit": "c"})
    monkeypatch.setattr(screen, "mounted", lambda *a: tmp_path / "manifest.json")
    monkeypatch.setattr(screen, "sha256_file", lambda *a: "y")
    monkeypatch.setattr(screen.subprocess, "check_call", lambda *a, **k: None)
    monkeypatch.setattr(screen.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="GPU0 T4\nGPU1 T4\n"))
    monkeypatch.setenv("MOLGAP_SCREEN_ROOT", "test")
    monkeypatch.setenv("MOLGAP_SCREEN_MODES", "[]")
    monkeypatch.setattr(scheduler, "ROOT", tmp_path)
    calls = []
    def worker(arm, device, context, deadline, *, stages):
        calls.append(stages)
        if arm == routing.MODES[0]:
            raise ValueError("synthetic qualification failure")
        return {"variant": arm, "complete": True}
    monkeypatch.setattr(scheduler, "worker", worker)
    with pytest.raises(RuntimeError, match="Retain independent"):
        screen.run_author_screen(tmp_path, source, output, "sha", config_ref="screen.json")
    assert calls == [("preflight",), ("preflight",)]
    assert (output / "preflight_phase.json").exists() and not (output / "training_phase.json").exists()
