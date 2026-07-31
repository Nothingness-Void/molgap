"""Contract tests for the public repaired-2M pure-2D inference path.

These run without the real checkpoints: the registry entries are patched to tiny
synthetic weights so the loader, expert ordering, gate averaging, and
invalid-SMILES behavior are covered on any machine. The accepted-metric
reproduction lives in
`production/04_evaluate/scripts/evaluation/verify_repaired_2m_public_inference.py`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from molgap import constants
from molgap.gps import GPSWrapper
from molgap.multi2d_router_fusion import EXPERTS, GateTrainingConfig, TARGETS


pytest.importorskip("torch_geometric")


def _tiny_gps(num_layers: int, hidden_channels: int) -> dict:
    return {
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "num_heads": 2,
        "dropout": 0.0,
    }


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Register small synthetic repaired-2M presets in place of the real ones."""
    torch.manual_seed(0)
    entries = {}
    for index, (name, layers, hidden) in enumerate(
        (("gps7", 2, 32), ("gps9", 3, 32), ("gps11_160", 2, 16))
    ):
        params = _tiny_gps(layers, hidden)
        path = tmp_path / f"{name}.pt"
        model = GPSWrapper(**params)
        # A distinct constant bias per expert makes expert identity and ordering
        # observable in the returned predictions.
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.fill_(float(index + 1))
        torch.save(model.state_dict(), path)
        entries[f"repaired_2m_{name}"] = {
            "kind": "gps",
            "checkpoint": path,
            "params": params,
            "normalized": False,
        }

    from molgap.multi2d_router_fusion import DenseSoftGate

    # dense_gate_features emits 27 columns for three experts by three targets.
    input_dim = 27
    gate_paths = []
    for seed in (42, 43):
        gate_path = tmp_path / f"dense_seed{seed}.pt"
        config = GateTrainingConfig(hidden_channels=8)
        gate = DenseSoftGate(
            input_dim,
            np.zeros(input_dim, dtype=np.float32),
            np.ones(input_dim, dtype=np.float32),
            config,
        )
        torch.save(
            {
                "kind": "three_gps_dense_soft_gate",
                "experts": EXPERTS,
                "targets": TARGETS,
                "config": {**config.__dict__, "seed": seed},
                "state_dict": gate.state_dict(),
            },
            gate_path,
        )
        gate_paths.append(gate_path)

    entries["repaired_2m_dense_2d"] = {
        "kind": "multi2d_dense",
        "normalized": False,
        "experts": [
            "repaired_2m_gps7",
            "repaired_2m_gps9",
            "repaired_2m_gps11_160",
        ],
        "gates": gate_paths,
        "encoder_passes": 3,
    }
    entries["repaired_2m_equal_2d"] = {
        "kind": "multi2d_equal",
        "normalized": False,
        "experts": ["repaired_2m_gps7", "repaired_2m_gps9"],
        "encoder_passes": 2,
    }
    patched = {**constants.MODEL_REGISTRY, **entries}
    monkeypatch.setattr(constants, "MODEL_REGISTRY", patched)
    from molgap import inference

    monkeypatch.setattr(inference, "MODEL_REGISTRY", patched)
    return patched


def test_dense_preset_loads_three_experts_and_all_gates(registry):
    from molgap.inference import load_repaired_2m_2d

    models = load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")
    assert len(models["experts"]) == 3
    assert len(models["gates"]) == 2
    assert models["encoder_passes"] == 3


def test_equal_preset_loads_two_experts_and_no_gate(registry):
    from molgap.inference import load_repaired_2m_2d

    models = load_repaired_2m_2d("cpu", key="repaired_2m_equal_2d")
    assert len(models["experts"]) == 2
    assert models["gates"] == []
    assert models["encoder_passes"] == 2


def test_loader_rejects_a_non_repaired_2m_key(registry):
    from molgap.inference import load_repaired_2m_2d

    with pytest.raises(ValueError, match="not a repaired-2M 2D preset"):
        load_repaired_2m_2d("cpu", key="phase8_routed_dualgps_hybrid")


def test_dense_preset_requires_all_three_experts(registry, monkeypatch):
    from molgap import constants as module
    from molgap.inference import load_repaired_2m_2d

    broken = {
        **registry,
        "repaired_2m_dense_2d": {
            **registry["repaired_2m_dense_2d"],
            "experts": ["repaired_2m_gps7", "repaired_2m_gps9"],
        },
    }
    monkeypatch.setattr(module, "MODEL_REGISTRY", broken)
    from molgap import inference

    monkeypatch.setattr(inference, "MODEL_REGISTRY", broken)
    with pytest.raises(ValueError, match="Dense gate needs"):
        load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")


def test_equal_preset_averages_its_experts(registry):
    from molgap.inference import (
        load_repaired_2m_2d,
        predict_smiles_batch_repaired_2m_2d,
    )

    models = load_repaired_2m_2d("cpu", key="repaired_2m_equal_2d")
    valid_idx, predictions, experts = predict_smiles_batch_repaired_2m_2d(
        ["c1ccccc1", "CCO"],
        models=models,
        return_expert_predictions=True,
    )
    assert valid_idx.tolist() == [0, 1]
    assert experts.shape == (2, 2, 3)
    # Expert heads were fixed to constants 1.0 and 2.0, so the equal average is
    # exactly 1.5 everywhere. This pins the fixed-weight contract.
    assert np.allclose(experts[:, 0], 1.0)
    assert np.allclose(experts[:, 1], 2.0)
    assert np.allclose(predictions, 1.5)


def test_dense_preset_blends_within_the_expert_range(registry):
    from molgap.inference import (
        load_repaired_2m_2d,
        predict_smiles_batch_repaired_2m_2d,
    )

    models = load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")
    _, predictions, experts = predict_smiles_batch_repaired_2m_2d(
        ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"],
        models=models,
        return_expert_predictions=True,
    )
    assert experts.shape == (3, 3, 3)
    # A softmax gate is a convex blend, so the fused value can never leave the
    # per-target expert envelope no matter what the gate weights are.
    assert (predictions >= experts.min(axis=1) - 1e-5).all()
    assert (predictions <= experts.max(axis=1) + 1e-5).all()


def test_invalid_smiles_are_dropped_not_imputed(registry):
    from molgap.inference import (
        load_repaired_2m_2d,
        predict_smiles_batch_repaired_2m_2d,
    )

    models = load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")
    valid_idx, predictions = predict_smiles_batch_repaired_2m_2d(
        ["c1ccccc1", "not_a_smiles", "", "CCO"],
        models=models,
    )
    assert valid_idx.tolist() == [0, 3]
    assert predictions.shape == (2, 3)
    assert np.isfinite(predictions).all()


def test_all_invalid_input_returns_aligned_empty_arrays(registry):
    from molgap.inference import (
        load_repaired_2m_2d,
        predict_smiles_batch_repaired_2m_2d,
    )

    models = load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")
    valid_idx, predictions, experts = predict_smiles_batch_repaired_2m_2d(
        ["not_a_smiles", ""],
        models=models,
        return_expert_predictions=True,
    )
    assert valid_idx.shape == (0,)
    assert predictions.shape == (0, 3)
    assert experts.shape == (0, 3, 3)


def test_batching_does_not_change_predictions(registry):
    from molgap.inference import (
        load_repaired_2m_2d,
        predict_smiles_batch_repaired_2m_2d,
    )

    smiles = ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "Clc1ccccc1"]
    models = load_repaired_2m_2d("cpu", key="repaired_2m_dense_2d")
    _, large = predict_smiles_batch_repaired_2m_2d(smiles, models=models, batch_size=64)
    _, small = predict_smiles_batch_repaired_2m_2d(smiles, models=models, batch_size=1)
    assert np.allclose(large, small, atol=1e-5)


def test_package_exports_the_public_names():
    import molgap

    assert "load_repaired_2m_2d" in dir(molgap)
    assert "predict_smiles_batch_repaired_2m_2d" in dir(molgap)


def test_real_registry_declares_both_presets_without_a_schnet_branch():
    for key, passes in (("repaired_2m_dense_2d", 3), ("repaired_2m_equal_2d", 2)):
        spec = constants.MODEL_REGISTRY[key]
        assert spec["encoder_passes"] == passes
        assert len(spec["experts"]) == passes
        # The dual-SchNet residual was rejected; no 3D component may appear here.
        assert "components" not in spec
        for expert in spec["experts"]:
            assert constants.MODEL_REGISTRY[expert]["kind"] == "gps"
