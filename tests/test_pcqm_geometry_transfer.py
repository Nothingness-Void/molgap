from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from molgap.gptrans import OGBGPTransTiny
from molgap.pcqm_geometry_transfer import (
    GEOMETRY_GPTRANS_T,
    GEOMETRY_NEURAL_ATOM_K1,
    GeometryGPTransTiny,
    GeometryNeuralAtomK1,
    make_geometry_transfer_model,
)
from molgap.pcqm_geometry_transfer_runner import (
    DEVELOPMENT_ROWS,
    EXPECTED_PARAMETERS,
    TRAIN_ROWS,
    GeometryTransferConfig,
    run_fusion,
)
from molgap.qm9_neural_atom import make_encoder


def _graph(nodes: int) -> Data:
    source = []
    target = []
    for node in range(nodes - 1):
        source.extend((node, node + 1))
        target.extend((node + 1, node))
    edge_index = torch.tensor([source, target], dtype=torch.long)
    wedges = [[2 * center - 1, 2 * center] for center in range(1, nodes - 1)]
    return Data(
        x=torch.zeros((nodes, 9), dtype=torch.long),
        edge_index=edge_index,
        edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
        y=torch.tensor([1.0]),
        random_walk_pe=torch.zeros((nodes, 16)),
        wedge_edge_ids=torch.tensor(wedges, dtype=torch.long).reshape(-1, 2),
        edge_distance=torch.full((edge_index.shape[1], 1), 1.4),
        wedge_angle_cos=torch.zeros((len(wedges), 1)),
        geometry_valid=torch.tensor([True]),
        source_idx=torch.tensor([nodes]),
    )


def _batch():
    return Batch.from_data_list([_graph(4), _graph(5)])


def _geometry_forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
        batch.wedge_edge_ids,
        batch.edge_distance,
        batch.wedge_angle_cos,
        batch.geometry_valid,
    )


def test_parameter_contracts_and_training_contracts():
    for model_id, expected in EXPECTED_PARAMETERS.items():
        model = make_geometry_transfer_model(model_id)
        assert sum(parameter.numel() for parameter in model.parameters()) == expected
        config = GeometryTransferConfig(model_id=model_id)
        config.validate()
        assert config.batch_size == 128
        assert config.drop_last is True
    assert GeometryTransferConfig(GEOMETRY_GPTRANS_T).epochs == 60
    assert GeometryTransferConfig(GEOMETRY_NEURAL_ATOM_K1).epochs == 40


def test_geometry_gptrans_zero_start_matches_original_function():
    batch = _batch()
    torch.manual_seed(42)
    baseline = OGBGPTransTiny().eval()
    torch.manual_seed(42)
    candidate = GeometryGPTransTiny().eval()
    expected = baseline(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    observed = _geometry_forward(candidate, batch)
    assert torch.equal(observed, expected)


def test_geometry_k1_zero_start_matches_original_function():
    batch = _batch()
    torch.manual_seed(42)
    baseline = make_encoder("neural_atom_k1").eval()
    torch.manual_seed(42)
    candidate = GeometryNeuralAtomK1().eval()
    expected = baseline(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    observed = _geometry_forward(candidate, batch)
    assert torch.equal(observed, expected)


def test_geometry_candidates_have_finite_backward():
    batch = _batch()
    for model_id in (GEOMETRY_GPTRANS_T, GEOMETRY_NEURAL_ATOM_K1):
        model = make_geometry_transfer_model(model_id).train()
        _geometry_forward(model, batch).sum().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_fusion_requires_alignment_and_reports_oof(tmp_path):
    target = torch.linspace(-1.0, 1.0, DEVELOPMENT_ROWS)
    source_idx = torch.arange(TRAIN_ROWS, TRAIN_ROWS + DEVELOPMENT_ROWS)
    predictions = {
        GEOMETRY_GPTRANS_T: target + 0.08,
        GEOMETRY_NEURAL_ATOM_K1: target - 0.04,
    }
    for model_id, prediction in predictions.items():
        root = tmp_path / model_id
        root.mkdir()
        payload = root / "best_validation_payload.pt"
        torch.save(
            {
                "source_idx": source_idx,
                "target_eV": target,
                "prediction_eV": prediction,
            },
            payload,
        )
        import hashlib

        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (root / "completion_manifest.json").write_text(
            __import__("json").dumps(
                {
                    "complete": True,
                    "best_development_mae_eV": float(
                        (prediction - target).abs().mean()
                    ),
                    "artifacts": {"best_validation_payload.pt": digest},
                }
            ),
            encoding="utf-8",
        )
    result = run_fusion(tmp_path)
    assert result["complete"] is True
    assert result["rows"] == DEVELOPMENT_ROWS
    assert result["development_mae_eV"]["five_fold_oof_convex_blend"] < 5e-5
    assert len(result["five_fold_reference_weights"]) == 5
