import ast
import json
from pathlib import Path

from molgap.pcqm_local_geometry_pretraining import (
    BATCH_SIZE,
    EXPECTED_MODEL_PARAMETERS,
    FINETUNE_EPOCHS,
    MIN_PAIRED_GAIN_EV,
    PRETRAIN_EPOCHS,
    SCRATCH_EPOCHS,
    LocalGeometryHeads,
    _forward,
    _geometry_loss,
    _make_encoder,
)


ROOT = Path(__file__).resolve().parents[1]


def test_equal_encoder_exposure_and_xian_contract():
    assert BATCH_SIZE == 96
    assert SCRATCH_EPOCHS == PRETRAIN_EPOCHS + FINETUNE_EPOCHS == 40
    assert EXPECTED_MODEL_PARAMETERS == 4_771_073
    assert MIN_PAIRED_GAIN_EV == 0.003


def test_pure_2d_forward_excludes_geometry_inputs():
    source = (ROOT / "src/molgap/pcqm_local_geometry_pretraining.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_forward"
    )
    names = {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}
    assert "edge_distance" not in names
    assert "wedge_angle_cos" not in names
    assert "geometry_valid" not in names


def test_scnet_templates_are_durable_and_fail_closed():
    preflight = (ROOT / "platforms/scnet/pcqm_local_geometry_preflight_xian.slurm").read_text()
    pair = (ROOT / "platforms/scnet/pcqm_local_geometry_pair_xian.slurm").read_text()
    assert "set -euo pipefail" in preflight
    assert "set -euo pipefail" in pair
    assert "--gres=dcu:Hygon:1" in preflight
    assert "--gres=dcu:Hygon:1" in pair
    assert "--time=12:00:00" in pair
    assert " train " in pair.replace("\\\n", " ")
    assert " accept " in pair.replace("\\\n", " ")


def test_protocol_keeps_roles_sealed_and_requires_two_replicates():
    protocol = (
        ROOT
        / "experiments/pcqm_gap_architecture/local_geometry_pretraining_scnet_protocol.md"
    ).read_text()
    assert "Two independent Xi'an paired replicates" in protocol
    assert "0.003 eV" in protocol
    assert "official validation" in protocol
    assert "test-dev" in protocol


def test_current_state_points_to_protocol():
    state = (ROOT / "CURRENT_STATE.md").read_text()
    assert "local_geometry_pretraining_scnet_protocol.md" in state


def test_local_geometry_heads_run_on_a_real_pyg_batch():
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    graph = Data(
        x=torch.zeros((3, 9), dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        edge_attr=torch.zeros((4, 3), dtype=torch.long),
        y=torch.tensor([0.5], dtype=torch.float32),
        random_walk_pe=torch.zeros((3, 16), dtype=torch.float32),
        wedge_edge_ids=torch.tensor([[0, 2], [3, 1]], dtype=torch.long),
        edge_distance=torch.tensor([[1.4], [1.4], [1.5], [1.5]]),
        wedge_angle_cos=torch.tensor([[-0.5], [-0.5]]),
        geometry_valid=torch.tensor([True]),
        row_index=torch.tensor([7]),
    )
    batch = next(iter(DataLoader([graph], batch_size=1)))
    model = _make_encoder()
    heads = LocalGeometryHeads(model)
    _forward(model, batch)
    loss, distance, angle = _geometry_loss(heads, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(distance)
    assert torch.isfinite(angle)
    assert any(parameter.grad is not None for parameter in model.parameters())
    heads.close()
