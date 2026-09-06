"""K3 cache and static architecture contract tests; no model execution."""
from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch_geometric.data import Data

from molgap.pcqm_conjugated_cache import with_conjugated_components


ROOT = Path(__file__).resolve().parents[1]


def test_component_cache_is_deterministic_and_excludes_nonmembers() -> None:
    directed = [
        (0, 1), (1, 0), (1, 2), (2, 1),
        (3, 4), (4, 3), (4, 5), (5, 4),
    ]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    edge_attr = torch.zeros((len(directed), 3), dtype=torch.long)
    edge_attr[:6, 2] = 1
    edge_attr[4:6, 0] = 3
    x = torch.zeros((6, 9), dtype=torch.long)
    x[:, 0] = 5
    x[1, 7] = 1
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=6)
    result = with_conjugated_components(graph)
    assert result.conjugated_component_count.tolist() == [2]
    assert result.conjugated_component_id.tolist() == [0, 0, 0, 1, 1, -1]
    assert torch.equal(result.conjugated_features[0], result.conjugated_features[2])
    assert torch.equal(result.conjugated_features[3], result.conjugated_features[4])
    assert torch.count_nonzero(result.conjugated_features[5]) == 0
    before = (
        result.conjugated_component_id.clone(),
        result.conjugated_features.clone(),
    )
    with_conjugated_components(result)
    assert torch.equal(result.conjugated_component_id, before[0])
    assert torch.equal(result.conjugated_features, before[1])


def test_k3_sources_parse_and_freeze_one_chemical_mechanism() -> None:
    paths = [
        ROOT / "src/molgap/pcqm_conjugated_cache.py",
        ROOT / "src/molgap/pcqm_conjugated_state.py",
        ROOT / "src/molgap/pcqm_kunshan_screen.py",
        ROOT / "experiments/pcqm_gap_architecture/build_kunshan_conjugated_cache.py",
        ROOT / "experiments/pcqm_gap_architecture/accept_kunshan_conjugated_cache.py",
        ROOT / "experiments/pcqm_gap_architecture/accept_kunshan_conjugated_descriptor.py",
        ROOT / "experiments/pcqm_gap_architecture/accept_kunshan_conjugated_component.py",
        ROOT / "tests/remote_pcqm_conjugated_descriptor.py",
        ROOT / "tests/remote_pcqm_conjugated_component.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"))
    model = paths[1].read_text(encoding="utf-8")
    assert "COMPONENT_BLOCKS = (3, 6, 9)" in model
    assert "DESCRIPTOR_PARAMETERS = BASELINE_PARAMETERS + DESCRIPTOR_PARAMETER_DELTA" in model
    assert "COMPONENT_STATE_PARAMETERS = DESCRIPTOR_PARAMETERS + COMPONENT_PARAMETER_DELTA" in model
    assert "nn.init.zeros_(self.descriptor_to_atom.weight)" in model
    assert "_LowRankGatedProjection" in model
    assert "Dropout" not in model


def test_k3b_runner_and_slurm_freeze_component_state_comparison() -> None:
    runner = (ROOT / "src/molgap/pcqm_kunshan_screen.py").read_text(
        encoding="utf-8"
    )
    remote = (ROOT / "tests/remote_pcqm_conjugated_component.py").read_text(
        encoding="utf-8"
    )
    slurm = (
        ROOT
        / "experiments/pcqm_gap_architecture/kunshan_conjugated_component.slurm"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT / "experiments/pcqm_gap_architecture/kunshan_conjugated_component_protocol.md"
    ).read_text(encoding="utf-8")
    assert '"conjugated_component"' in runner
    assert 'format_name = "molgap-kunshan-conjugated-component-screen-v1"' in runner
    assert "3_694_033" in remote
    assert "shared initialization changed" in remote
    assert "zero component return changed initial prediction" in remote
    assert "--screen conjugated_component" in slurm
    assert "#SBATCH --gres=dcu:Hygon:1" in slurm
    assert "#SBATCH --time=12:00:00" in slurm
    assert "0.0001969527" in protocol
    assert "K3a checkpoints are not" in protocol
