from pathlib import Path

import torch
from torch_geometric.data import Data

from molgap.k1_functional_group_token import (
    ADDED_PARAMETERS,
    MODE,
    PARAMETERS,
    make_encoder,
)
from molgap.pcqm_functional_group_sidecar import (
    GROUP_NAMES,
    functional_group_labels_from_graph,
)
from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS


ROOT = Path(__file__).resolve().parent


def test_contract_and_registry_match() -> None:
    config = ARCHITECTURE_CONFIGS[MODE]
    assert config["added_parameters"] == ADDED_PARAMETERS == 42_112
    assert config["expected_parameters"] == PARAMETERS[MODE] == 3_700_929
    assert config["external_features"] is False
    assert "PENDING_CPU_ACCEPTANCE" in (ROOT / "training_contract.json").read_text()


def test_graph_rules_use_existing_ogb_tensors() -> None:
    # O=C-N with O-H/C alcohol-like local roles; exact chemistry is less
    # important here than deterministic tensor-only incidence semantics.
    x = torch.zeros((3, 9), dtype=torch.long)
    x[:, 0] = torch.tensor([5, 7, 6])  # C, O, N as OGB atomic-number indices.
    x[1, 4] = 0
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    edge_attr = torch.zeros((4, 3), dtype=torch.long)
    edge_attr[0, 0] = edge_attr[1, 0] = 1  # C=O
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    labels = functional_group_labels_from_graph(graph)
    assert labels.shape == (3, len(GROUP_NAMES))
    assert labels[:, GROUP_NAMES.index("carbonyl")].tolist() == [1, 1, 0]
    assert labels[:, GROUP_NAMES.index("amide")].tolist() == [1, 1, 1]


def test_model_parameter_identity() -> None:
    model = make_encoder(MODE)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETERS[MODE]
    assert torch.count_nonzero(
        model.functional_group_token.return_projection.weight
    ).item() == 0
