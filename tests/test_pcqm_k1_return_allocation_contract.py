import ast
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.pcqm_k1_variants import make_encoder


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_k1_return_allocation_100k"
MODES = (
    "neural_atom_k1_uniform_return",
    "neural_atom_k1_inverse_return",
)


def _batch():
    graphs = []
    for offset, nodes in enumerate((3, 5)):
        source = torch.arange(nodes - 1)
        target = source + 1
        edge_index = torch.cat(
            [torch.stack([source, target]), torch.stack([target, source])], dim=1
        )
        graphs.append(
            Data(
                x=torch.zeros((nodes, 9), dtype=torch.long),
                edge_index=edge_index,
                edge_attr=torch.zeros((edge_index.shape[1], 3), dtype=torch.long),
                random_walk_pe=torch.zeros((nodes, 16)),
                y=torch.tensor([float(offset)]),
                source_idx=torch.tensor([offset]),
            )
        )
    return Batch.from_data_list(graphs)


def test_round3_sources_parse():
    for path in (
        ROOT / "src/molgap/pcqm_k1_variants.py",
        ROOT / "src/molgap/pcqm_k1_variants_runner.py",
        EXPERIMENT / "accept.py",
        EXPERIMENT / "t4x2_candidates/run_candidates.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_round3_contract_is_exact_v4_and_t4x2_private():
    contract = json.loads((EXPERIMENT / "training_contract.json").read_text())
    metadata = json.loads(
        (EXPERIMENT / "t4x2_candidates/kernel-metadata.json").read_text()
    )
    assert contract["benchmark_id"] == "pcqm4mv2-ogb-fixed-100k-gap-v4"
    assert contract["candidates"] == list(MODES)
    assert contract["physical_batch_per_device"] == 128
    assert contract["precision"] == "fp32"
    assert contract["tf32_enabled"] is False
    assert contract["total_optimizer_steps"] == 31_240
    assert contract["total_sample_presentations"] == 3_998_720
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] == "true"
    assert metadata["dataset_sources"][-1] == (
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"
    )


def test_round3_candidates_match_k1_at_initialization_and_parameter_count():
    batch = _batch()
    torch.manual_seed(42)
    reference = make_encoder("neural_atom_k1_v4").eval()
    with torch.no_grad():
        expected = reference(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    for mode in MODES:
        torch.manual_seed(42)
        candidate = make_encoder(mode).eval()
        with torch.no_grad():
            observed = candidate(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.random_walk_pe,
            )
        assert torch.equal(expected, observed)
        assert sum(parameter.numel() for parameter in candidate.parameters()) == 3_658_817
        assert candidate.base.state_dict().keys() == reference.state_dict().keys()
