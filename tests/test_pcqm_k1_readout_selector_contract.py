import ast
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.pcqm_k1_variants import (
    MIXER_LAYERS,
    _tied_selector_single_slot_update,
    make_encoder,
)
from molgap.pcqm_k1_variants_runner import (
    BATCH_SIZE,
    BENCHMARK_ID,
    EPOCHS,
    ROW_ORDER_FINGERPRINT,
    SAMPLE_EXPOSURE,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_k1_readout_selector_100k"
CANDIDATES = (
    "neural_atom_k1_repset_readout",
    "neural_atom_k1_tied_selector",
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


def test_round1_sources_parse():
    paths = [
        ROOT / "src/molgap/pcqm_k1_variants.py",
        ROOT / "src/molgap/pcqm_k1_variants_runner.py",
        EXPERIMENT / "t4x2_candidates/run_candidates.py",
        EXPERIMENT / "accept.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_round1_contract_is_exact_v4():
    contract = json.loads((EXPERIMENT / "training_contract.json").read_text())
    assert contract["benchmark_id"] == BENCHMARK_ID
    assert contract["candidates"] == list(CANDIDATES)
    assert contract["seed"] == 42
    assert contract["precision"] == "fp32"
    assert contract["tf32_enabled"] is False
    assert contract["physical_batch_per_device"] == BATCH_SIZE == 128
    assert contract["epochs"] == EPOCHS == 40
    assert contract["total_sample_presentations"] == SAMPLE_EXPOSURE == 3_998_720
    assert contract["row_order_fingerprint"] == ROW_ORDER_FINGERPRINT
    assert contract["roles"]["official_validation_role_read"] is False
    assert contract["roles"]["test_dev_role_read"] is False
    assert contract["roles"]["test_challenge_role_read"] is False


def test_round1_t4x2_kernel_is_private_and_isolated():
    metadata = json.loads(
        (EXPERIMENT / "t4x2_candidates/kernel-metadata.json").read_text()
    )
    assert metadata["is_private"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-k1-readout-selector-source",
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
    ]
    source = (EXPERIMENT / "t4x2_candidates/run_candidates.py").read_text()
    assert 'environment["CUDA_VISIBLE_DEVICES"] = str(device)' in source
    assert 'output / mode_name' in source
    assert "subprocess.Popen" in source


def test_round1_candidates_are_nested_and_parameter_bounded():
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
    expected_parameters = {
        "neural_atom_k1_repset_readout": 3_683_985,
        "neural_atom_k1_tied_selector": 3_622_401,
    }
    for mode in CANDIDATES:
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
        assert torch.equal(observed, expected)
        assert sum(parameter.numel() for parameter in candidate.parameters()) == expected_parameters[mode]


def test_repset_readout_matches_frozen_shape_and_is_permutation_invariant():
    batch = _batch()
    torch.manual_seed(42)
    model = make_encoder("neural_atom_k1_repset_readout").eval()
    readout = model.repset_readout
    assert readout.n_hidden_sets == 8
    assert readout.n_elements == 8
    assert tuple(readout.prototype.shape) == (192, 64)
    hidden = torch.randn(batch.num_nodes, 192)
    permutation = torch.tensor([2, 0, 1, 7, 4, 6, 3, 5])
    with torch.no_grad():
        expected = readout(hidden, batch.batch)
        observed = readout(hidden[permutation], batch.batch[permutation])
    assert torch.equal(expected, observed)
    assert torch.count_nonzero(expected) == 0


def test_tied_selector_shares_only_selection_path():
    batch = _batch()
    torch.manual_seed(42)
    model = make_encoder("neural_atom_k1_tied_selector").eval()
    assert len({id(model.shared_selector) for _ in MIXER_LAYERS}) == 1
    assert len(
        {
            id(model.base.neural_atom_mixers[str(layer)].node_value)
            for layer in MIXER_LAYERS
        }
    ) == 3
    assert len(
        {
            id(model.base.neural_atom_mixers[str(layer)].return_projection)
            for layer in MIXER_LAYERS
        }
    ) == 3
    hidden = torch.randn(batch.num_nodes, 192)
    for layer in MIXER_LAYERS:
        mixer = model.base.neural_atom_mixers[str(layer)]
        assert not hasattr(mixer, "node_key")
        assert not hasattr(mixer, "slot_query")
        with torch.no_grad():
            update, _, assignment, valid, diagnostics = (
                _tied_selector_single_slot_update(
                    mixer,
                    model.shared_selector,
                    hidden,
                    batch.batch,
                )
            )
        assert diagnostics["selector_scope"] == "shared-across-layers-3-6-9"
        assert diagnostics["value_scope"] == "independent-per-layer"
        assert torch.allclose(
            assignment.sum(dim=-1),
            torch.ones_like(assignment.sum(dim=-1)),
            atol=1e-6,
            rtol=0,
        )
        assert torch.count_nonzero(assignment.masked_select(~valid.unsqueeze(1))) == 0
        assert torch.count_nonzero(update) == 0

