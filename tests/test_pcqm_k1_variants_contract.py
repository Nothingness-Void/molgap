import ast
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.pcqm_k1_variants import (
    ARCHITECTURE_CONFIGS,
    _cluster_mixer_update,
    _dynamic_query_single_slot_update,
    _multihead_single_slot_update,
    make_encoder,
)
from molgap.pcqm_k1_variants_runner import (
    BATCH_SIZE,
    EPOCHS,
    FIXED_MANIFEST_SHA256,
    ROW_ORDER_FINGERPRINT,
    ROWS_PER_EPOCH,
    SAMPLE_EXPOSURE,
    compute_row_order_fingerprint,
    epoch_order,
)


ROOT = Path(__file__).resolve().parents[1]


def test_sources_and_remote_entrypoints_parse():
    paths = [
        ROOT / "src/molgap/pcqm_k1_variants.py",
        ROOT / "src/molgap/pcqm_k1_variants_runner.py",
        ROOT / "experiments/pcqm_k1_variants_100k/p100_reference/run_reference.py",
        ROOT / "experiments/pcqm_k1_variants_100k/t4x2_candidates/run_candidates.py",
        ROOT / "experiments/pcqm_k1_variants_100k/accept.py",
        ROOT / "experiments/pcqm_k1_variants_100k/package_source_dataset.py",
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_v4_exposure_and_row_order_are_frozen():
    assert BATCH_SIZE == 128
    assert EPOCHS == 40
    assert ROWS_PER_EPOCH == 99_968
    assert SAMPLE_EXPOSURE == 3_998_720
    assert len(epoch_order(0)) == ROWS_PER_EPOCH
    assert len(set(epoch_order(0))) == ROWS_PER_EPOCH
    assert epoch_order(0) != epoch_order(1)
    assert compute_row_order_fingerprint() == ROW_ORDER_FINGERPRINT
    assert len(FIXED_MANIFEST_SHA256) == 64


def _batch():
    graphs = []
    for offset, nodes in enumerate((3, 4)):
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


def test_candidates_are_exactly_nested_in_k1_at_initialization():
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
    expected_counts = {
        "neural_atom_k1_v4": 3_658_817,
        "neural_atom_k1_g": 3_698_180,
        "neural_atom_k1_r": 3_739_841,
        "neural_atom_k4_cluster": 3_658_817,
        "neural_atom_k1_h4": 3_658_817,
        "neural_atom_k1_dynamic_query": 3_695_681,
    }
    for mode in ("neural_atom_k1_g", "neural_atom_k1_r"):
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
        assert candidate.base.state_dict().keys() == reference.state_dict().keys()
    torch.manual_seed(42)
    clustered = make_encoder("neural_atom_k4_cluster").eval()
    with torch.no_grad():
        clustered_initial = clustered(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    assert torch.equal(expected, clustered_initial)
    assert all(
        mixer.active_slots == 4
        for mixer in clustered.base.neural_atom_mixers.values()
    )
    for mode, expected_count in expected_counts.items():
        assert sum(p.numel() for p in make_encoder(mode).parameters()) == expected_count
    assert set(ARCHITECTURE_CONFIGS) == set(expected_counts)


def test_clustered_neural_atoms_allocate_each_atom_across_slots():
    batch = _batch()
    torch.manual_seed(42)
    model = make_encoder("neural_atom_k4_cluster").eval()
    mixer = model.base.neural_atom_mixers["3"]
    hidden = torch.randn(batch.num_nodes, 192)
    with torch.no_grad():
        update, _, assignment, valid, diagnostics = _cluster_mixer_update(
            mixer, hidden, batch.batch
        )
    assert diagnostics["allocation_normalization_axis"] == "slots"
    assert assignment.shape[1] == 4
    assert torch.allclose(
        assignment.sum(dim=1).masked_select(valid),
        torch.ones_like(valid, dtype=assignment.dtype).masked_select(valid),
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(assignment.masked_select(~valid.unsqueeze(1))) == 0
    assert torch.count_nonzero(update) == 0


def test_dynamic_query_retains_one_atom_distribution_and_exact_k1_start():
    batch = _batch()
    torch.manual_seed(42)
    reference = make_encoder("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    model = make_encoder("neural_atom_k1_dynamic_query").eval()
    with torch.no_grad():
        expected = reference(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
        observed = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    assert torch.equal(expected, observed)
    mixer = model.base.neural_atom_mixers["3"]
    conditioner = model.query_conditioners["3"]
    assert torch.count_nonzero(conditioner.weight) == 0
    hidden = torch.randn(batch.num_nodes, 192)
    with torch.no_grad():
        update, slots, assignment, valid, diagnostics = (
            _dynamic_query_single_slot_update(
                mixer, conditioner, hidden, batch.batch
            )
        )
    assert slots.shape == (2, 1, 64)
    assert assignment.shape[:2] == (2, 1)
    assert diagnostics["query_context"] == "mean-current-node-state"
    assert torch.allclose(
        assignment.sum(dim=-1),
        torch.ones_like(assignment.sum(dim=-1)),
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(
        assignment.masked_select(~valid.unsqueeze(1))
    ) == 0
    assert torch.count_nonzero(update) == 0


def test_multihead_single_slot_selects_atoms_per_head():
    batch = _batch()
    torch.manual_seed(42)
    reference = make_encoder("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    model = make_encoder("neural_atom_k1_h4").eval()
    with torch.no_grad():
        expected = reference(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
        observed = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.random_walk_pe,
        )
    assert torch.equal(expected, observed)
    mixer = model.base.neural_atom_mixers["3"]
    hidden = torch.randn(batch.num_nodes, 192)
    with torch.no_grad():
        update, slots, assignment, valid, diagnostics = (
            _multihead_single_slot_update(mixer, hidden, batch.batch)
        )
    assert slots.shape == (2, 1, 64)
    assert assignment.shape[:3] == (2, 4, 1)
    assert diagnostics["active_slots"] == 1
    assert diagnostics["allocation_heads"] == 4
    assert torch.allclose(
        assignment.sum(dim=-1),
        torch.ones_like(assignment.sum(dim=-1)),
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(
        assignment.masked_select(~valid[:, None, None, :])
    ) == 0
    assert torch.count_nonzero(update) == 0
