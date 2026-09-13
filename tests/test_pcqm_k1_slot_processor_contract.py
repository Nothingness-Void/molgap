import ast
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from molgap.pcqm_k1_variants import (
    MIXER_LAYERS,
    _single_slot_processor_update,
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
EXPERIMENT = ROOT / "experiments/pcqm_k1_slot_processor_100k"
CANDIDATES = (
    "neural_atom_k1_collapsed_mha",
    "neural_atom_k1_no_slot_attention",
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


def test_round2_sources_parse():
    for path in (
        ROOT / "src/molgap/pcqm_k1_variants.py",
        ROOT / "src/molgap/pcqm_k1_variants_runner.py",
        EXPERIMENT / "accept.py",
        EXPERIMENT / "t4x2_candidates/run_candidates.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_round2_t4x2_kernel_is_private_and_isolated():
    metadata = json.loads(
        (EXPERIMENT / "t4x2_candidates/kernel-metadata.json").read_text()
    )
    assert metadata["is_private"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-k1-slot-processor-source",
        "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
    ]
    source = (EXPERIMENT / "t4x2_candidates/run_candidates.py").read_text()
    assert 'environment["CUDA_VISIBLE_DEVICES"] = str(device)' in source
    assert 'output / mode_name' in source
    assert "subprocess.Popen" in source


def test_round2_contract_is_exact_v4():
    contract = json.loads((EXPERIMENT / "training_contract.json").read_text())
    assert contract["round"] == 2
    assert contract["benchmark_id"] == BENCHMARK_ID
    assert contract["candidates"] == list(CANDIDATES)
    assert contract["reference_retrained"] is False
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


def test_round2_candidates_are_nested_and_parameter_bounded():
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
        "neural_atom_k1_collapsed_mha": 3_633_857,
        "neural_atom_k1_no_slot_attention": 3_608_897,
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
        assert sum(p.numel() for p in candidate.parameters()) == expected_parameters[mode]
        assert all(
            not hasattr(mixer, "slot_attention")
            for mixer in candidate.base.neural_atom_mixers.values()
        )


def test_collapsed_projection_matches_length_one_attention():
    torch.manual_seed(42)
    reference = make_encoder("neural_atom_k1_v4").eval()
    torch.manual_seed(42)
    candidate = make_encoder("neural_atom_k1_collapsed_mha").eval()
    slots = torch.linspace(-0.5, 0.5, steps=2 * 64).reshape(2, 1, 64)
    with torch.no_grad():
        for layer in MIXER_LAYERS:
            expected, _ = reference.neural_atom_mixers[str(layer)].slot_attention(
                slots,
                slots,
                slots,
                need_weights=False,
            )
            observed = candidate.collapsed_slot_projections[str(layer)](slots)
            assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)


def test_no_attention_retains_one_slot_pool_ffn_and_zero_return():
    batch = _batch()
    torch.manual_seed(42)
    candidate = make_encoder("neural_atom_k1_no_slot_attention").eval()
    hidden = torch.randn(batch.num_nodes, 192)
    for layer in MIXER_LAYERS:
        mixer = candidate.base.neural_atom_mixers[str(layer)]
        assert hasattr(mixer, "slot_ffn")
        assert hasattr(mixer, "return_projection")
        with torch.no_grad():
            update, slots, assignment, valid, diagnostics = (
                _single_slot_processor_update(
                    mixer,
                    None,
                    hidden,
                    batch.batch,
                )
            )
        assert slots.shape == (2, 1, 64)
        assert diagnostics["slot_processor"] == "none"
        assert diagnostics["slot_attention_module_removed"] is True
        assert torch.allclose(
            assignment.sum(dim=-1),
            torch.ones_like(assignment.sum(dim=-1)),
            atol=1e-6,
            rtol=0,
        )
        assert torch.count_nonzero(assignment.masked_select(~valid.unsqueeze(1))) == 0
        assert torch.count_nonzero(update) == 0
