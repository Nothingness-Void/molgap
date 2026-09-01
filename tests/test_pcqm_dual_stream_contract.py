from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/"
    "sparse_atom_bond_dual_stream_seed42/run_dual_stream_screen.py"
)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dual_stream_model_and_information_flow_are_frozen():
    source = read("src/molgap/pcqm_gap_architecture.py")
    tree = ast.parse(source)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "OGBDualStreamGeometrySparseTriangleEdgeStateGPSWrapper" in classes
    assert "_SparseBondAttentionBlock" in classes
    assert "_SharedAtomBondExchange" in classes
    assert 'DUAL_STREAM_LAYERS = (1, 3, 5, 7)' in source
    assert '"ogb_distance_angle_dual_stream_triangle_edge_state_gps9"' in source
    assert "from torch_geometric.utils import softmax" in source
    assert "attended.index_add_(0, target, message)" in source
    assert "endpoint_mean = 0.5 * (h[source] + h[destination])" in source
    assert "nn.init.zeros_(self.value.weight)" in source


def test_dual_stream_runner_uses_one_candidate_and_frozen_comparator():
    runner = read(RUNNER)
    assert "SEARCH_BUDGET_S = 14_400" in runner
    assert "EXPECTED_CACHE_SOURCE_COMMIT" in runner
    assert "EXPECTED_COMPARATOR_METRICS_SHA256" in runner
    assert "load_frozen_comparator(" in runner
    assert "result = train_one(\n            graphs,\n            CANDIDATE," in runner
    assert "for candidate in CANDIDATES:" in runner  # preflight only
    assert "hydrate_resume_state(" not in runner
    assert "del graph[name]" in runner
    assert "train_generator = torch.Generator().manual_seed(SEED)" in runner
    assert '"train_generator_state": train_generator.get_state()' in runner
    assert "train_generator.set_state(" in runner
    assert 'checkpoint["torch_rng_state"]' in runner
    assert 'checkpoint["cuda_rng_state_all"]' in runner
    assert "shared_backbone_parameters_match" in runner
    assert "dual_stream_injection_zero" in runner
    assert "len(zero_parameters) != 20" in runner
    assert '"official_validation_role_read": False' in runner
    assert '"test_dev_role_read": False' in runner
    assert "piero0/pcqm4mv2" not in runner


def test_dual_stream_acceptance_is_no_inference_and_hash_pinned():
    acceptance = read(
        "experiments/pcqm_gap_architecture/accept_pcqm100k_dual_stream_seed42.py"
    )
    assert "EXPECTED_COMPARATOR_HASHES" in acceptance
    assert "EXPECTED_RESUME_MANIFEST_SHA256" in acceptance
    assert "EXPECTED_VALIDATION_ROW_SHA256" in acceptance
    assert "EXPECTED_VALIDATION_TARGET_SHA256" in acceptance
    assert '"model_inference_executed": False' in acceptance
    assert '"official_validation_role_read": False' in acceptance
    assert '"test_dev_role_read": False' in acceptance
    assert "torch" not in acceptance


def test_dual_stream_kernel_is_one_private_gpu_task():
    metadata = json.loads(
        read(
            "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/"
            "sparse_atom_bond_dual_stream_seed42/kernel-metadata.json"
        )
    )
    assert metadata["id"] == (
        "nothingnessvoid/molgap-pcqm-sparse-atom-bond-dual-stream-s42"
    )
    assert metadata["code_file"] == "run_dual_stream_screen.py"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []
    assert metadata["competition_sources"] == []
    assert metadata["dataset_sources"][1:] == [
        "nothingnessvoid/molgap-pcqm-torsion-cache-s42-dataset",
        "nothingnessvoid/molgap-pcqm-sparse-torsion-s42-resume-v3",
    ]
