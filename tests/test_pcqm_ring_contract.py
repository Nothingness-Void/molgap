from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
MODEL = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
CACHE_RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "ring_hierarchy_cache"
    / "run_ring_hierarchy_cache.py"
)
CACHE_METADATA = CACHE_RUNNER.with_name("kernel-metadata.json")
CACHE_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_ring_hierarchy_cache.py"
PROTOCOL = EXPERIMENT / "ring_hierarchy_seed42_protocol.md"
CANDIDATE = "ogb_distance_angle_ring_hierarchy_triangle_edge_state_gps9"
COMPARATOR = "ogb_distance_angle_triangle_edge_state_gps9"


def test_ring_candidate_is_shared_narrow_and_within_budget():
    candidate = make_pcqm_gap_encoder(CANDIDATE)
    comparator = make_pcqm_gap_encoder(COMPARATOR)
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    comparator_count = sum(parameter.numel() for parameter in comparator.parameters())
    assert comparator_count == 4_891_057
    assert candidate_count == 4_949_097
    assert candidate_count - comparator_count == 58_040
    assert candidate_count <= 5_000_000
    assert candidate.HIERARCHY_LAYERS == (1, 3, 5, 7)
    assert candidate.ring_channels == 64
    assert torch.count_nonzero(candidate.ring_update.ring_to_atom.value.weight) == 0
    assert torch.count_nonzero(candidate.ring_update.ring_to_atom.value.bias) == 0


def test_ring_model_has_one_shared_update_and_atom_only_pooling():
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hierarchy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OGBRingHierarchyGeometrySparseTriangleEdgeStateGPSWrapper"
    )
    segment = ast.get_source_segment(source, hierarchy)
    assert "self.ring_update = _SharedRingHierarchyUpdate(" in segment
    assert "self.ring_updates = nn.ModuleList" not in segment
    assert "ring_pool" not in segment
    assert "ring_head" not in segment
    assert "self._pool(h, batch)" not in segment
    assert "auxiliary_state[\"ring_state\"]" in segment


def test_ring_cache_runner_is_cpu_only_and_role_sealed():
    source = CACHE_RUNNER.read_text(encoding="utf-8")
    assert "with_ring_hierarchy" in source
    assert "from rdkit" not in source
    assert "OFFICIAL_TRAIN_ROWS = 3_378_606" in source
    assert "58f425258031062c3c3762f13b7d4c160dffba65" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "model_inference_executed" in source
    metadata = json.loads(CACHE_METADATA.read_text(encoding="utf-8"))
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["code_file"] == "run_ring_hierarchy_cache.py"
    assert "piero0/pcqm4mv2" in metadata["dataset_sources"]
    assert "nothingnessvoid/molgap-pcqm-geometry-cache-s42-dataset" in metadata[
        "dataset_sources"
    ]


def test_ring_cache_acceptance_executes_no_model_and_pins_parent_hashes():
    source = CACHE_ACCEPTANCE.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "model_inference_executed" in source
    assert "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22" in source
    assert "58f425258031062c3c3762f13b7d4c160dffba65" in source
    assert "zero unresolved failures" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source


def test_ring_protocol_freezes_one_seed_and_closes_same_mechanism_retries():
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "seed: 42" in source
    assert "at or below 5,000,000" in source
    assert "14,400 seconds" in source
    assert "strictly lower" in source
    assert "does not submit" in source
    assert "ring-definition" in source
    assert "official validation/test-dev" in source
