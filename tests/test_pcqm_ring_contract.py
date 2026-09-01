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
GPU_RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "ring_hierarchy_seed42"
    / "run_ring_hierarchy_screen.py"
)
GPU_METADATA = GPU_RUNNER.with_name("kernel-metadata.json")
GPU_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_ring_hierarchy_seed42.py"
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


def test_ring_gpu_runner_is_one_candidate_against_the_frozen_comparator():
    source = GPU_RUNNER.read_text(encoding="utf-8")
    assert f'CANDIDATE = "{CANDIDATE}"' in source
    assert "CANDIDATE: 4_949_097" in source
    assert "PARAMETER_BUDGET = 5_000_000" in source
    assert "BATCH_SIZE = 48" in source
    assert "LEARNING_RATE = 1.6e-4" in source
    assert "MAX_EPOCHS = 40" in source
    assert "PATIENCE = 8" in source
    assert "ring_update.ring_to_atom.value.weight" in source
    assert "ring_return_gradient_nonzero" in source
    assert "batch.ring_features" in source
    assert "batch.atom_ring_index" in source
    assert "train_one(\n            graphs,\n            CANDIDATE," in source
    assert "seed43_44_submitted" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source


def test_ring_gpu_metadata_is_one_private_p100_compatible_task():
    metadata = json.loads(GPU_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "nothingnessvoid/molgap-pcqm-ring-hierarchy-s42"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["code_file"] == "run_ring_hierarchy_screen.py"
    assert metadata["kernel_sources"] == [
        "nothingnessvoid/molgap-pcqm-ring-hierarchy-cache-s42"
    ]
    assert "nothingnessvoid/molgap-pcqm-ring-hierarchy-source" in metadata[
        "dataset_sources"
    ]
    assert "nothingnessvoid/molgap-pcqm-sparse-torsion-s42-resume-v3" in metadata[
        "dataset_sources"
    ]
    source = GPU_RUNNER.read_text(encoding="utf-8")
    assert "torch==2.7.1" in source
    assert "sm_60" in source


def test_ring_gpu_acceptance_is_no_inference_and_arithmetic_only():
    source = GPU_ACCEPTANCE.read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "model_inference_executed" in source
    assert "candidate_minus_comparator_eV" in source
    assert "EXPECTED_COMPARATOR_MAE = 0.1353926807641983" in source
    assert "4_949_097" in source
    assert "ring_cache_aggregate_sha256" in source
    assert "official_validation_role_read" in source
    assert "test_dev_role_read" in source
