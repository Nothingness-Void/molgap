from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/pcqm_gap_architecture"
MODEL = ROOT / "src/molgap/pcqm_hop_path.py"
FACTORY = ROOT / "src/molgap/pcqm_gap_architecture.py"
RUNNER = ROOT / "src/molgap/pcqm_local_global_runner.py"
CACHE_RUNNER = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k/hop_path_cache/run_hop_path_cache.py"
)
CACHE_METADATA = CACHE_RUNNER.with_name("kernel-metadata.json")
CACHE_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_hop_path_cache.py"
GPU_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_hop_path_graphstate.py"
PROTOCOL = EXPERIMENT / "hop_path_graphstate_seed42_protocol.md"
SOURCE_PACKAGER = (
    EXPERIMENT / "kaggle_pcqm_gap100k/package_source_dataset.py"
)
GPU_RUNNER = (
    EXPERIMENT / "kaggle_pcqm_gap100k/hop_path_graphstate_seed42/run_screen.py"
)
GPU_METADATA = GPU_RUNNER.with_name("kernel-metadata.json")


def test_sources_parse_and_cache_contract_is_explicit() -> None:
    for path in (
        MODEL,
        FACTORY,
        RUNNER,
        CACHE_RUNNER,
        CACHE_ACCEPTANCE,
        GPU_RUNNER,
        GPU_ACCEPTANCE,
    ):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source = MODEL.read_text(encoding="utf-8")
    assert "exact-shortest 2/3-hop" in source
    assert "distances[target] == depth" in source
    assert "target not in path" in source
    assert "fully_conjugated / float(path_count)" in source
    assert "nn.init.zeros_(self.output_value.weight)" in source
    assert "PATH_BLOCKS = (3, 6, 9)" in source


def test_factory_and_remote_runner_freeze_one_paired_mode() -> None:
    factory = FACTORY.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    candidate = "ogb_distance_angle_hop_path_triangle_edge_state_graph_state9"
    assert candidate in factory
    assert candidate in runner
    assert '"hop_path_graphstate"' in runner
    assert "3_697_537" in runner
    assert "EXPECTED_HOP_PATH_CACHE_SHA256" in runner
    assert "SEARCH_BUDGET_S = 14_400" in runner
    assert "EXPECTED_GPU_COUNT = 2" in runner
    assert '"hop_path_mixer"' in runner


def test_cpu_kernel_is_private_and_never_requests_gpu() -> None:
    metadata = json.loads(CACHE_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "nothingnessvoid/molgap-pcqm-hop-path-cache-s42"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["dataset_sources"] == [
        "nothingnessvoid/molgap-pcqm-hop-path-source",
        "nothingnessvoid/molgap-pcqm-geometry-cache-s42-dataset",
    ]
    packager = SOURCE_PACKAGER.read_text(encoding="utf-8")
    assert '"isPrivate": True' in packager


def test_acceptance_is_no_model_and_roles_remain_sealed() -> None:
    cache_source = CACHE_ACCEPTANCE.read_text(encoding="utf-8")
    gpu_source = GPU_ACCEPTANCE.read_text(encoding="utf-8")
    for source in (cache_source, gpu_source):
        assert "import torch" not in source
        assert '"model_inference_executed": False' in source
        assert '"official_validation_role_read": False' in source
        assert '"test_dev_role_read": False' in source
    assert '"hops": [2, 3]' in cache_source
    assert '"feature_channels": 8' in cache_source
    assert '"material_gain_at_least_0_001_eV"' in gpu_source


def test_gpu_kernel_is_private_t4x2_and_pins_accepted_cache() -> None:
    metadata = json.loads(GPU_METADATA.read_text(encoding="utf-8"))
    runner = GPU_RUNNER.read_text(encoding="utf-8")
    assert metadata["id"] == "nothingnessvoid/molgap-pcqm-hop-path-graphstate-s42"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["kernel_sources"] == [
        "nothingnessvoid/molgap-pcqm-hop-path-cache-s42"
    ]
    assert "hop_path_graphstate" in runner
    assert "0253aaa2061d0e7ac655fc8b6d9effd283722096" in runner
    assert "6d3a7df67cfadc26db2a38c006fd20e0d4294c16968fa92b01739daf8527993e" in runner


def test_protocol_is_one_mechanism_and_seed42_only() -> None:
    source = PROTOCOL.read_text(encoding="utf-8")
    assert "seed 42" in source
    assert "100,000 train / 10,000 internal-validation" in source
    assert "blocks 3, 6 and 9" in source
    assert "3,697,537" in source
    assert "0.001 eV" in source
    assert "Completion never authorizes seed 43/44" in source
