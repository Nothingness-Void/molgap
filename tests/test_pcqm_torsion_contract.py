from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_torsion_model_is_registered_with_the_fixed_contract():
    source = read("src/molgap/pcqm_gap_architecture.py")
    tree = ast.parse(source)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "OGBTorsionGeometrySparseTriangleEdgeStateGPSWrapper" in classes
    assert "_SharedTorsionGatedUpdate" in classes
    assert '"ogb_distance_angle_torsion_triangle_edge_state_gps9"' in source
    assert "self.torsion_update = _SharedTorsionGatedUpdate(" in source
    assert "self.torsion_updates = nn.ModuleList" not in source
    assert "nn.init.zeros_(projection.weight)" in source
    assert "nn.init.zeros_(projection.bias)" in source


def test_torsion_cache_and_gpu_runner_keep_sealed_roles_and_budget():
    cache = read(
        "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/torsion_cache/run_torsion_cache.py"
    )
    runner = read(
        "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/sparse_torsion_edge_state_seed42/run_torsion_screen.py"
    )
    assert "with_torsion_cache" in cache
    assert "atomic_torch_save" in cache
    assert '"complete": False' in cache
    assert "SEARCH_BUDGET_S = 23_400" in runner
    assert "CANDIDATES = (COMPARATOR, CANDIDATE)" in runner
    assert "train_generator = torch.Generator().manual_seed(SEED)" in runner
    assert 'train_generator_state": train_generator.get_state()' in runner
    assert 'train_generator.set_state(checkpoint["train_generator_state"])' in runner
    assert "torch.allclose(" in runner
    assert "max_abs_diff" in runner
    assert '"official_validation_role_read": False' in runner
    assert '"test_dev_role_read": False' in runner
    assert "piero0/pcqm4mv2" not in runner


def test_kernel_metadata_separates_cpu_cache_and_gpu_task():
    cache = json.loads(
        read(
            "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/torsion_cache/kernel-metadata.json"
        )
    )
    gpu = json.loads(
        read(
            "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/sparse_torsion_edge_state_seed42/kernel-metadata.json"
        )
    )
    assert cache["enable_gpu"] == "false"
    assert gpu["enable_gpu"] == "true"
    assert cache["id"].startswith("nothingnessvoid/")
    assert gpu["id"].startswith("nothingnessvoid/")
    assert gpu["kernel_sources"] == []
    assert gpu["dataset_sources"] == [
        "nothingnessvoid/molgap-pcqm-torsion-source-3d4cdb73",
        "nothingnessvoid/molgap-pcqm-torsion-cache-s42-dataset",
    ]
