from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "molgap" / "pcqm_gap_data.py"
MODELS = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
GPS = ROOT / "src" / "molgap" / "gps.py"
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
PREP = EXPERIMENT / "kaggle_pcqm_gap100k" / "cpu_prep" / "run_prep.py"
METADATA = PREP.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_cache.py"
GPU = EXPERIMENT / "kaggle_pcqm_gap100k" / "gpu_seed42" / "run_screen.py"
GPU_METADATA = GPU.with_name("kernel-metadata.json")
GPU_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_seed42.py"


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name} in {path}")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    return modules


def test_official_train_only_split_contract() -> None:
    assert assignment_literal(DATA, "OFFICIAL_TRAIN_ROWS") == 3_378_606
    assert assignment_literal(DATA, "SCREEN_TRAIN_ROWS") == 100_000
    assert assignment_literal(DATA, "SCREEN_VALIDATION_ROWS") == 10_000
    source = DATA.read_text(encoding="utf-8")
    assert "nrows=OFFICIAL_TRAIN_ROWS" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "smiles2graph" in source
    assert "walk_length=RWSE_DIM" in source
    assert "_atomic_torch_save(part_path, graphs)" in source


def test_gap_models_use_ogb_categories_and_one_target() -> None:
    source = MODELS.read_text(encoding="utf-8")
    assert "class OGBStructuralGPSWrapper" in source
    assert "class OGBEdgeStateStructuralGPSWrapper" in source
    assert "AtomEncoder" in source and "BondEncoder" in source
    assert '"n_targets": 1' in source
    assert '"num_layers": 9' in source
    assert '"hidden_channels": 192' in source
    assert "edge_state_channels=64" in source
    gps_source = GPS.read_text(encoding="utf-8")
    assert "def _embed_nodes(self, x):" in gps_source
    assert "def _embed_edges(self, edge_attr):" in gps_source


def test_cpu_prep_is_gpu_free_and_uses_only_official_mirror() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "false"
    assert metadata["dataset_sources"] == [
        "piero0/pcqm4mv2",
        "kaseichou/molgap-pcqm-gap100k-r1-source",
    ]
    source = PREP.read_text(encoding="utf-8")
    assert "rdkit==2025.3.5" in source
    assert '"gpu_used": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_cache_acceptance_executes_no_model_runtime() -> None:
    assert "torch" not in imported_modules(ACCEPTANCE)
    source = ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_protocol_freezes_gap_only_kaggle_before_server() -> None:
    protocol = (EXPERIMENT / "pcqm100k_gap_screen_protocol.md").read_text(
        encoding="utf-8"
    )
    assert "official PCQM4Mv2 Gap task" in protocol
    assert "exactly `100,000`" in protocol
    assert "Seeds 43 and 44" in protocol
    assert "12 hours" in protocol
    assert "/lustre/home/users/sm2/chou/" in protocol


def test_gpu_screen_is_matched_gap_only_and_sequential() -> None:
    assert assignment_literal(GPU, "EXPECTED_SOURCE_COMMIT") == (
        "a67724999dbe145b38c2792b86d4e654f5589a20"
    )
    assert assignment_literal(GPU, "CANDIDATES") == (
        "ogb_structural_gps9",
        "ogb_edge_state_structural_gps9",
    )
    assert assignment_literal(GPU, "BATCH_SIZE") == 48
    assert assignment_literal(GPU, "MAX_EPOCHS") == 40
    source = GPU.read_text(encoding="utf-8")
    assert "results = [train_candidate(candidate, graphs) for candidate in CANDIDATES]" in source
    assert '"target": "homolumogap"' in source
    assert '"precision": "fp32"' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    metadata = json.loads(GPU_METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-official-pcqm-gap100k-r1-prep"
    ]


def test_gpu_acceptance_executes_no_model_runtime() -> None:
    assert "torch" not in imported_modules(GPU_ACCEPTANCE)
    source = GPU_ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
