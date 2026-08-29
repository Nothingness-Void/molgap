from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "molgap" / "pcqm_gap_data.py"
MODELS = ROOT / "src" / "molgap" / "pcqm_gap_architecture.py"
GEOMETRY = ROOT / "src" / "molgap" / "pcqm_geometry.py"
GPS = ROOT / "src" / "molgap" / "gps.py"
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
PREP = EXPERIMENT / "kaggle_pcqm_gap100k" / "cpu_prep" / "run_prep.py"
METADATA = PREP.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_cache.py"
GPU = EXPERIMENT / "kaggle_pcqm_gap100k" / "gpu_seed42" / "run_screen.py"
GPU_METADATA = GPU.with_name("kernel-metadata.json")
GPU_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_seed42.py"
NOVEL_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_novel_seed42.py"
QUERY_POOL_PROTOCOL = EXPERIMENT / "query_pool_seed42_protocol.md"
LOCAL_OPERATOR_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_local_operator_search.py"
LOCAL_OPERATOR_PROTOCOL = EXPERIMENT / "local_operator_search_protocol.md"
RECURRENT_STATE_GPU = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "recurrent_graph_state_seed42"
    / "run_screen.py"
)
RECURRENT_STATE_METADATA = RECURRENT_STATE_GPU.with_name("kernel-metadata.json")
RECURRENT_STATE_ACCEPTANCE = (
    EXPERIMENT / "accept_pcqm100k_recurrent_graph_state.py"
)
RECURRENT_STATE_PROTOCOL = EXPERIMENT / "recurrent_graph_state_seed42_protocol.md"
SPARSE_TRIANGLE_GPU = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "sparse_triangle_edge_state_seed42"
    / "run_screen.py"
)
SPARSE_TRIANGLE_METADATA = SPARSE_TRIANGLE_GPU.with_name("kernel-metadata.json")
SPARSE_TRIANGLE_PROTOCOL = EXPERIMENT / "sparse_triangle_edge_state_protocol.md"
SPARSE_TRIANGLE_MULTISEED_GPU = (
    EXPERIMENT
    / "kaggle_pcqm_gap100k"
    / "sparse_triangle_edge_state_multiseed"
    / "run_confirmation.py"
)
SPARSE_TRIANGLE_MULTISEED_METADATA = SPARSE_TRIANGLE_MULTISEED_GPU.with_name(
    "kernel-metadata.json"
)
SPARSE_TRIANGLE_MULTISEED_ACCEPTANCE = (
    EXPERIMENT / "accept_pcqm100k_sparse_triangle_multiseed.py"
)
SPARSE_TRIANGLE_MULTISEED_PROTOCOL = (
    EXPERIMENT / "sparse_triangle_edge_state_multiseed_protocol.md"
)
GEOMETRY_BOTTOM_FUSION_PROTOCOL = (
    EXPERIMENT / "geometry_bottom_fusion_seed42_protocol.md"
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_sha256(indices) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_official_train_only_split_contract() -> None:
    assert assignment_literal(DATA, "OFFICIAL_TRAIN_ROWS") == 3_378_606
    assert assignment_literal(DATA, "SCREEN_TRAIN_ROWS") == 100_000
    assert assignment_literal(DATA, "SCREEN_VALIDATION_ROWS") == 10_000
    assert assignment_literal(DATA, "SCREEN_RESERVE_ROWS") == 1_024
    source = DATA.read_text(encoding="utf-8")
    assert "nrows=OFFICIAL_TRAIN_ROWS" in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "smiles2graph" in source
    assert "walk_length=RWSE_DIM" in source
    assert "_atomic_torch_save(part_path, graphs)" in source
    assert 'if key == "bond":' in source
    assert '"bondless_graphs": bondless_graphs' in source
    assert '"replacement_ledger_file": replacement_ledger_path.name' in source
    assert '"unresolved_graphs": 0' in source
    assert '"attempt_kind": attempt_kind' in source


def test_reserve_stream_preserves_the_frozen_initial_split() -> None:
    from molgap.pcqm_gap_data import fixed_screen_split

    split = fixed_screen_split()
    assert split["train_sha256"] == (
        "d08e04ef73090b77963a6959d7efa22d22a4085e3869a0e13086491b8f8c9678"
    )
    assert split["validation_sha256"] == (
        "40b210c03789249f89950d7eb9df6de93ec151903f75077cfa64fe0a94eca5b0"
    )
    assert len(split["reserve"]) == 1_024
    assert set(split["reserve"]).isdisjoint(split["train"])
    assert set(split["reserve"]).isdisjoint(split["validation"])


def test_gap_models_use_ogb_categories_and_one_target() -> None:
    source = MODELS.read_text(encoding="utf-8")
    assert "class OGBStructuralGPSWrapper" in source
    assert "class OGBEdgeStateStructuralGPSWrapper" in source
    assert "AtomEncoder" in source and "BondEncoder" in source
    assert '"n_targets": 1' in source
    assert '"num_layers": 9' in source
    assert '"hidden_channels": 192' in source
    assert "edge_state_channels=64" in source
    assert "class OGBQueryPoolStructuralGPSWrapper(OGBStructuralGPSWrapper)" in source
    assert "nn.MultiheadAttention" in source
    assert "num_pool_queries=4" in source
    assert 'candidate == "ogb_query_pool_structural_gps9"' in source
    assert "class OGBLocalOperatorStructuralGPSWrapper" in source
    assert '"ogb_gated_local_gps9": "resgated"' in source
    assert '"ogb_edge_attention_local_gps9": "transformer"' in source
    assert '"ogb_gen_local_gps9": "gen"' in source
    assert '"ogb_gatv2_local_gps9": "gatv2"' in source
    assert "class OGBGraphTokenStructuralGPSWrapper" in source
    assert 'candidate == "ogb_recurrent_graph_state_gps9"' in source
    assert "token_channels=16" in source
    assert "class OGBGeometrySparseTriangleEdgeStateGPSWrapper" in source
    assert '"ogb_distance_triangle_edge_state_gps9": "distance"' in source
    assert '"ogb_angle_triangle_edge_state_gps9": "angle"' in source
    assert (
        '"ogb_distance_angle_triangle_edge_state_gps9": "distance_angle"'
        in source
    )
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
    assert '"bondless_graphs": manifest.get("bondless_graphs")' in source
    assert '"replacement_count": manifest.get("replacement_count")' in source
    assert '"failed_graph_attempts": manifest.get("failed_graph_attempts")' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_cache_acceptance_verifies_audited_replacement(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("pcqm_cache_acceptance", ACCEPTANCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    initial_train = list(range(100_000))
    initial_validation = list(range(100_000, 110_000))
    reserve = list(range(110_000, 111_024))
    train = list(initial_train)
    train[17] = reserve[0]
    split = {
        "format": "molgap-pcqm-gap100k-split-v2",
        "official_train_rows": 3_378_606,
        "seed": 42,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "reserve": reserve,
        "train": train,
        "validation": initial_validation,
        "initial_train_sha256": _index_sha256(initial_train),
        "initial_validation_sha256": _index_sha256(initial_validation),
        "reserve_sha256": _index_sha256(reserve),
        "train_sha256": _index_sha256(train),
        "validation_sha256": _index_sha256(initial_validation),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    failures = {
        "format": "molgap-pcqm-gap100k-failures-v2",
        "attempts": [
            {
                "attempt_kind": "initial",
                "role": "train",
                "slot": 17,
                "row_index": 17,
                "type": "AttributeError",
                "message": "invalid molecule",
            }
        ],
    }
    ledger = {
        "format": "molgap-pcqm-gap100k-replacements-v1",
        "policy": "seed42-reserve-in-priority-order",
        "replacements": [
            {
                "role": "train",
                "slot": 17,
                "original_row_index": 17,
                "replacement_row_index": reserve[0],
                "reserve_rank": 0,
                "failure_type": "AttributeError",
                "failure_message": "invalid molecule",
            }
        ],
    }
    for name, payload in (
        ("split.json", split),
        ("failures.json", failures),
        ("replacement_ledger.json", ledger),
    ):
        (tmp_path / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    shards = []
    aggregate = hashlib.sha256()
    for role, parts in (("train", 20), ("validation", 2)):
        for part in range(parts):
            name = f"{role}_part_{part:03d}.pt"
            path = tmp_path / name
            path.write_bytes(f"{role}-{part}".encode("ascii"))
            digest = _sha256(path)
            record = {
                "role": role,
                "file": name,
                "source_start": part * 5_000,
                "source_stop": (part + 1) * 5_000,
                "graph_count": 5_000,
                "sha256": digest,
            }
            shards.append(record)
            aggregate.update(f"{role}\t{name}\t{digest}\n".encode("ascii"))
    manifest = {
        "format": "molgap-pcqm-gap100k-cache-v2",
        "complete": True,
        "source_dataset": "piero0/pcqm4mv2",
        "source_commit": "synthetic",
        "official_train_rows_read": 3_378_606,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "gpu_used": False,
        "unresolved_graphs": 0,
        "replacement_policy": "seed42-reserve-in-priority-order",
        "replacement_count": 1,
        "reserve_rows_consumed": 1,
        "failed_graph_attempts": 1,
        "bondless_graphs": 0,
        "train_graphs": 100_000,
        "validation_graphs": 10_000,
        "atom_feature_dim": 9,
        "bond_feature_dim": 3,
        "rwse_dim": 16,
        "feature_ranges": {
            "atom_feature_min": [0] * 9,
            "atom_feature_max": [1] * 9,
            "bond_feature_min": [0] * 3,
            "bond_feature_max": [1] * 3,
        },
        "split_file": "split.json",
        "split_file_sha256": _sha256(tmp_path / "split.json"),
        "failures_file": "failures.json",
        "failures_file_sha256": _sha256(tmp_path / "failures.json"),
        "replacement_ledger_file": "replacement_ledger.json",
        "replacement_ledger_sha256": _sha256(tmp_path / "replacement_ledger.json"),
        "initial_train_index_sha256": _index_sha256(initial_train),
        "initial_validation_index_sha256": _index_sha256(initial_validation),
        "reserve_index_sha256": _index_sha256(reserve),
        "train_index_sha256": _index_sha256(train),
        "validation_index_sha256": _index_sha256(initial_validation),
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    result = module.accept(tmp_path, "synthetic")
    assert result["accepted"] is True
    assert result["replacement_count"] == 1
    assert result["failed_graph_attempts"] == 1


def test_protocol_freezes_gap_only_kaggle_before_server() -> None:
    protocol = (EXPERIMENT / "pcqm100k_gap_screen_protocol.md").read_text(
        encoding="utf-8"
    )
    assert "official PCQM4Mv2 Gap task" in protocol
    assert "exactly `100,000`" in protocol
    assert "Seeds 43 and 44" in protocol
    assert "12 hours" in protocol
    assert "/lustre/home/users/sm2/chou/" in protocol


def test_gpu_screen_is_time_bounded_local_operator_sequence() -> None:
    assert assignment_literal(GPU, "EXPECTED_SOURCE_COMMIT") == (
        "bfaffe332d4c89dc041669679d6fb066e01bce1f"
    )
    assert assignment_literal(GPU, "EXPECTED_CACHE_SOURCE_COMMIT") == (
        "ba82461c53243d733474c8930ac1b86d82451c91"
    )
    assert assignment_literal(GPU, "EXPECTED_CACHE_SHA256") == (
        "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
    )
    assert assignment_literal(GPU, "CORE_CANDIDATES") == (
        "ogb_gated_local_gps9",
        "ogb_edge_attention_local_gps9",
        "ogb_gen_local_gps9",
    )
    assert assignment_literal(GPU, "OPTIONAL_CANDIDATE") == "ogb_gatv2_local_gps9"
    assert assignment_literal(GPU, "FROZEN_COMPARATOR") == {
        "candidate": "ogb_edge_state_structural_gps9",
        "validation_gap_mae_eV": 0.13798263211250306,
        "acceptance": "results/seed42_structural_vs_edge_state/acceptance.json",
    }
    assert assignment_literal(GPU, "BATCH_SIZE") == 48
    assert assignment_literal(GPU, "MAX_EPOCHS") == 40
    assert assignment_literal(GPU, "SEARCH_BUDGET_S") == 14_400
    source = GPU.read_text(encoding="utf-8")
    assert "for candidate in CORE_CANDIDATES" in source
    assert "remaining_s >= max(completed_durations) + OPTIONAL_BUFFER_S" in source
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


def test_novel_acceptance_executes_no_model_runtime() -> None:
    assert "torch" not in imported_modules(NOVEL_ACCEPTANCE)
    source = NOVEL_ACCEPTANCE.read_text(encoding="utf-8")
    assert 'CANDIDATE = "ogb_query_pool_structural_gps9"' in source
    assert 'COMPARATOR = "ogb_edge_state_structural_gps9"' in source
    assert "COMPARATOR_MAE = 0.13798263211250306" in source
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_query_pool_protocol_preserves_ogb_selection_boundary() -> None:
    source = QUERY_POOL_PROTOCOL.read_text(encoding="utf-8")
    assert "direct graph regression" in source
    assert "No external training data" in source
    assert "official test-dev access" in source
    assert "four-hour single-GPU plus single-CPU inference budget" in source
    assert "One candidate and one GPU task" in source


def test_local_operator_acceptance_executes_no_model_runtime() -> None:
    assert "torch" not in imported_modules(LOCAL_OPERATOR_ACCEPTANCE)
    source = LOCAL_OPERATOR_ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "len(rows) in (3, 4)" in source


def test_local_operator_protocol_freezes_three_plus_optional() -> None:
    source = LOCAL_OPERATOR_PROTOCOL.read_text(encoding="utf-8")
    assert "Official OGB categorical 2D" in source
    assert "at most 40 epochs, patience 8" in source
    assert "Parameter ceiling 5.2M" in source
    assert "The first three run sequentially" in source
    assert "No candidate may be retried" in source


def test_recurrent_graph_state_screen_preserves_frozen_contract() -> None:
    assert assignment_literal(RECURRENT_STATE_GPU, "EXPECTED_SOURCE_COMMIT") == (
        "96ca9ba22a021fcdd8fdf8daecfe60fc0878c5c8"
    )
    assert assignment_literal(RECURRENT_STATE_GPU, "EXPECTED_CACHE_SHA256") == (
        "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
    )
    assert assignment_literal(RECURRENT_STATE_GPU, "CANDIDATE") == (
        "ogb_recurrent_graph_state_gps9"
    )
    assert assignment_literal(RECURRENT_STATE_GPU, "BATCH_SIZE") == 48
    assert assignment_literal(RECURRENT_STATE_GPU, "LEARNING_RATE") == 1.6e-4
    assert assignment_literal(RECURRENT_STATE_GPU, "WEIGHT_DECAY") == 1.0e-6
    assert assignment_literal(RECURRENT_STATE_GPU, "MAX_EPOCHS") == 40
    assert assignment_literal(RECURRENT_STATE_GPU, "PATIENCE") == 8
    assert assignment_literal(RECURRENT_STATE_GPU, "PARAMETER_BUDGET") == 5_200_000
    assert assignment_literal(RECURRENT_STATE_GPU, "SEARCH_BUDGET_S") == 14_400
    source = RECURRENT_STATE_GPU.read_text(encoding="utf-8")
    assert '"precision": "fp32"' in source
    assert '"target": "homolumogap"' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    metadata = json.loads(RECURRENT_STATE_METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-gap100k-global-state-source"
    ]
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-official-pcqm-gap100k-r1-prep"
    ]


def test_recurrent_graph_state_acceptance_executes_no_model_runtime() -> None:
    assert "torch" not in imported_modules(RECURRENT_STATE_ACCEPTANCE)
    source = RECURRENT_STATE_ACCEPTANCE.read_text(encoding="utf-8")
    assert 'CANDIDATE = "ogb_recurrent_graph_state_gps9"' in source
    assert 'COMPARATOR = "ogb_edge_state_structural_gps9"' in source
    assert "COMPARATOR_MAE = 0.13798263211250306" in source
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source


def test_recurrent_graph_state_protocol_is_gap_only_and_single_candidate() -> None:
    source = RECURRENT_STATE_PROTOCOL.read_text(encoding="utf-8")
    assert "official PCQM4Mv2 task predicts Gap only" in source
    assert "One Kaggle GPU task" in source
    assert "Parameter ceiling 5.2M" in source
    assert "at most 40 epochs, patience 8" in source
    assert "official validation and test-dev remain unread" in source
    assert "token-width, seed, optimizer, schedule" in source


def test_sparse_triangle_hidden_width_does_not_depend_on_ogb_encoder_api() -> None:
    tree = ast.parse(MODELS.read_text(encoding="utf-8"))
    sparse_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "OGBSparseTriangleEdgeStateGPSWrapper"
    )
    source = ast.get_source_segment(MODELS.read_text(encoding="utf-8"), sparse_class)
    assert source is not None
    assert "self.head[0].in_features" in source
    assert "self.node_emb.out_features" not in source


def test_geometry_bottom_fusion_is_early_and_not_prediction_fusion() -> None:
    source = MODELS.read_text(encoding="utf-8")
    geometry_class = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and node.name == "OGBGeometrySparseTriangleEdgeStateGPSWrapper"
    )
    segment = ast.get_source_segment(source, geometry_class)
    assert segment is not None
    assert "edge_state + self.distance_initial" in segment
    assert "edge_state + self.distance_updates[layer]" in segment
    assert "wedge_state + self.angle_initial" in segment
    assert "wedge_state + self.angle_updates[layer]" in segment
    assert "SchNet" not in segment
    assert "residual_head" not in segment
    assert "prediction_fusion" not in segment


def test_geometry_cache_contract_is_deterministic_and_visible_on_failure() -> None:
    source = GEOMETRY.read_text(encoding="utf-8")
    assert "AllChem.ETKDGv3()" in source
    assert 'MMFF_VARIANT = "MMFF94s"' in source
    assert "geometry_seed(row_index)" in source
    assert "params.useRandomCoords = True" in source
    assert "geometry_valid=False" in source
    assert "failure_type=type(error).__name__" in source
    assert "np.zeros" in source


def test_geometry_screen_protocol_is_three_seed42_candidates_only() -> None:
    source = GEOMETRY_BOTTOM_FUSION_PROTOCOL.read_text(encoding="utf-8")
    assert "Seed 42 only" in source
    assert "three candidates run sequentially" in source
    assert "does not run additional seeds" in source
    assert "0.13790177369117737 eV" in source
    assert "There is no SchNet" in source
    lowered = source.lower()
    assert "official validation" in lowered
    assert "test-dev remain unread" in lowered


def test_sparse_triangle_retry_preserves_frozen_training_contract() -> None:
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "EXPECTED_SOURCE_COMMIT") == (
        "76dd6efa76c8236ce80a82a8a43d9f5df426165e"
    )
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "CANDIDATE") == (
        "ogb_sparse_triangle_edge_state_gps9"
    )
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "BATCH_SIZE") == 48
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "LEARNING_RATE") == 1.6e-4
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "WEIGHT_DECAY") == 1.0e-6
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "MAX_EPOCHS") == 40
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "PATIENCE") == 8
    assert assignment_literal(SPARSE_TRIANGLE_GPU, "PARAMETER_BUDGET") == 5_200_000
    source = SPARSE_TRIANGLE_GPU.read_text(encoding="utf-8")
    assert '"precision": "fp32"' in source
    assert '"target": "homolumogap"' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "pcqm_gap100k_sparse_triangle_edge_state_r3_seed42" in source
    metadata = json.loads(SPARSE_TRIANGLE_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == (
        "kaseichou/molgap-pcqm-triangle-edge-state-r3-s42"
    )
    assert len(metadata["title"]) <= 50
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-gap100k-sparse-triangle-source"
    ]
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pcqm-gap100k-sparse-triangle-wedge-cache-r2"
    ]
    protocol = SPARSE_TRIANGLE_PROTOCOL.read_text(encoding="utf-8")
    assert "Seed 42, FP32, batch 48" in protocol
    assert "Parameter ceiling: `5,200,000`" in protocol
    assert "one R3 implementation-only retry" in protocol


def test_sparse_triangle_multiseed_is_paired_and_frozen() -> None:
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "EXPECTED_SOURCE_COMMIT") == (
        "76dd6efa76c8236ce80a82a8a43d9f5df426165e"
    )
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "CANDIDATES") == (
        "ogb_edge_state_structural_gps9",
        "ogb_sparse_triangle_edge_state_gps9",
    )
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "SEEDS") == (43, 44)
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "BATCH_SIZE") == 48
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "LEARNING_RATE") == 1.6e-4
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "WEIGHT_DECAY") == 1.0e-6
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "MAX_EPOCHS") == 40
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "PATIENCE") == 8
    assert assignment_literal(SPARSE_TRIANGLE_MULTISEED_GPU, "SEARCH_BUDGET_S") == 39_600
    source = SPARSE_TRIANGLE_MULTISEED_GPU.read_text(encoding="utf-8")
    assert "for seed in SEEDS" in source
    assert "for candidate in CANDIDATES" in source
    assert '"precision": "fp32"' in source
    assert '"target": "homolumogap"' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    metadata = json.loads(SPARSE_TRIANGLE_MULTISEED_METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-pcqm-triangle-r3-confirm-s4344"
    assert len(metadata["title"]) <= 50
    assert metadata["dataset_sources"] == [
        "kaseichou/molgap-pcqm-gap100k-sparse-triangle-source"
    ]
    assert metadata["kernel_sources"] == [
        "kaseichou/molgap-pcqm-gap100k-sparse-triangle-wedge-cache-r2"
    ]


def test_sparse_triangle_multiseed_acceptance_is_no_inference() -> None:
    assert "torch" not in imported_modules(SPARSE_TRIANGLE_MULTISEED_ACCEPTANCE)
    source = SPARSE_TRIANGLE_MULTISEED_ACCEPTANCE.read_text(encoding="utf-8")
    assert '"model_inference_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    protocol = SPARSE_TRIANGLE_MULTISEED_PROTOCOL.read_text(encoding="utf-8")
    assert "fresh comparator and candidate" in protocol
    assert "seeds 42, 43, and 44" in protocol
    assert "Only one GPU kernel" in protocol


def test_sparse_triangle_multiseed_acceptance_replays_pair_arithmetic(
    tmp_path: Path,
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sparse_triangle_multiseed_acceptance",
        SPARSE_TRIANGLE_MULTISEED_ACCEPTANCE,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source_commit = "synthetic-source"
    cache_sha = "a" * 64
    values = {
        (43, "ogb_edge_state_structural_gps9"): 0.1400,
        (43, "ogb_sparse_triangle_edge_state_gps9"): 0.1390,
        (44, "ogb_edge_state_structural_gps9"): 0.1410,
        (44, "ogb_sparse_triangle_edge_state_gps9"): 0.1405,
    }
    runs = []
    for (seed, candidate), value in values.items():
        run_dir = tmp_path / "results" / f"seed{seed}" / candidate
        run_dir.mkdir(parents=True)
        artifacts = {}
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            suffix = ".json" if name == "trace" else ".pt"
            path = run_dir / f"{name}{suffix}"
            path.write_bytes(f"{seed}-{candidate}-{name}".encode("utf-8"))
            artifacts[name] = str(path.relative_to(tmp_path)).replace("\\", "/")
            artifacts[f"{name}_sha256"] = _sha256(path)
        metrics = {
            "format": "molgap-pcqm-gap100k-sparse-triangle-paired-run-v1",
            "complete": True,
            "candidate": candidate,
            "source_commit": source_commit,
            "seed": seed,
            "parameter_count": 4_800_000,
            "parameter_budget": 5_200_000,
            "best_epoch": 19,
            "validation_gap_mae_eV": value,
            "validation_rows": 10_000,
            "validation_row_index_sha256": "b" * 64,
            "validation_target_sha256": "c" * 64,
            "target_mean_eV": 5.5,
            "target_std_eV": 1.2,
            "epochs_completed": 20,
            "training_elapsed_s": 100.0,
            "contract": {
                "batch_size": 48,
                "learning_rate": 1.6e-4,
                "weight_decay": 1.0e-6,
                "max_epochs": 40,
                "patience": 8,
                "precision": "fp32",
                "target": "homolumogap",
            },
            "artifacts": artifacts,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        runs.append(metrics)

    pairs = [
        {
            "seed": 42,
            "comparator_validation_gap_mae_eV": 0.13798263211250306,
            "candidate_validation_gap_mae_eV": 0.13790177369117737,
            "candidate_minus_comparator_eV": (
                0.13790177369117737 - 0.13798263211250306
            ),
        }
    ]
    for seed in (43, 44):
        comparator = values[(seed, "ogb_edge_state_structural_gps9")]
        candidate = values[(seed, "ogb_sparse_triangle_edge_state_gps9")]
        pairs.append(
            {
                "seed": seed,
                "comparator_validation_gap_mae_eV": comparator,
                "candidate_validation_gap_mae_eV": candidate,
                "candidate_minus_comparator_eV": candidate - comparator,
            }
        )
    mean_comparator = sum(
        row["comparator_validation_gap_mae_eV"] for row in pairs
    ) / 3
    mean_candidate = sum(
        row["candidate_validation_gap_mae_eV"] for row in pairs
    ) / 3
    paired_comparison = [
        *pairs,
        {
            "seed": "mean",
            "comparator_validation_gap_mae_eV": mean_comparator,
            "candidate_validation_gap_mae_eV": mean_candidate,
            "candidate_minus_comparator_eV": mean_candidate - mean_comparator,
        },
    ]
    selection = {
        "format": "molgap-pcqm-gap100k-sparse-triangle-multiseed-v1",
        "complete": True,
        "source_commit": source_commit,
        "cache_aggregate_sha256": cache_sha,
        "parent_cache_aggregate_sha256": "d" * 64,
        "seeds": [43, 44],
        "candidates": [
            "ogb_edge_state_structural_gps9",
            "ogb_sparse_triangle_edge_state_gps9",
        ],
        "preflight": [
            {
                "candidate": candidate,
                "parameter_count": 4_800_000,
                "finite_prediction": True,
                "finite_loss": True,
                "finite_gradients": True,
            }
            for candidate in (
                "ogb_edge_state_structural_gps9",
                "ogb_sparse_triangle_edge_state_gps9",
            )
        ],
        "runs": runs,
        "paired_comparison": paired_comparison,
        "multiseed_gate_passed": True,
        "selected_candidate": "ogb_sparse_triangle_edge_state_gps9",
        "search_budget_s": 39_600,
        "elapsed_s": 400.0,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    (tmp_path / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    result = module.accept(tmp_path, source_commit, cache_sha)
    assert result["accepted"] is True
    assert result["multiseed_gate_passed"] is True
