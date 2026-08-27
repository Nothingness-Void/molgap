from __future__ import annotations

import ast
import hashlib
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
NOVEL_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_novel_seed42.py"
QUERY_POOL_PROTOCOL = EXPERIMENT / "query_pool_seed42_protocol.md"
LOCAL_OPERATOR_ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_local_operator_search.py"
LOCAL_OPERATOR_PROTOCOL = EXPERIMENT / "local_operator_search_protocol.md"


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
        "__PIN_AFTER_SOURCE_COMMIT__"
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
