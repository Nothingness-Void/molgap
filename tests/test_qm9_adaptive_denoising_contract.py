from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "qm9_adaptive_denoising.py"
EXPERIMENT = ROOT / "experiments" / "qm9_adaptive_denoising"
CACHE_RUNNER = EXPERIMENT / "cache_prep" / "run_cache.py"
GPU_RUNNER = EXPERIMENT / "gpu_seed42" / "run_screen.py"
CACHE_ACCEPTANCE = EXPERIMENT / "accept_cache.py"
SCREEN_ACCEPTANCE = EXPERIMENT / "accept.py"
SOURCE_PACKAGER = EXPERIMENT / "package_source_dataset.py"
CACHE_METADATA = EXPERIMENT / "cache_prep" / "kernel-metadata.json"
GPU_METADATA = EXPERIMENT / "gpu_seed42" / "kernel-metadata.json"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _literal_assignments(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in _tree(path).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def test_execution_layer_and_wrappers_parse_without_running_a_model() -> None:
    for path in (
        MODULE,
        CACHE_RUNNER,
        GPU_RUNNER,
        CACHE_ACCEPTANCE,
        SCREEN_ACCEPTANCE,
        SOURCE_PACKAGER,
    ):
        ast.parse(_source(path), filename=str(path))


def test_reuses_the_accepted_gape_roles_and_freezes_geometry_cache_contract() -> None:
    values = _literal_assignments(MODULE)
    source = _source(MODULE)
    assert values["TRAIN_ROWS"] == 30_000
    assert values["VALIDATION_ROWS"] == 3_000
    assert values["TOTAL_SELECTED_ROWS"] == 33_000
    assert values["SPLIT_FINGERPRINT"] == "62f1cdefdaec6877"
    assert values["PARENT_CACHE_AGGREGATE_SHA256"] == (
        "80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340"
    )
    assert values["EXPECTED_SOURCE_ROWS"] == 130_831
    assert values["EXPECTED_RDKIT_VERSION"] == "2023.09.6"
    assert values["EXPECTED_PROCESSED_SHA256"] == (
        "90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721"
    )
    assert values["EXPECTED_RAW_SDF_SHA256"] == (
        "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e"
    )
    assert "qm9_gape" in source
    assert "qm9_local_hierarchy" in source
    assert "compute_etkdg_geometry" in source
    assert "ETKDGv3" in source
    assert "MMFF94s" in source
    assert "dft_coordinates_used" in source
    assert "canonical_validity.json" in source
    assert "_verify_parent_graph_identity" in source
    assert "node_feat" in source and "edge_feat" in source
    assert "geometry_failure_mask" in source
    assert "geometry_failure_type" in source
    assert "test_role_read" in source
    assert "roles[\"test\"]" not in source
    assert "GetConformer" not in source


def test_cache_is_sharded_atomic_and_model_free() -> None:
    source = _source(MODULE)
    cache_runner = _source(CACHE_RUNNER)
    acceptance = _source(CACHE_ACCEPTANCE)
    for required in (
        "atomic_torch_save",
        "progress.json",
        "geometry_failures.json",
        "aggregate_sha256",
        "sha256_file",
        "completed_shards",
        "completed_by_file",
        "gpu_used",
        "model_inference_executed",
    ):
        assert required in source
    assert "build_cache" in cache_runner
    assert "enable_gpu" in _source(CACHE_METADATA)
    assert '"false"' in _source(CACHE_METADATA)
    assert "make_encoder" not in acceptance
    assert "forward(" not in acceptance
    assert '"test_role_read": False' in acceptance


def test_downstream_is_the_same_distance_angle_edgestate_gps9_encoder() -> None:
    values = _literal_assignments(MODULE)
    source = _source(MODULE)
    assert values["BATCH_SIZE"] == 128
    assert values["RWSE_DIM"] == 16
    assert values["ATOM_FEATURE_CHANNELS"] == 9
    assert values["BOND_FEATURE_CHANNELS"] == 3
    assert values["HIDDEN_CHANNELS"] == 192
    assert values["NUM_LAYERS"] == 9
    assert values["EDGE_STATE_CHANNELS"] == 64
    assert values["WEDGE_CHANNELS"] == 16
    assert "OGBGeometrySparseTriangleEdgeStateGPSWrapper" in source
    assert 'geometry_mode="distance_angle"' in source
    assert "wedge_channels=16" in source
    assert "geometry_basis_channels=16" in source
    assert "random_walk_pe" in source
    assert "edge_distance" in source
    assert "wedge_angle_cos" in source
    assert "geometry_valid" in source


def test_three_arms_and_exact_training_budget_are_frozen() -> None:
    values = _literal_assignments(MODULE)
    source = _source(MODULE)
    assert values["SEED"] == 42
    assert values["SCRATCH_EPOCHS"] == 40
    assert values["PRETRAIN_EPOCHS"] == 10
    assert values["GAP_EPOCHS"] == 30
    assert values["TOTAL_ENCODER_EPOCHS"] == 40
    assert values["LEARNING_RATE"] == 4e-4
    assert values["WEIGHT_DECAY"] == 1e-5
    assert values["MIN_LEARNING_RATE"] == 1e-6
    assert values["DENOISING_SIGMA"] == 0.1
    assert values["DENOISING_PRIOR_SIGMA"] == 0.1
    assert values["KL_WEIGHT"] == 1.0
    for arm in ("scratch40", "fixed10_gap30", "adaptive10_gap30"):
        assert f'"{arm}"' in source
    assert "AdamW" in source
    assert "CosineAnnealingLR" in source
    assert "validate_screen_arm" in source
    assert "gradient_accumulation_steps" in source
    assert "direct_gap" in source
    assert "roles[\"validation\"]" in source
    assert "roles[\"test\"]" not in source


def test_adaptive_noise_is_invariant_atom_specific_reparameterized_and_regularized() -> None:
    source = _source(MODULE)
    for required in (
        "InvariantNoiseGenerator",
        "EquivariantDenoisingHead",
        "clean_local_environment",
        "log_sigma",
        "reparameterized",
        "torch.randn",
        "KL_WEIGHT",
        "kl_loss",
        "exp()",
        "adaptive_noise_generator",
        "unit_vectors",
    ):
        assert required in source
    assert "DENOISING_SIGMA" in source
    assert "same_denoising_head" in source or "denoising_head_initial_sha256" in source
    assert "pretrain_only" in source
    assert "inference_parameter_count" in source


def test_atomic_checkpoints_and_runtime_evidence_are_reported() -> None:
    source = _source(MODULE)
    for required in (
        '"optimizer"',
        '"scheduler"',
        '"rng"',
        '"best_epoch"',
        '"pretrain_loss"',
        '"sigma_mean"',
        '"sigma_std"',
        '"parameter_count"',
        '"throughput_graphs_per_s"',
        '"peak_memory_mib"',
        '"test_role_read": False',
        "os.replace(temporary, path)",
    ):
        assert required in source
    assert "best_validation_payload.pt" in source
    assert "completion_manifest.json" in source


def test_gpu_wrapper_requests_t4x2_and_serializes_gpu1_denoising_arms() -> None:
    metadata = json.loads(_source(GPU_METADATA))
    source = _source(GPU_RUNNER)
    assert metadata["enable_gpu"] == "true"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert len(metadata["dataset_sources"]) == 2
    assert '"0" if arm == "scratch40" else "1"' in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "torch.cuda.device_count() != 2" in source
    assert "fixed10_gap30" in source
    assert "adaptive10_gap30" in source
    assert "process.wait()" in source
    assert "PENDING_CPU_ACCEPTANCE" in source
    assert "EXPECTED_CACHE_MANIFEST_SHA256" in source
    assert 'acceptance.get("accepted") is not True' in source
    assert "official" not in source.lower()
    assert "/lustre" not in source


def test_no_model_acceptance_recomputes_validation_and_checks_all_paired_fields() -> None:
    cache_acceptance = _source(CACHE_ACCEPTANCE)
    screen_acceptance = _source(SCREEN_ACCEPTANCE)
    for source in (cache_acceptance, screen_acceptance):
        assert "model_inference_executed" in source
        assert '"test_role_read": False' in source
        assert "sha256_file" in source
        assert "official_pcqm_roles_read" in source
        assert "make_encoder" not in source
    assert "validate_paired_screen_contract" in screen_acceptance
    assert "validation_gap_mae_eV" in screen_acceptance
    assert "best_validation_payload.pt" in screen_acceptance
    assert "torch.load" in screen_acceptance
    assert "sigma_mean" in screen_acceptance
    assert "throughput_graphs_per_s" in screen_acceptance
    assert "peak_memory_mib" in screen_acceptance
    assert "_validate_stage" in screen_acceptance
    assert '"precision_verified": True' in screen_acceptance


def test_source_packaging_keeps_reusable_logic_out_of_thin_wrappers() -> None:
    source = _source(SOURCE_PACKAGER)
    for required in (
        "src",
        "molgap",
        "SOURCE_COMMIT.txt",
        "SOURCE_TREE_SHA256.txt",
        "dataset-metadata.json",
        "git",
        "status",
        "--porcelain",
    ):
        assert required in source
    assert "make_encoder" not in _source(CACHE_RUNNER)
    assert "make_encoder" not in _source(GPU_RUNNER)


def test_source_tree_digest_uses_cross_platform_relative_path_order(tmp_path: Path) -> None:
    package = tmp_path / "molgap"
    (package / "archive").mkdir(parents=True)
    (package / "archive" / "README.md").write_bytes(b"upper\n")
    (package / "archive" / "__init__.py").write_bytes(b"lower\n")

    spec = importlib.util.spec_from_file_location("adaptive_packager", SOURCE_PACKAGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    digest = hashlib.sha256()
    for relative in ("archive/README.md", "archive/__init__.py"):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((package / relative).read_bytes()).digest())
    assert module.tree_sha256(package) == digest.hexdigest()

    for runner in (CACHE_RUNNER, GPU_RUNNER):
        source = _source(runner)
        assert "key=lambda item: item[0]" in source
