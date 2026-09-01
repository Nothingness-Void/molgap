from __future__ import annotations

import ast
import hashlib
import json
import importlib.util
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
    assert "shared_backbone_parameters_match" in runner
    assert "torsion_injection_zero" in runner
    assert "torch.equal(value.detach(), reference)" in runner
    assert "torch.count_nonzero(value.detach()).item() != 0" in runner
    assert "torch.allclose(" not in runner
    assert '"official_validation_role_read": False' in runner
    assert '"test_dev_role_read": False' in runner
    assert "piero0/pcqm4mv2" not in runner
    assert "hydrate_resume_state(source_commit, cache_manifest[\"aggregate_sha256\"])" in runner
    assert "candidate_checkpoint_epoch" in runner
    assert "train_generator_state" in runner
    assert "cpu_byte_rng_state" in runner
    assert "value.detach().cpu().contiguous()" in runner


def test_torsion_resume_bundle_is_hash_checked_and_hydrated(tmp_path):
    runner_path = ROOT / (
        "experiments/pcqm_gap_architecture/kaggle_pcqm_gap100k/"
        "sparse_torsion_edge_state_seed42/run_torsion_screen.py"
    )
    spec = importlib.util.spec_from_file_location("torsion_resume_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source_commit = "3" * 40
    cache_sha = "4" * 64
    input_root = tmp_path / "input"
    bundle = input_root / "resume"
    output = tmp_path / "output"
    artifacts = []
    expected = [
        f"results/{module.COMPARATOR}/best_model.pt",
        f"results/{module.COMPARATOR}/checkpoint.pt",
        f"results/{module.COMPARATOR}/metrics.json",
        f"results/{module.COMPARATOR}/trace.json",
        f"results/{module.COMPARATOR}/validation_payload.pt",
        f"results/{module.CANDIDATE}/best_model.pt",
        f"results/{module.CANDIDATE}/checkpoint.pt",
        f"results/{module.CANDIDATE}/trace.json",
    ]
    for index, relative in enumerate(expected):
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"artifact-{index}".encode("ascii")
        path.write_bytes(payload)
        artifacts.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "format": module.RESUME_FORMAT,
        "complete": True,
        "source_commit": source_commit,
        "torsion_cache_aggregate_sha256": cache_sha,
        "source_kernel": "nothingnessvoid/molgap-pcqm-sparse-torsion-s42",
        "source_kernel_version": 3,
        "comparator_complete": True,
        "comparator_checkpoint_epoch": 39,
        "candidate_checkpoint_epoch": 38,
        "candidate_best_epoch": 36,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "artifacts": artifacts,
    }
    manifest_path = bundle / "resume_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module.EXPECTED_RESUME_MANIFEST_SHA256 = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    accepted = module.hydrate_resume_state(
        source_commit,
        cache_sha,
        input_root=input_root,
        output_root=output,
    )
    assert accepted["complete"] is True
    assert accepted["candidate_checkpoint_epoch"] == 38
    for relative in expected:
        assert (output / relative).read_bytes() == (bundle / relative).read_bytes()


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
        "nothingnessvoid/molgap-pcqm-sparse-torsion-s42-resume-v3",
    ]


def test_torsion_acceptance_requires_exact_resume_evidence():
    acceptance = read(
        "experiments/pcqm_gap_architecture/accept_pcqm100k_torsion_seed42.py"
    )
    assert 'root / "resume_acceptance.json"' in acceptance
    assert '"source_kernel_version": 3' in acceptance
    assert '"comparator_checkpoint_epoch": 39' in acceptance
    assert '"candidate_checkpoint_epoch": 38' in acceptance
    assert '"artifact_count": 8' in acceptance
    assert '"resume_verified"' in acceptance
