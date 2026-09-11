from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np

from molgap.pcqm_k1_full_runner import (
    ARCHITECTURE_FILE_SHA256,
    BATCHES_PER_PASS,
    EXPECTED_INITIAL_MODEL_SHA256,
    FULL_MANIFEST_CANONICAL_SHA256,
    MAX_OPTIMIZER_STEPS,
    PHYSICAL_BATCH,
    REFERENCE_EPOCHS,
    REFERENCE_TRAIN_ROWS,
    SAMPLE_PRESENTATIONS,
    TAIL_ROWS_PER_PASS,
    TRAINING_CONTRACT,
    TRAINING_CONTRACT_SHA256,
    DeterministicPassBatchSampler,
    _canonical_manifest_sha256,
    progress_from_step,
    validate_source_archive,
)
from molgap.training_reproducibility import canonical_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_schedule_preserves_reference_sample_exposure_exactly():
    assert SAMPLE_PRESENTATIONS == REFERENCE_TRAIN_ROWS * REFERENCE_EPOCHS
    assert SAMPLE_PRESENTATIONS == 20_000_000
    assert MAX_OPTIMIZER_STEPS == 156_250
    assert MAX_OPTIMIZER_STEPS * PHYSICAL_BATCH == SAMPLE_PRESENTATIONS
    assert BATCHES_PER_PASS == 26_395
    assert TAIL_ROWS_PER_PASS == 46
    assert progress_from_step(MAX_OPTIMIZER_STEPS) == (5, 24_275)
    assert TRAINING_CONTRACT["tail_batch_policy"] == "drop_last-global-pass"
    assert TRAINING_CONTRACT["scheduler"] == "cosine-per-optimizer-step"
    assert TRAINING_CONTRACT["selection"].startswith("fixed-final-step")
    assert (
        TRAINING_CONTRACT["seed42_initial_model_sha256"]
        == EXPECTED_INITIAL_MODEL_SHA256
    )
    assert TRAINING_CONTRACT["finite_check_every_optimizer_steps"] == 50
    assert canonical_fingerprint(TRAINING_CONTRACT) == TRAINING_CONTRACT_SHA256
    frozen = json.loads(
        (REPO_ROOT / "experiments/pcqm_k1_full/training_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen == TRAINING_CONTRACT


def test_resumable_sampler_never_emits_a_partial_batch():
    complete = list(DeterministicPassBatchSampler(300, pass_index=2))
    resumed = list(
        DeterministicPassBatchSampler(300, pass_index=2, start_batch=1)
    )
    assert len(complete) == 2
    assert all(len(batch) == PHYSICAL_BATCH for batch in complete)
    assert resumed == complete[1:]
    flattened = [index for batch in complete for index in batch]
    assert len(flattened) == len(set(flattened)) == 256
    assert min(flattened) >= 0 and max(flattened) < 300


def test_full_manifest_identity_is_logical_not_line_ending_dependent():
    manifest = json.loads(
        (
            REPO_ROOT
            / "platforms/_records/ims/pcqm_fixed_datasets_v1/ogb-train-full.manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert _canonical_manifest_sha256(manifest) == FULL_MANIFEST_CANONICAL_SHA256
    assert manifest["aggregates"]["topology"] == TRAINING_CONTRACT[
        "topology_aggregate_sha256"
    ]


def test_source_archiver_is_reproducible_and_excludes_generated_metadata(tmp_path):
    script = REPO_ROOT / "experiments/pcqm_k1_scale500k/package_source_dataset.py"
    spec = importlib.util.spec_from_file_location("k1_source_packager", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = [
        REPO_ROOT / "src/molgap/screen_policy.py",
        REPO_ROOT / "src/molgap/training_reproducibility.py",
    ]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_sha = module.write_reproducible_archive(first, selected)
    second_sha = module.write_reproducible_archive(second, selected)
    assert first_sha == second_sha
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert not any("__pycache__" in name or ".egg-info" in name for name in names)


def test_source_archive_is_bound_to_commit_hash_and_inventory(tmp_path):
    script = REPO_ROOT / "experiments/pcqm_k1_scale500k/package_source_dataset.py"
    spec = importlib.util.spec_from_file_location("k1_source_packager_bound", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = [REPO_ROOT / relative for relative in ARCHITECTURE_FILE_SHA256]
    archive = tmp_path / "src.zip"
    archive_sha256 = module.write_reproducible_archive(archive, files)
    commit = "1" * 40
    (tmp_path / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    (tmp_path / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        archive_sha256 + "\n", encoding="utf-8"
    )
    inventory = [path.relative_to(REPO_ROOT).as_posix() for path in files]
    (tmp_path / "SOURCE_FILES.json").write_text(
        json.dumps(sorted(inventory), indent=2) + "\n", encoding="utf-8"
    )
    assert validate_source_archive(archive, commit) == archive_sha256


def test_rng_state_round_trip_restores_python_numpy_and_torch():
    import random

    import torch

    from molgap.training_reproducibility import capture_rng_state, restore_rng_state

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
