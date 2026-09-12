import hashlib
import json
import tarfile
from pathlib import Path

import torch

from molgap.pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH,
    EXPECTED_ARCHITECTURE_SHA256,
    EXPECTED_INITIAL_MODEL_SHA256,
    EXPECTED_PARAMETERS,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    DeterministicEpochBatchSampler,
    FrozenEpochScheduler,
    _forward,
    _make_model,
    _scientific_fields,
    _source_sha256,
    _state_sha256,
    _target_stats,
    validate_source_archive,
)
from molgap.gptrans import OGBGPTransTiny
from molgap.screen_policy import REFERENCE_MATCH_FIELDS, validate_screen_arm
from molgap.training_reproducibility import configure_fp32_determinism, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_model_identity_and_backward():
    configure_fp32_determinism(42)
    model = _make_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS
    assert _state_sha256(model) == EXPECTED_INITIAL_MODEL_SHA256
    assert _source_sha256(ROOT / "src/molgap/gptrans.py") == EXPECTED_ARCHITECTURE_SHA256

    x = torch.zeros((6, 9), dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]], dtype=torch.long
    )
    edge_attr = torch.zeros((edge_index.shape[1], 3), dtype=torch.long)
    batch = type("Batch", (), {})()
    batch.x = x
    batch.edge_index = edge_index
    batch.edge_attr = edge_attr
    batch.batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    prediction = _forward(model, batch)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    prediction.abs().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_global_sampler_has_only_full_unique_batches():
    first = DeterministicEpochBatchSampler(100_000, 0)
    second = DeterministicEpochBatchSampler(100_000, 0)
    batches = list(first)
    assert batches == list(second)
    assert len(batches) == BATCHES_PER_EPOCH == 781
    assert all(len(batch) == PHYSICAL_BATCH for batch in batches)
    flattened = [index for batch in batches for index in batch]
    assert len(flattened) == len(set(flattened)) == 99_968
    assert SAMPLE_PRESENTATIONS == 5_998_080
    next_epoch = list(DeterministicEpochBatchSampler(100_000, 1))
    assert len(next_epoch) == 781
    assert next_epoch[0] != batches[0]


def test_local_edge_offsets_match_bincount_reference():
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    edge_index = torch.tensor(
        [[0, 1, 3, 4, 5, 8], [1, 2, 4, 3, 8, 5]], dtype=torch.long
    )
    edge_batch, edge_src, edge_dst = OGBGPTransTiny._local_edges(
        edge_index, batch, int(batch.numel())
    )
    counts = torch.bincount(batch, minlength=3)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    local = torch.arange(batch.numel()) - offsets[batch]
    assert torch.equal(edge_batch, batch[edge_index[0]])
    assert torch.equal(edge_src, local[edge_index[0]])
    assert torch.equal(edge_dst, local[edge_index[1]])


def test_schedule_and_contract_are_frozen():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    scheduler = FrozenEpochScheduler(optimizer)
    assert scheduler.step(0) == 0.00025
    assert scheduler.step(3) == 0.001
    assert scheduler.step(59) == 0.000001
    assert scheduler.state_dict() == {"epoch": 59}

    fields = _scientific_fields()
    assert all(field in fields for field in REFERENCE_MATCH_FIELDS)
    assert fields["tail_batch_policy"] == "drop_last"
    assert fields["sample_exposure"] == SAMPLE_PRESENTATIONS
    assert validate_screen_arm(physical_batch_per_device=128)["effective_batch_per_optimizer_step"] == 128

    contract = json.loads(
        (ROOT / "experiments/pcqm_gptrans_t_100k_v4/training_contract.json").read_text()
    )
    for key, value in fields.items():
        assert contract[key] == value


def test_target_stats_use_legacy_torch_compatible_sample_std():
    shard = type("Shard", (), {})()
    shard._data = type("Data", (), {"y": torch.arange(100_000, dtype=torch.float64)})()
    mean, std = _target_stats([shard])
    assert mean == float(shard._data.y.mean())
    assert std == float(shard._data.y.std(unbiased=True))


def test_remote_payload_requires_v4_guards():
    slurm = (ROOT / "platforms/scnet/pcqm_gptrans_t_100k_v4_kunshan.slurm").read_text()
    assert "--gres=dcu:Hygon:1" in slurm
    assert "--cpus-per-task=8" in slurm
    assert "test -f \"$PREFLIGHT\"" in slurm
    assert "SOURCE_ARCHIVE_SHA256" in slurm
    runner = (ROOT / "src/molgap/pcqm_gptrans_v4.py").read_text()
    assert "configure_fp32_determinism" in runner
    assert "validate_runtime_certificate" in runner
    assert "Optimizer received a non-128 batch" in runner


def test_source_archive_is_bound_to_commit_and_inventory(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("fixed source\n", encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="payload.txt")
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "SOURCE_COMMIT.txt").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "SOURCE_ARCHIVE_SHA256.txt").write_text(
        archive_sha + "\n", encoding="utf-8"
    )
    (tmp_path / "SOURCE_FILES.json").write_text(
        json.dumps({"files": [{"path": "payload.txt", "sha256": payload_sha}]}),
        encoding="utf-8",
    )
    assert validate_source_archive(archive, archive_sha, "abc123") == archive_sha
