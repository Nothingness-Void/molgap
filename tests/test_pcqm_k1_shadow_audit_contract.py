from __future__ import annotations

import ast
import json
from pathlib import Path

from molgap.pcqm_k1_shadow_audit import (
    BATCH_SIZE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_FULL_MODEL_SHA256,
    EXPECTED_K1_MODEL_SHA256,
    EXPECTED_SHADOW_CACHE_SHA256,
    MAX_INFERENCE_TIME_RATIO,
    MIN_MEMORY_RESERVE,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "molgap" / "pcqm_k1_shadow_audit.py"
ENTRY = ROOT / "experiments" / "pcqm_k1_shadow" / "gpu_audit" / "run_audit.py"
METADATA = ENTRY.with_name("kernel-metadata.json")
PROTOCOL = ROOT / "experiments" / "pcqm_k1_shadow" / "protocol.md"


def test_audit_identity_and_resource_gate_are_frozen():
    assert BATCH_SIZE == 128
    assert BOOTSTRAP_SEED == 2_026_091_142
    assert BOOTSTRAP_REPLICATES == 20_000
    assert MAX_INFERENCE_TIME_RATIO == 1.25
    assert MIN_MEMORY_RESERVE == 0.15
    assert EXPECTED_SHADOW_CACHE_SHA256 == (
        "4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44"
    )
    assert EXPECTED_FULL_MODEL_SHA256 == (
        "467753c8caa26e3e8d537aa7933e693073fcc45851caf561174d604742d48e6a"
    )
    assert EXPECTED_K1_MODEL_SHA256 == (
        "9e9ac63a9887030dcfbdc795d843cc10e70f8d9dcec7421a13cff98970203784"
    )


def test_predictions_are_frozen_before_label_access():
    source = MODULE.read_text(encoding="utf-8")
    ast.parse(source)
    assert source.index('output / "predictions_before_label_read.pt"') < source.index(
        "labels = read_shadow_labels_once"
    )
    assert "torch.optim" not in source
    assert '"training_executed": False' in source
    assert '"official_validation_role_read": False' in source
    assert '"test_dev_role_read": False' in source
    assert "graph.row_id = graph.row_index" in source
    assert "batch.row_id" in source
    assert "batch.row_index" not in source
    assert 'if "y" in graph:' in source
    assert 'hasattr(graph, "y")' not in source


def test_gpu_kernel_uses_only_frozen_inputs():
    ast.parse(ENTRY.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] == "true"
    assert metadata["dataset_sources"] == [
        "piero0/pcqm4mv2",
        "kaseichou/molgap-pcqm-k1-shadow-audit-source",
        "kaseichou/molgap-pcqm-k1-shadow-cache-v2",
        "kaseichou/molgap-pcqm-k1-shadow-checkpoints",
    ]


def test_protocol_freezes_one_time_gate_before_label_read():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "20,000 row-level replicates" in text
    assert "2026091142" in text
    assert "at most `1.25`" in text
    assert "at least `15%`" in text
