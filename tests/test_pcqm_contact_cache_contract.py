from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pcqm_gap_architecture"
RUNNER = EXPERIMENT / "kaggle_pcqm_gap100k" / "contact_state_cache" / "run_contact_state_cache.py"
METADATA = RUNNER.with_name("kernel-metadata.json")
ACCEPTANCE = EXPERIMENT / "accept_pcqm100k_contact_state_cache.py"
PROTOCOL = EXPERIMENT / "contact_state_seed42_protocol.md"
MODULE = ROOT / "src" / "molgap" / "pcqm_contact.py"


def test_contact_cache_sources_parse() -> None:
    ast.parse(RUNNER.read_text(encoding="utf-8"))
    ast.parse(ACCEPTANCE.read_text(encoding="utf-8"))


def test_contact_cache_metadata_is_cpu_only_and_private() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["id"] == "kaseichou/molgap-pcqm-contactstate-cache-s42"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert "kaseichou/molgap-pcqm-contactstate-source" in metadata["dataset_sources"]
    assert "kaseichou/molgap-pcqm-geometry-cache-s42-dataset" in metadata["dataset_sources"]


def test_contact_cache_contract_is_frozen_in_code_and_protocol() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    acceptance = ACCEPTANCE.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    module = MODULE.read_text(encoding="utf-8")
    for token in ("EXCLUDED_COVALENT_HOPS", "10_000_000", "official_validation_role_read", "test_dev_role_read"):
        assert token in runner
    assert "CONTACT_CUTOFF_ANGSTROM = 5.0" in module
    for token in ("5.0", "excluded_covalent_hops", "10_000_000", "model_inference_executed"):
        assert token in acceptance
    assert "distance is greater than" in protocol
    assert "three or disconnected" in protocol
    assert "not authorize GPU training" in protocol


def test_contact_cache_runner_never_reads_targets_or_runs_models() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".y" not in source
    assert "make_pcqm_gap_encoder" not in source
    assert "optimizer" not in source
    assert "backward" not in source
