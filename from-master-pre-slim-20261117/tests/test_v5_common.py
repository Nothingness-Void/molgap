from __future__ import annotations

import importlib
import sys

import pytest

from molgap.v5_common import (
    OUTCOME_FIELDS,
    V5_CONTRACT_ID,
    V5_EVIDENCE_FORMAT,
    validate_v5_evidence_envelope,
)


def _envelope(pointer: str) -> dict:
    return {
        "format": V5_EVIDENCE_FORMAT,
        "contract": V5_CONTRACT_ID,
        "evidence_id": "evidence-common-test",
        "track": "C",
        "scope": "unit-test",
        "legacy_contract": "fixture-v1",
        "outcome": {field: "not_evaluated" for field in OUTCOME_FIELDS},
        "authority": {"pointers": [pointer]},
        "role_use": {
            "official_validation": "untouched",
            "test_dev": "untouched",
            "test_challenge": "untouched",
        },
        "artifacts": [
            {
                "name": "fixture",
                "locator": "external://fixture",
                "availability": "durable_remote_verified",
                "sha256": "a" * 64,
            }
        ],
        "migration": {
            "migrated_at": "2026-09-19",
            "verification_scope": "schema-only",
            "training_executed": False,
            "inference_executed": False,
            "scientific_reinterpretation": False,
        },
    }


def test_shared_validator_is_desktop_independent(tmp_path, monkeypatch):
    pointer = tmp_path / "authority.json"
    pointer.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "molgap.v5_desktop", None)
    module = importlib.reload(importlib.import_module("molgap.research_memory.validate"))
    checked = module.validate_v5_evidence_envelope(
        _envelope(pointer.name), repo_root=tmp_path
    )
    assert checked["valid"] is True


def test_shared_validator_rejects_escape_and_overloaded_acceptance(tmp_path):
    escaped = _envelope("../outside.json")
    with pytest.raises(ValueError, match="escapes repository"):
        validate_v5_evidence_envelope(escaped, repo_root=tmp_path)

    overloaded = _envelope("authority.json")
    overloaded["accepted"] = True
    with pytest.raises(ValueError, match="separate outcome dimensions"):
        validate_v5_evidence_envelope(overloaded)
