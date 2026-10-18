"""Modern2D input-role manifests must stay separate from observed RML roles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molgap.research_memory.discovery import discover_records
from molgap.research_memory.validate import validate_repository_records


def _write_role(root: Path, name: str, record: dict) -> Path:
    path = root / "experiments" / "modern2d" / "roles" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _known_input_manifest(role: str) -> dict:
    return {
        "format": "molgap-pcqm4mv2-unified-roles-v2",
        "role": role,
        "status": "complete",
        "protected": False,
        "ordering": "ascending-source-index",
        "rows": 100000 if role.endswith("100k") else 50000,
        "graph_chunks": [],
        "source_index_sha256": "a" * 64,
    }


def test_known_input_manifests_are_not_observed_role_events(tmp_path, monkeypatch):
    paths = tuple(
        _write_role(tmp_path, f"{role}.json", _known_input_manifest(role))
        for role in ("train_100k", "train_500k", "common_dev")
    )

    import molgap.research_memory.validate as validator

    def forbidden(*args, **kwargs):
        raise AssertionError("input-role metadata reached the observed RML validator")

    monkeypatch.setattr(validator, "validate_role_event", forbidden)
    result = validate_repository_records(tmp_path)

    assert result["discovered"].roles == ()
    assert result["records"]["roles"] == []
    assert all(path.is_file() for path in paths)


def test_unknown_role_event_schema_fails_closed(tmp_path):
    path = _write_role(tmp_path, "unknown.json", {
        "schema": "molgap-role-event-v999",
        "role_event_id": "role-unknown-schema",
    })

    with pytest.raises(ValueError, match="unsupported role-event schema"):
        validate_repository_records(tmp_path)
    assert path.is_file()


def test_input_manifest_with_event_fields_is_ambiguous_and_rejected(tmp_path):
    record = _known_input_manifest("train_100k")
    record.update(schema="molgap-role-event-v1", role_event_id="role-forged")

    _write_role(tmp_path, "ambiguous.json", record)
    with pytest.raises(ValueError, match="Ambiguous or malformed retained input-role manifest"):
        discover_records(tmp_path)


def test_original_rml_role_event_remains_discoverable(tmp_path):
    event = _write_role(tmp_path, "original.json", {
        "schema": "molgap-role-event-v1",
        "role_event_id": "role-fixture-original-training-membership",
        "trajectory_id": "TC-fixture-original",
        "action_id": "A001",
        "run_id": "fixture:original",
        "dataset_identity": "pcqm4mv2-ogb-fixed-100k-v1",
        "row_manifest_hash": "b" * 64,
        "role_name": "internal_train",
        "evidence_ref": "experiments/fixture/observed.json",
        "access_kind": "training_membership",
        "selection_used": False,
    })

    result = discover_records(tmp_path)

    assert result.roles == (event.resolve(),)
