"""Focused CPU checks for Modern2D accepted-input metadata preparation.

The fixture below is a generated accepted indexed ABI, not PCQM data.  It uses
the same packed ``Data``/slice layout as the existing indexed-input/export
tests, but stores one shared singleton topology view per role.  This keeps the
fixture bounded without constructing 150,000 Python graph objects or calling
the collator.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch_geometric.data import Data

from molgap.common.wire import canonical_fingerprint
from molgap.cross_scale_checkpoint import state_digest
from molgap.data.indexed_inputs import AcceptedIndexedInputPermit
from molgap.experiment_family_workflow import tensor_digest
from molgap.modern2d_input_preparation import (
    METADATA_FORMAT,
    REQUEST_FORMAT,
    ROLE_ROWS,
    prepare_qualification_inputs,
)
from molgap.pcqm_indexed_export import EXPORT_FORMAT, _chunk_paths
from molgap.pcqm_unified_roles import FORMAT
from molgap.research_memory.trace import file_digest, json_bytes


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "experiments/modern2d_model_integration/scripts/prepare_qualification_inputs.py"
MAX_CHUNK_BYTES = 64 * 1024 * 1024
VERSION_IDENTITY = hashlib.sha256(b"generated-modern2d-accepted-abi-v1").hexdigest()
PARENT_ACCEPTANCE = hashlib.sha256(b"generated-parent-acceptance-v1").hexdigest()


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def _read_json(path):
    return json.loads(path.read_bytes())


def _jsonable(value):
    return json.loads(json.dumps(value))


def _file_inventory(root):
    return {
        path.relative_to(root).as_posix(): {
            "sha256": file_digest(path),
            "bytes": path.stat().st_size,
        }
        for path in root.rglob("*")
        if path.is_file()
    }


def _compact_payload(path, *, role, rows, source_start):
    """Write one accepted packed shard without a Python graph list."""
    source_idx = torch.arange(source_start, source_start + rows, dtype=torch.int64)
    baseline = 0.0 if role == "train_100k" else 100.0
    target = (torch.arange(rows, dtype=torch.float32) / 1000.0) + baseline

    # A stride-zero expanded singleton is a valid packed tensor view.  It keeps
    # this generated acceptance fixture small while preserving the reviewed ABI.
    x = torch.zeros((1, 9), dtype=torch.int64).expand(rows, -1)
    random_walk_pe = torch.zeros((1, 16), dtype=torch.float32).expand(rows, -1)
    data = Data(
        x=x,
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        edge_attr=torch.empty((0, 3), dtype=torch.int64),
        random_walk_pe=random_walk_pe,
        source_idx=source_idx,
        y=target,
    )
    row_slices = torch.arange(rows + 1, dtype=torch.int64)
    empty_slices = torch.zeros(rows + 1, dtype=torch.int64)
    slices = {
        "x": row_slices,
        "edge_index": empty_slices.clone(),
        "edge_attr": empty_slices.clone(),
        "random_walk_pe": row_slices.clone(),
        "source_idx": row_slices.clone(),
        "y": row_slices.clone(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        torch.save((data, slices), stream)

    topology = {key: value for key, value in data.to_dict().items()
                if key not in {"source_idx", "y"}}
    topology_slices = {key: value for key, value in slices.items()
                       if key not in {"source_idx", "y"}}
    facts = {
        "rows": rows,
        "row_sha256": tensor_digest(source_idx, role="source_idx"),
        "target_sha256": tensor_digest(target, role="target"),
        "topology_sha256": state_digest({"data": topology, "slices": topology_slices}),
    }
    return {
        "source_idx": source_idx,
        "target": target,
        "facts": facts,
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
    }


def _census(rows):
    return {
        "rows": rows,
        "node_histogram": {"1": rows},
        "directed_edge_histogram": {"0": rows},
        "isolated_atom_count": rows,
        "zero_edge_graph_count": rows,
        "max_nodes": 1,
        "max_directed_edges": 0,
        "exceeding_caps_source_ids": {"ppgt": [], "esa": []},
    }


def _make_accepted_export(workspace):
    """Create accepted ABI bytes and the external pins consumed by the helper."""
    export_root = workspace / "accepted-export"
    export_root.mkdir()
    parent_manifest_sha = {
        role: _sha("parent-manifest-" + role)
        for role in ROLE_ROWS
    }
    parent_authorities = {}
    parent_histories = {}
    parent_permits = {}
    for role in ROLE_ROWS:
        authority = {
            "scope": "ACCEPTED_INDEXED_INPUT_V1",
            "role": role,
            "acceptance_sha256": PARENT_ACCEPTANCE,
            "manifest_sha256": parent_manifest_sha[role],
            "version_identity": VERSION_IDENTITY,
            "operations": ["prediction_input", "labels_read"],
            "protected": False,
            "max_chunk_bytes": MAX_CHUNK_BYTES,
        }
        history = {
            "role": role,
            "manifest_sha256": parent_manifest_sha[role],
            "usage": "training" if role.startswith("train_") else "selection-used",
        }
        parent_authorities[role] = authority
        parent_histories[role] = history
        parent_permits[role] = AcceptedIndexedInputPermit(
            role,
            PARENT_ACCEPTANCE,
            parent_manifest_sha[role],
            VERSION_IDENTITY,
            canonical_fingerprint(authority),
            canonical_fingerprint(history),
            MAX_CHUNK_BYTES,
        )

    # Keep Common-Dev disjoint from training, as the real role contract does.
    specs = {
        "train_100k": (100_000, 0),
        "common_dev": (50_000, 200_000),
    }
    role_details = {}
    exported_request = {
        "format": EXPORT_FORMAT,
        "private_required": True,
        "roles": list(ROLE_ROWS),
        "parent_acceptance_sha256": PARENT_ACCEPTANCE,
        "permits": {role: asdict(parent_permits[role]) for role in ROLE_ROWS},
        "max_chunk_bytes": MAX_CHUNK_BYTES,
        "max_rows": dict(ROLE_ROWS),
        "scope": "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",
        "status": "DATA_ROLE_ACCEPTED",
    }
    _write_json(export_root / "export_request.json", exported_request)

    for role, (rows, source_start) in specs.items():
        payload_relative, receipt_relative = _chunk_paths(role, 0)
        payload = _compact_payload(export_root / payload_relative, role=role,
                                   rows=rows, source_start=source_start)
        member_relative = f"members/{role}.i64le"
        member_path = export_root / member_relative
        member_path.parent.mkdir(parents=True, exist_ok=True)
        member_path.write_bytes(
            payload["source_idx"].numpy().astype("<i8", copy=False).tobytes()
        )
        member_sha = file_digest(member_path)
        chunk = {
            "file": payload_relative,
            "sha256": payload["sha256"],
            "bytes": payload["bytes"],
            "rows": rows,
            "offsets": None,
            "row_sha256": payload["facts"]["row_sha256"],
            "target_sha256": payload["facts"]["target_sha256"],
            "topology_sha256": payload["facts"]["topology_sha256"],
        }
        logical = canonical_fingerprint({
            "members": payload["facts"]["row_sha256"],
            "chunks": [{key: chunk[key] for key in (
                "row_sha256", "target_sha256", "topology_sha256")}]})
        manifest = {
            "format": FORMAT,
            "status": "complete",
            "version_identity": VERSION_IDENTITY,
            "role": role,
            "rows": rows,
            "source_index_sha256": payload["facts"]["row_sha256"],
            "ordering": "ascending-source-index",
            "source_index_file": member_relative,
            "source_index_file_sha256": member_sha,
            "feature_schema": "ogb-atom9-bond3-rwse16-v1",
            "pure_2d": True,
            "protected": False,
            "access": "TRAIN_OR_COMMON_DEVELOPMENT",
            "label_statistics_published": False,
            "logical_content_sha256": logical,
            "parent_acceptance_sha256": PARENT_ACCEPTANCE,
            "parent_manifest_sha256": parent_manifest_sha[role],
            "graph_chunks": [chunk],
        }
        manifest_path = export_root / f"roles/{role}.json"
        _write_json(manifest_path, manifest)
        manifest_sha = file_digest(manifest_path)
        census = _census(rows)
        receipt = {
            "format": EXPORT_FORMAT + "-chunk",
            "status": "CONTENT_VERIFIED",
            "request_identity": canonical_fingerprint(exported_request),
            "role": role,
            "parent_chunk_index": 0,
            "parent_manifest_sha256": parent_manifest_sha[role],
            "parent_chunk_identity": canonical_fingerprint(chunk),
            "file": payload_relative,
            "offsets": None,
            "rows": rows,
            "row_sha256": chunk["row_sha256"],
            "target_sha256": chunk["target_sha256"],
            "topology_sha256": chunk["topology_sha256"],
            "sha256": payload["sha256"],
            "bytes": payload["bytes"],
            "census": census,
        }
        _write_json(export_root / receipt_relative, receipt)
        role_details[role] = {
            "rows": rows,
            "source_idx": payload["source_idx"],
            "target": payload["target"],
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "logical_content_sha256": logical,
            "source_index_sha256": payload["facts"]["row_sha256"],
        }

    census = {
        "format": EXPORT_FORMAT + "-census",
        "label_free": True,
        "request_identity": canonical_fingerprint(exported_request),
        "caps": {"ppgt": {"nodes": 256, "directed_edges": 1024},
                 "esa": {"nodes": 1024, "directed_edges": 4096}},
        "roles": {role: _census(detail["rows"])
                  for role, detail in role_details.items()},
    }
    _write_json(export_root / "census.json", census)

    inventory = _file_inventory(export_root)
    acceptance = {
        "format": EXPORT_FORMAT + "-acceptance",
        "status": "DATA_ROLE_ACCEPTED",
        "scope": "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",
        "rule_identity": VERSION_IDENTITY,
        "parent_acceptance_sha256": PARENT_ACCEPTANCE,
        "private_required": True,
        "request_identity": canonical_fingerprint(exported_request),
        "roles": {
            role: {
                "path": f"roles/{role}.json",
                "sha256": detail["manifest_sha256"],
                "rows": detail["rows"],
                "source_index_sha256": detail["source_index_sha256"],
                "logical_content_sha256": detail["logical_content_sha256"],
            }
            for role, detail in role_details.items()
        },
        "inventory": inventory,
        "official_validation_read": False,
        "test_roles_read": False,
        "graphs_rebuilt": False,
        "audit_label_statistics_exposed": False,
        "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING",
        "publication_status": "NOT_PUBLISHED",
    }
    _write_json(export_root / "acceptance.json", acceptance)
    export_acceptance_sha = file_digest(export_root / "acceptance.json")

    roles = {}
    for role, detail in role_details.items():
        authority = {
            "scope": "ACCEPTED_INDEXED_INPUT_V1",
            "role": role,
            "acceptance_sha256": export_acceptance_sha,
            "manifest_sha256": detail["manifest_sha256"],
            "version_identity": VERSION_IDENTITY,
            "operations": ["prediction_input", "labels_read"],
            "protected": False,
            "max_chunk_bytes": MAX_CHUNK_BYTES,
        }
        history = {
            "role": role,
            "manifest_sha256": detail["manifest_sha256"],
            "usage": "training" if role.startswith("train_") else "selection-used",
        }
        permit = AcceptedIndexedInputPermit(
            role,
            export_acceptance_sha,
            detail["manifest_sha256"],
            VERSION_IDENTITY,
            canonical_fingerprint(authority),
            canonical_fingerprint(history),
            MAX_CHUNK_BYTES,
        )
        roles[role] = {
            "permit": asdict(permit),
            "authority": authority,
            "role_history": history,
            "rows": detail["rows"],
            "source_index_sha256": detail["source_index_sha256"],
            "logical_content_sha256": detail["logical_content_sha256"],
        }

    workspace = workspace.absolute()
    request = {
        "format": REQUEST_FORMAT,
        "export_root": str(export_root.absolute()),
        "export_acceptance_sha256": export_acceptance_sha,
        "parent_acceptance_sha256": PARENT_ACCEPTANCE,
        "output_root": str((workspace / "metadata-output").absolute()),
        "boundary": str(workspace),
        "roles": roles,
        "inventory": acceptance["inventory"],
    }
    return {
        "workspace": workspace,
        "export_root": export_root,
        "request": request,
        "roles": role_details,
        "export_files": _file_inventory(export_root),
    }


@pytest.fixture(scope="module")
def accepted_fixture(tmp_path_factory):
    return _make_accepted_export(tmp_path_factory.mktemp("modern2d-input"))


def _request_for(ctx, name, *, export_root=None):
    request = deepcopy(ctx["request"])
    request["output_root"] = str((ctx["workspace"] / name).absolute())
    if export_root is not None:
        request["export_root"] = str(Path(export_root).absolute())
    return request


def _run_cli(request, *, execute):
    request_path = Path(request["boundary"]) / (
        "input_metadata_request_execute.json" if execute
        else "input_metadata_request_dryrun.json"
    )
    request_path.write_bytes(json_bytes(request))
    command = [sys.executable, str(CLI), "--request", str(request_path)]
    if execute:
        command.append("--execute")
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True)


def test_cli_defaults_to_dry_run_without_opening_payload(accepted_fixture):
    request = _request_for(
        accepted_fixture,
        "cli-dryrun-output",
        export_root=accepted_fixture["workspace"] / "not-opened-export",
    )
    result = _run_cli(request, execute=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == {
        "output_root": request["output_root"],
        "payloads_read": False,
        "remote_submission": False,
        "roles": ROLE_ROWS,
        "status": "PREPARE_ONLY_NOT_EXECUTED",
    }
    assert not Path(request["output_root"]).exists()
    assert not Path(request["export_root"]).exists()


def test_cli_execute_covers_the_focused_positive_path(accepted_fixture):
    request = _request_for(accepted_fixture, "cli-execute-output")
    result = _run_cli(request, execute=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "INPUT_METADATA_PREPARED"
    assert report["remote_submission"] is False
    assert report["scientific_training_authorized"] is False
    assert {path.name for path in Path(request["output_root"]).iterdir()} == {
        "training_input.json", "qualification_inputs.json"
    }


def test_positive_helper_uses_accepted_abi_and_metadata_only(accepted_fixture, monkeypatch):
    import molgap.esa_native as esa_native
    import molgap.ppgt_author as ppgt_author
    from molgap.data.indexed_inputs import IndexedRoleDataset
    from torch_geometric.data import InMemoryDataset

    def forbidden(*args, **kwargs):
        raise AssertionError("model construction or graph collation is out of scope")

    monkeypatch.setattr(ppgt_author, "PPGTAuthor", forbidden)
    monkeypatch.setattr(esa_native, "ESANative", forbidden)
    monkeypatch.setattr(InMemoryDataset, "collate", forbidden)

    selected_target_roles = []
    original_selected_targets = IndexedRoleDataset.selected_targets

    def selected_targets(dataset):
        selected_target_roles.append(dataset.permit.role)
        yield from original_selected_targets(dataset)

    monkeypatch.setattr(IndexedRoleDataset, "selected_targets", selected_targets)
    request = _request_for(accepted_fixture, "direct-positive-output")
    before = dict(accepted_fixture["export_files"])
    result = prepare_qualification_inputs(request, execute=True)
    assert result["status"] == "INPUT_METADATA_PREPARED"
    assert result["remote_submission"] is False
    assert result["scientific_training_authorized"] is False
    assert selected_target_roles == ["train_100k"]
    assert before == _file_inventory(accepted_fixture["export_root"])

    output = Path(result["output_root"])
    metadata = _read_json(output / "qualification_inputs.json")
    training_input = _read_json(output / "training_input.json")
    train = accepted_fixture["roles"]["train_100k"]
    development = accepted_fixture["roles"]["common_dev"]
    train_targets = train["target"].double()
    all_targets = torch.cat((train_targets, development["target"].double()))
    assert metadata["target_mean_eV"] == pytest.approx(float(train_targets.mean()))
    assert metadata["target_std_eV"] == pytest.approx(float(train_targets.std()))
    assert metadata["target_mean_eV"] != pytest.approx(float(all_targets.mean()))
    assert metadata["development_label_statistics_exposed"] is False
    assert set(metadata["roles"]["common_dev"]) == {
        "rows", "source_idx_tensor_sha256"
    }
    assert metadata["roles"]["train_100k"]["source_idx_tensor_sha256"] == tensor_digest(
        train["source_idx"], role="source_idx"
    )
    assert metadata["roles"]["common_dev"]["source_idx_tensor_sha256"] == tensor_digest(
        development["source_idx"], role="source_idx"
    )

    assert training_input["format"] == "molgap-indexed-training-input-v1"
    assert training_input["data_root"] == "indexed_assets"
    assert metadata["data_root"] == "indexed_assets"
    assert not (output / "indexed_assets").exists()
    assert training_input["train"] == {
        key: request["roles"]["train_100k"][key]
        for key in ("permit", "authority", "role_history")
    }
    assert training_input["development"] == {
        key: request["roles"]["common_dev"][key]
        for key in ("permit", "authority", "role_history")
    }

    from molgap.esa_native import ESANativeConfig
    from molgap.ppgt_author import PPGTAuthorConfig

    expected_configs = {
        "ppgt_author_v2": _jsonable(asdict(PPGTAuthorConfig())),
        "esa_native_v3": _jsonable(asdict(ESANativeConfig())),
    }
    from molgap.experiment_execution import MODERN2D_PREVIEW_INTERFACES

    expected_schemas = {
        "ppgt_author_v2": MODERN2D_PREVIEW_INTERFACES[("ppgt_small", "2")][1],
        "esa_native_v3": MODERN2D_PREVIEW_INTERFACES[("esa_small", "3")][1],
    }
    assert set(metadata["models"]) == set(expected_configs)
    for name, model in metadata["models"].items():
        assert model["model_config"] == expected_configs[name]
        assert model["feature_binding_sha256"] == {
            role: canonical_fingerprint(binding)
            for role, binding in model["feature_bindings"].items()
        }
        assert set(model["feature_bindings"]) == set(ROLE_ROWS)
        bindings = model["feature_bindings"]
        assert {binding["transform_code_sha"] for binding in bindings.values()}.copy().__len__() == 1
        assert {binding["transform_config_sha"] for binding in bindings.values()}.copy().__len__() == 1
        for role, binding in bindings.items():
            assert binding["role"] == role
            assert binding["scope"] == "ACCEPTED_INDEXED_INPUT_V1"
            assert binding["feature_schema_version"] == expected_schemas[name]
            assert binding["derived_feature_sha"] is None
            assert binding["base_dataset_manifest_sha"] == request["roles"][role]["permit"]["manifest_sha256"]
            assert binding["source_row_sha"] == accepted_fixture["roles"][role]["manifest"]["source_index_sha256"]
    assert set(metadata) >= {
        "format", "status", "training_input", "export_acceptance_sha256",
        "parent_acceptance_sha256", "inventory_sha256", "target_mean_eV",
        "target_std_eV", "normalization", "roles", "models",
        "development_label_statistics_exposed", "scientific_training_authorized",
        "publication_status",
    }
    assert metadata["format"] == METADATA_FORMAT
    assert metadata["publication_status"] == "NOT_PUBLISHED"


def test_absolute_boundary_and_disjoint_output_guards(accepted_fixture, tmp_path):
    base = _request_for(accepted_fixture, "boundary-base-output")
    bad_requests = []

    relative = deepcopy(base)
    relative["export_root"] = "relative/export"
    bad_requests.append(relative)

    outside_export = deepcopy(base)
    outside_export["export_root"] = str(Path(base["boundary"]).parent / "outside-export")
    bad_requests.append(outside_export)

    outside_output = deepcopy(base)
    outside_output["output_root"] = str(Path(base["boundary"]).parent / "outside-output")
    bad_requests.append(outside_output)

    same = deepcopy(base)
    same["output_root"] = same["export_root"]
    bad_requests.append(same)

    nested_output = deepcopy(base)
    nested_output["output_root"] = str(Path(base["export_root"]) / "metadata")
    bad_requests.append(nested_output)

    nested_export = deepcopy(base)
    nested_export["export_root"] = str(Path(base["output_root"]) / "export")
    bad_requests.append(nested_export)

    existing_output = deepcopy(base)
    Path(existing_output["output_root"]).mkdir()
    bad_requests.append(existing_output)

    for request in bad_requests:
        with pytest.raises(ValueError):
            prepare_qualification_inputs(request, execute=False)


def test_parent_acceptance_inventory_and_manifest_pins_are_external(accepted_fixture):
    mutations = []

    bad_acceptance = _request_for(accepted_fixture, "bad-acceptance-output")
    bad_acceptance["export_acceptance_sha256"] = "0" * 64
    mutations.append(bad_acceptance)

    bad_parent = _request_for(accepted_fixture, "bad-parent-output")
    bad_parent["parent_acceptance_sha256"] = "1" * 64
    mutations.append(bad_parent)

    bad_inventory = _request_for(accepted_fixture, "bad-inventory-output")
    bad_inventory["inventory"] = dict(bad_inventory["inventory"])
    bad_inventory["inventory"]["census.json"] = {
        "sha256": "2" * 64,
        "bytes": 1,
    }
    mutations.append(bad_inventory)

    bad_manifest = _request_for(accepted_fixture, "bad-manifest-output")
    bad_manifest["roles"]["train_100k"]["permit"]["manifest_sha256"] = "3" * 64
    mutations.append(bad_manifest)

    for request in mutations:
        with pytest.raises(ValueError):
            prepare_qualification_inputs(request, execute=True)


def test_cpu_fixture_and_promoted_toy_counts_are_rejected_before_payload_open(accepted_fixture):
    synthetic = _request_for(
        accepted_fixture,
        "synthetic-output",
        export_root=accepted_fixture["workspace"] / "missing-synthetic-export",
    )
    synthetic["roles"]["train_100k"]["permit"]["scope"] = "CPU_SYNTHETIC_ONLY"
    synthetic["roles"]["train_100k"]["authority"]["scope"] = "CPU_SYNTHETIC_ONLY"

    toy = _request_for(
        accepted_fixture,
        "toy-output",
        export_root=accepted_fixture["workspace"] / "missing-toy-export",
    )
    toy["roles"]["train_100k"]["rows"] = 3

    for request in (synthetic, toy):
        with pytest.raises(ValueError):
            prepare_qualification_inputs(request, execute=False)


@pytest.mark.parametrize("extra_kind", ["audit", "five_hundred_k", "top_level", "role"])
def test_audit_500k_and_unallowlisted_extras_are_rejected(accepted_fixture, extra_kind):
    request = _request_for(
        accepted_fixture,
        f"extra-{extra_kind}-output",
        export_root=accepted_fixture["workspace"] / f"missing-{extra_kind}-export",
    )
    if extra_kind == "audit":
        request["roles"]["independent_audit"] = {}
    elif extra_kind == "five_hundred_k":
        request["roles"]["train_500k"] = {}
    elif extra_kind == "top_level":
        request["unexpected"] = True
    else:
        request["roles"]["train_100k"]["unexpected"] = True
    with pytest.raises(ValueError):
        prepare_qualification_inputs(request, execute=False)


def test_physical_tamper_is_rejected_and_cost_metadata_is_irrelevant(accepted_fixture):
    request = _request_for(accepted_fixture, "tamper-output")
    payload = accepted_fixture["export_root"] / "payloads/full_minus_holdout/train_100k_000000.pt"
    original = payload.read_bytes()
    try:
        payload.write_bytes(original + b"tampered")
        with pytest.raises(ValueError, match="bytes changed"):
            prepare_qualification_inputs(request, execute=True)
    finally:
        payload.write_bytes(original)

    # Cost is a later scheduler/accounting concern; it is intentionally absent
    # from this metadata-only request and must not block dry-run validation.
    assert "cost" not in request
    report = prepare_qualification_inputs(
        _request_for(accepted_fixture, "no-cost-dryrun"), execute=False
    )
    assert report["status"] == "PREPARE_ONLY_NOT_EXECUTED"


def test_cli_request_path_must_be_inside_its_declared_boundary(accepted_fixture):
    request = _request_for(accepted_fixture, "outside-request-output")
    outside_request = accepted_fixture["workspace"].parent / "input-metadata-outside.json"
    outside_request.write_bytes(json_bytes(request))
    command = [sys.executable, str(CLI), "--request", str(outside_request)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "outside" in (result.stderr + result.stdout).lower()

