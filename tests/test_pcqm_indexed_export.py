"""Private export checks over generated indexed fixtures; no real acceptance."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from molgap.common.wire import canonical_fingerprint
from molgap.data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit, IndexedRoleDataset
from molgap.data.packed_graph import PackedGraphDataset
from molgap.pcqm_indexed_export import export_indexed_roles, verify_indexed_export
from molgap.research_memory.trace import file_digest, json_bytes
from test_pcqm_indexed_inputs import fixture


def _read(path):
    return json.loads(path.read_bytes())


def _bind(root, role):
    acceptance = _read(root / "acceptance.json")
    manifest_sha = file_digest(root / f"roles/{role}.json")
    authority = {"scope": "CPU_SYNTHETIC_ONLY", "role": role, "manifest_sha256": manifest_sha,
        "operations": ["prediction_input", "labels_read"], "protected": False}
    history = {"role": role, "manifest_sha256": manifest_sha,
        "usage": "training" if role.startswith("train_") else "selection-used"}
    permit = IndexedInputPermit(role, file_digest(root / "acceptance.json"), manifest_sha,
        acceptance["rule_identity"], canonical_fingerprint(authority), canonical_fingerprint(history))
    return permit, authority, history


def _inputs(tmp_path, train="train_100k"):
    root, _, _, _ = fixture(tmp_path, train)
    roles = [train, "common_dev"]
    bindings = {r: _bind(root, r) for r in roles}
    return dict(parent_root=root, output_root=tmp_path / "export", roles=roles,
        permits={r: b[0] for r, b in bindings.items()}, authorities={r: b[1] for r, b in bindings.items()},
        role_histories={r: b[2] for r, b in bindings.items()},
        parent_acceptance_sha256=file_digest(root / "acceptance.json"), max_chunk_bytes=1024**2,
        max_rows={r: 512 for r in roles})


def _files(root):
    return {p.relative_to(root).as_posix(): file_digest(p) for p in root.rglob("*") if p.is_file()}


def _verify(args):
    return verify_indexed_export(export_acceptance_sha256=file_digest(args["output_root"] / "acceptance.json"), **args)


@pytest.mark.parametrize("train", ["train_100k", "train_500k"])
def test_exact_private_inventory_logical_identity_and_loader_abi(tmp_path, train):
    args = _inputs(tmp_path, train)
    parent, output = args["parent_root"], args["output_root"]
    before = _files(parent)
    accepted = export_indexed_roles(**args)
    assert accepted["scope"] == "CPU_SYNTHETIC_ONLY"
    assert accepted["status"] == "SYNTHETIC_DATA_ROLES_VERIFIED"
    assert accepted["publication_status"] == "NOT_PUBLISHED"
    assert set(accepted["roles"]) == set(args["roles"])
    assert set(_files(output)) == set(accepted["inventory"]) | {"acceptance.json"}
    assert _verify(args) == accepted
    for role in args["roles"]:
        m = _read(output / f"roles/{role}.json")
        original = _read(parent / f"roles/{role}.json")
        assert m["logical_content_sha256"] == original["logical_content_sha256"]
        assert m["parent_manifest_sha256"] == args["permits"][role].manifest_sha256
        assert all(c["offsets"] is None for c in m["graph_chunks"])
        assert [(c["row_sha256"], c["target_sha256"], c["topology_sha256"]) for c in m["graph_chunks"]] == [
            (c["row_sha256"], c["target_sha256"], c["topology_sha256"]) for c in original["graph_chunks"]]
        p, a, h = _bind(output, role)
        selected = IndexedRoleDataset(output, p, authority=a, role_history=h)
        source = IndexedRoleDataset(parent, args["permits"][role],
            authority=args["authorities"][role], role_history=args["role_histories"][role])
        assert selected.source_indices == source.source_indices
        assert len(selected) == m["rows"]
        for i in range(len(selected)):
            for key in ("x", "edge_index", "edge_attr", "random_walk_pe", "source_idx", "y"):
                assert torch.equal(selected[i][key], source[i][key])
        for c in m["graph_chunks"]:
            packed = PackedGraphDataset(output / c["file"])
            assert len(packed) == c["rows"]
            assert "role_name" not in packed._data.keys()
            # No backing source chunk, unselected rows or labels were serialized.
            assert packed._data.source_idx.untyped_storage().nbytes() == packed._data.source_idx.numel() * 8
            assert packed._data.y.untyped_storage().nbytes() == packed._data.y.numel() * packed._data.y.element_size()
    metadata = b"\n".join(p.read_bytes() for p in output.rglob("*.json"))
    assert b"independent_audit" not in metadata
    assert b"private_identity" not in metadata
    assert not any("train_1m" in p or "independent_audit" in p or p.endswith("sqlite") for p in _files(output))
    assert before == _files(parent)


def test_completed_export_is_read_only_and_missing_files_are_not_repaired(tmp_path):
    args = _inputs(tmp_path)
    first = export_indexed_roles(**args)
    output = args["output_root"]
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    assert export_indexed_roles(**args) == first
    assert before == {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    member = output / "members/common_dev.i64le"
    member.unlink()
    with pytest.raises(ValueError, match="inventory"):
        export_indexed_roles(**args)
    assert not member.exists()


@pytest.mark.parametrize("when", ["before_payload", "after_payload", "after_first_role"])
def test_receipt_first_interruption_resumes_stable_bytes_and_census(tmp_path, monkeypatch, when):
    import molgap.pcqm_indexed_export as module
    args = _inputs(tmp_path)
    original_copy, original_publish = module.copy_immutable_file, module._publish
    interrupted = False

    def copy(source, dest, **kwargs):
        nonlocal interrupted
        if dest.suffix == ".pt" and not interrupted and when != "after_first_role":
            interrupted = True
            if when == "after_payload":
                original_copy(source, dest, **kwargs)
            raise OSError("simulated interruption")
        return original_copy(source, dest, **kwargs)

    def publish(root, relative, value):
        nonlocal interrupted
        result = original_publish(root, relative, value)
        if relative == "roles/train_100k.json" and not interrupted and when == "after_first_role":
            interrupted = True
            raise OSError("simulated interruption")
        return result

    monkeypatch.setattr(module, "copy_immutable_file", copy)
    monkeypatch.setattr(module, "_publish", publish)
    with pytest.raises(OSError, match="interruption"):
        export_indexed_roles(**args)
    output = args["output_root"]
    assert not (output / "acceptance.json").exists()
    retained = _files(output)
    assert any(name.startswith("receipts/") for name in retained)
    monkeypatch.setattr(module, "copy_immutable_file", original_copy)
    monkeypatch.setattr(module, "_publish", original_publish)
    export_indexed_roles(**args)
    assert all(file_digest(output / name) == sha for name, sha in retained.items())
    assert _verify(args)["scope"] == "CPU_SYNTHETIC_ONLY"
    census = _read(output / "census.json")
    for role in args["roles"]:
        assert census["roles"][role]["rows"] == _read(output / f"roles/{role}.json")["rows"]


@pytest.mark.parametrize("kind", ["payload", "members", "manifest", "receipt", "census", "extra_payload", "extra_directory"])
def test_tamper_or_nonallowlisted_output_aborts_without_overwrite(tmp_path, kind):
    args = _inputs(tmp_path)
    export_indexed_roles(**args)
    output = args["output_root"]
    m = _read(output / "roles/train_100k.json")
    paths = {"payload": output / m["graph_chunks"][0]["file"], "members": output / "members/train_100k.i64le",
        "manifest": output / "roles/train_100k.json", "receipt": next((output / "receipts").glob("*.json")),
        "census": output / "census.json", "extra_payload": output / "unselected.pt"}
    if kind == "extra_directory":
        (output / "unselected").mkdir()
    else:
        paths[kind].write_bytes(b"tampered")
    before = _files(output)
    with pytest.raises((ValueError, KeyError)):
        export_indexed_roles(**args)
    assert before == _files(output)


@pytest.mark.parametrize("field", ["y", "x", "source_idx"])
def test_repinned_output_cannot_change_parent_decoded_content(tmp_path, field):
    args = _inputs(tmp_path)
    export_indexed_roles(**args)
    output = args["output_root"]
    manifest_path = output / "roles/train_100k.json"
    m = _read(manifest_path)
    c = m["graph_chunks"][0]
    packed = PackedGraphDataset(output / c["file"])
    graphs = [packed.get(i).clone() for i in range(len(packed))]
    graphs[0][field].reshape(-1)[0] += 1
    torch.save(InMemoryDataset.collate(graphs), output / c["file"])
    c["sha256"] = file_digest(output / c["file"])
    c["bytes"] = (output / c["file"]).stat().st_size
    manifest_path.write_bytes(json_bytes(m))
    receipt_path = output / "receipts/train_100k_000000.json"
    receipt = _read(receipt_path)
    receipt.update(sha256=c["sha256"], bytes=c["bytes"])
    receipt_path.write_bytes(json_bytes(receipt))
    acceptance_path = output / "acceptance.json"
    a = _read(acceptance_path)
    a["roles"]["train_100k"]["sha256"] = file_digest(manifest_path)
    for name, entry in a["inventory"].items():
        entry.update(sha256=file_digest(output / name), bytes=(output / name).stat().st_size)
    acceptance_path.write_bytes(json_bytes(a))
    with pytest.raises(ValueError, match="parent row/target/topology"):
        _verify(args)


def test_external_parent_and_export_acceptance_pins_are_required(tmp_path):
    args = _inputs(tmp_path)
    with pytest.raises(TypeError):
        export_indexed_roles(**{k: v for k, v in args.items() if k != "parent_acceptance_sha256"})
    export_indexed_roles(**args)
    accepted_sha = file_digest(args["output_root"] / "acceptance.json")
    with pytest.raises(ValueError, match="acceptance bytes"):
        verify_indexed_export(export_acceptance_sha256="a" * 64, **args)
    assert verify_indexed_export(export_acceptance_sha256=accepted_sha, **args)
    with pytest.raises(ValueError, match="acceptance/role pin"):
        export_indexed_roles(**{**args, "parent_acceptance_sha256": "b" * 64})


@pytest.mark.parametrize("roles", [["common_dev"], ["train_100k", "train_500k", "common_dev"],
    ["train_1m", "common_dev"], ["full_minus_holdout", "common_dev"], ["train_100k", "independent_audit"]])
def test_nonexact_role_request_denied_before_io(tmp_path, monkeypatch, roles):
    args = _inputs(tmp_path)
    def forbidden(*a, **k):
        raise AssertionError("Filesystem access before role gate")
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(ValueError, match="ROLE_BLOCKED"):
        export_indexed_roles(**{**args, "roles": roles})


def test_export_does_not_open_unselected_members_audit_identity_or_source(tmp_path, monkeypatch):
    args = _inputs(tmp_path)
    parent = args["parent_root"]
    def check(path):
        if parent in path.parents:
            relative = path.relative_to(parent).as_posix()
            if ("independent_audit" in relative or "private_identity" in relative or
                    "train_1m" in relative or relative == "roles/full_minus_holdout.json" or
                    relative == "members/full_minus_holdout.i64le" or relative == "roles/train_500k.json"):
                raise AssertionError("Opened unselected parent asset")
        if args["parent_root"].parent / "fixed" in path.parents:
            raise AssertionError("Opened original source shard")
    def guard(original):
        def guarded(path, *a, **k):
            check(path)
            return original(path, *a, **k)
        return guarded
    for name in ("open", "stat", "lstat"):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    export_indexed_roles(**args)


def test_runtime_role_annotation_is_bypassed(tmp_path, monkeypatch):
    args = _inputs(tmp_path)
    def forbidden(*a, **k):
        raise AssertionError("Annotated __getitem__ must not be exported")
    monkeypatch.setattr(IndexedRoleDataset, "__getitem__", forbidden)
    export_indexed_roles(**args)
    assert _verify(args)


@pytest.mark.parametrize("change", ["bytes", "rows", "scope", "authority", "history"])
def test_budget_authority_and_scope_fail_before_output(tmp_path, change):
    args = _inputs(tmp_path)
    if change == "bytes":
        args["max_chunk_bytes"] = 1
    elif change == "rows":
        args["max_rows"]["train_100k"] = 1
    elif change == "scope":
        args["permits"]["train_100k"] = replace(args["permits"]["train_100k"], scope="CPU_SYNTHETIC_ONLY")
        args["permits"]["train_100k"] = AcceptedIndexedInputPermit("train_100k",
            *[getattr(args["permits"]["train_100k"], k) for k in (
                "acceptance_sha256", "manifest_sha256", "version_identity", "authority_sha256", "role_history_sha256")],
            max_chunk_bytes=1024**2)
    elif change == "authority":
        args["authorities"]["train_100k"]["protected"] = True
    else:
        args["role_histories"]["common_dev"]["usage"] = "unseen"
    with pytest.raises((ValueError, TypeError)):
        export_indexed_roles(**args)
    assert not args["output_root"].exists()


def test_scratch_uses_export_parent_even_when_default_tmp_is_forbidden(tmp_path, monkeypatch):
    import molgap.pcqm_indexed_export as module
    args = _inputs(tmp_path)
    original = module.tempfile.TemporaryDirectory
    seen = []
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path / "forbidden_os_temp"))
    def controlled(*a, **k):
        assert Path(k["dir"]).resolve() == args["output_root"].parent.resolve()
        seen.append(k["dir"])
        return original(*a, **k)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", controlled)
    export_indexed_roles(**args)
    assert seen
    assert not (tmp_path / "forbidden_os_temp").exists()
    assert not list(tmp_path.glob("molgap_selected_export_*"))


def test_label_free_census_matches_selected_graphs_only(tmp_path):
    args = _inputs(tmp_path)
    export_indexed_roles(**args)
    census = _read(args["output_root"] / "census.json")
    assert census["label_free"] is True
    assert census["caps"] == {"ppgt": {"nodes": 256, "directed_edges": 1024},
        "esa": {"nodes": 1024, "directed_edges": 4096}}
    for role in args["roles"]:
        source = IndexedRoleDataset(args["parent_root"], args["permits"][role],
            authority=args["authorities"][role], role_history=args["role_histories"][role])
        sizes = [source[i].num_nodes for i in range(len(source))]
        c = census["roles"][role]
        assert c["node_histogram"] == {str(n): sizes.count(n) for n in sizes}
        assert c["directed_edge_histogram"] == {"0": len(sizes)}
        assert c["isolated_atom_count"] == sum(sizes)
        assert c["zero_edge_graph_count"] == len(sizes)
        assert c["max_nodes"] == max(sizes) and c["max_directed_edges"] == 0
        assert c["exceeding_caps_source_ids"] == {"ppgt": [], "esa": []}
        assert set(c) == {"rows", "node_histogram", "directed_edge_histogram", "isolated_atom_count",
            "zero_edge_graph_count", "max_nodes", "max_directed_edges", "exceeding_caps_source_ids"}


def test_selected_topology_mismatch_retains_failure_evidence(tmp_path, monkeypatch):
    import molgap.pcqm_indexed_export as module
    args = _inputs(tmp_path)
    original = module._facts
    def changed(*a, **k):
        facts = original(*a, **k)
        facts["topology_sha256"] = "0" * 64
        return facts
    monkeypatch.setattr(module, "_facts", changed)
    with pytest.raises(ValueError, match="parent row/target/topology"):
        export_indexed_roles(**args)
    output = args["output_root"]
    assert not (output / "acceptance.json").exists()
    failure = _read(output / "failure.json")
    assert failure["observed"]["topology_sha256"] == "0" * 64
    assert failure["expected"]["topology_sha256"] != "0" * 64
    assert not list(output.rglob("*.pt"))


def test_census_caps_are_strict_and_isolates_use_both_directed_endpoints():
    from molgap.pcqm_indexed_export import _empty_census, _observe, _merge_census
    census = _empty_census()
    for source, nodes, edges in [(10, 256, 1024), (11, 257, 1024), (12, 1024, 4096),
                                  (13, 1025, 4096), (14, 2, 4097), (15, 3, 0)]:
        edge_index = torch.stack([torch.zeros(edges, dtype=torch.long), torch.ones(edges, dtype=torch.long)])
        graph = Data(x=torch.zeros((nodes, 9), dtype=torch.long), edge_index=edge_index,
            source_idx=torch.tensor([source]))
        _observe(census, graph)
    assert census["exceeding_caps_source_ids"] == {"ppgt": [11, 12, 13, 14], "esa": [13, 14]}
    assert census["isolated_atom_count"] == (254 + 255 + 1022 + 1023 + 0 + 3)
    assert census["zero_edge_graph_count"] == 1
    assert census["max_nodes"] == 1025 and census["max_directed_edges"] == 4097
    assert _merge_census([census, _empty_census()]) == census


def test_real_permits_cannot_promote_synthetic_acceptance(tmp_path):
    args = _inputs(tmp_path)
    for role in args["roles"]:
        p = args["permits"][role]
        authority = {"scope": "ACCEPTED_INDEXED_INPUT_V1", "role": role,
            "acceptance_sha256": p.acceptance_sha256, "manifest_sha256": p.manifest_sha256,
            "version_identity": p.version_identity, "operations": ["prediction_input", "labels_read"],
            "protected": False, "max_chunk_bytes": 1024**2}
        args["authorities"][role] = authority
        args["permits"][role] = AcceptedIndexedInputPermit(role, p.acceptance_sha256,
            p.manifest_sha256, p.version_identity, canonical_fingerprint(authority), p.role_history_sha256,
            max_chunk_bytes=1024**2)
    with pytest.raises(ValueError, match="Parent acceptance scope"):
        export_indexed_roles(**args)
    assert not args["output_root"].exists()


def test_changed_request_cannot_expand_completed_export(tmp_path):
    args = _inputs(tmp_path)
    export_indexed_roles(**args)
    before = _files(args["output_root"])
    with pytest.raises(ValueError, match="request changed"):
        export_indexed_roles(**{**args, "max_chunk_bytes": 512 * 1024})
    assert before == _files(args["output_root"])
