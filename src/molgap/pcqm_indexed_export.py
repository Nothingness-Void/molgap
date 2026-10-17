"""Private, selected-role V2 tensor export; no publication or training authority.

Trust pins, read authority and role history come from the caller. The physical
training directory retains the indexed loader ABI; its files contain only the
requested training role, never an original Full shard. Chunk grouping and the
parent logical identity are preserved. A receipt can precede its payload during
an interruption; only acceptance.json commits the complete, inventoried export.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import tempfile

from .common.io import _safe_local, copy_immutable_file, publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .cross_scale_checkpoint import state_digest
from .cross_scale_inputs import index_digest, target_digest
from .cross_scale_plan import digest
from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit, IndexedRoleDataset, _local
from .data.packed_graph import PackedGraphDataset
from .pcqm_unified_roles import FORMAT, GRAPH_FIELDS
from .research_memory.trace import file_digest, json_bytes, sync_directory


EXPORT_FORMAT = "molgap-pcqm-indexed-private-export-v1"
ROLE_LIMITS = {"train_100k": 100000, "train_500k": 500000, "common_dev": 50000}
LOGICAL_FIELDS = ("row_sha256", "target_sha256", "topology_sha256")
CAPS = {"ppgt": (256, 1024), "esa": (1024, 4096)}


def _json(path):
    return json.loads(path.read_bytes())


def _publish(root, relative, value):
    path = _local(root, relative)
    publish_immutable_bytes(path, value if isinstance(value, bytes) else json_bytes(value))
    sync_directory(path.parent)
    return path


def _chunk_paths(role, index):
    prefix = "common_dev" if role == "common_dev" else "full_minus_holdout"
    stem = f"{role}_{index:06d}"
    return f"payloads/{prefix}/{stem}.pt", f"receipts/{stem}.json"


def _inventory(root, allowed, *, complete=False):
    """Inspect only the destination, and reject even empty extra directories."""
    directories = set()
    for name in allowed:
        directories.update(p.as_posix() for p in Path(name).parents if p != Path("."))
    found = set()
    if root.exists():
        for base, dirs, files in os.walk(root, followlinks=False):
            for name in (*dirs, *files):
                path = Path(base) / name
                _safe_local(path)
                relative = path.relative_to(root).as_posix()
                mode = path.lstat().st_mode
                if stat.S_ISDIR(mode):
                    if relative not in directories:
                        raise ValueError("Unexpected export directory")
                elif not stat.S_ISREG(mode) or relative not in allowed:
                    raise ValueError("Unexpected export file")
                else:
                    found.add(relative)
    if complete and found != allowed:
        raise ValueError("Incomplete export inventory")
    return found


def _prepare(*, parent_root, output_root, roles, permits, authorities, role_histories,
             parent_acceptance_sha256, max_chunk_bytes, max_rows):
    # Reject an expanded request before inspecting any filesystem location.
    if (not isinstance(roles, (tuple, list)) or len(roles) != 2 or
            set(roles) not in ({"train_100k", "common_dev"}, {"train_500k", "common_dev"})):
        raise ValueError("ROLE_BLOCKED: request exactly one training scale plus common_dev")
    roles = tuple(sorted(roles, key=lambda r: r == "common_dev"))
    if any(set(mapping) != set(roles) for mapping in (permits, authorities, role_histories, max_rows)):
        raise ValueError("Explicit bindings required for exactly the selected roles")
    digest(parent_acceptance_sha256, "externally supplied parent acceptance")
    if type(max_chunk_bytes) is not int or not 1 <= max_chunk_bytes <= 512 * 1024**2:
        raise ValueError("Declare a chunk budget of at most 512 MiB")
    types = {type(permits[r]) for r in roles}
    if len(types) != 1 or not types <= {IndexedInputPermit, AcceptedIndexedInputPermit}:
        raise TypeError("Uniform reviewed indexed permits required")
    real = types == {AcceptedIndexedInputPermit}
    for role in roles:
        permit = permits[role]
        if permit.role != role or permit.acceptance_sha256 != parent_acceptance_sha256:
            raise ValueError("Parent acceptance/role pin mismatch")
        permit.verify(authorities[role], role_histories[role])
        if type(max_rows[role]) is not int or not 1 <= max_rows[role] <= ROLE_LIMITS[role]:
            raise ValueError("Declare a row budget capped by the selected role")
        if max_chunk_bytes > (permit.max_chunk_bytes if real else 64 * 1024**2):
            raise ValueError("Chunk budget exceeds input permit")
    if len({permits[r].version_identity for r in roles}) != 1:
        raise ValueError("Selected roles have different version identities")
    parent, output = Path(parent_root).absolute(), Path(output_root).absolute()
    _safe_local(parent)
    _safe_local(output)
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("Disjoint parent and private export roots required")
    acceptance_path = _local(parent, "acceptance.json")
    if file_digest(acceptance_path) != parent_acceptance_sha256:
        raise ValueError("Parent acceptance bytes changed")
    acceptance = _json(acceptance_path)
    scope = "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY" if real else "CPU_SYNTHETIC_ONLY"
    status = "DATA_ROLE_ACCEPTED" if real else "SYNTHETIC_DATA_ROLES_VERIFIED"
    if (acceptance.get("scope") != scope or acceptance.get("status") != status or
            acceptance.get("rule_identity") != permits[roles[0]].version_identity):
        raise ValueError("Parent acceptance scope/version mismatch")
    if real and any(acceptance.get(key) is not False for key in (
            "official_validation_read", "test_roles_read", "graphs_rebuilt", "audit_label_statistics_exposed")):
        raise ValueError("Parent acceptance lacks confined provenance")
    manifests = {}
    allowed = {"export_request.json", "acceptance.json", "census.json"}
    for role in roles:
        relative = f"roles/{role}.json"
        path = _local(parent, relative)
        permit = permits[role]
        binding = acceptance.get("roles", {}).get(role, {})
        if (binding.get("path") != relative or binding.get("sha256") != permit.manifest_sha256 or
                file_digest(path) != permit.manifest_sha256):
            raise ValueError("Parent role manifest pin mismatch")
        m = _json(path)
        rows = m.get("rows")
        if (type(rows) is not int or not 1 <= rows <= max_rows[role] or
                (not real and rows > 512) or
                (real and rows != ROLE_LIMITS[role]) or
                m.get("role") != role or m.get("version_identity") != permit.version_identity or
                m.get("format") != FORMAT or m.get("status") != "complete" or
                m.get("feature_schema") != "ogb-atom9-bond3-rwse16-v1" or
                m.get("pure_2d") is not True or m.get("protected") is not False or
                m.get("ordering") != "ascending-source-index" or
                m.get("source_index_file") != f"members/{role}.i64le"):
            raise ValueError("Parent role ABI/row budget mismatch")
        chunks = m.get("graph_chunks")
        if type(chunks) is not list or not 1 <= len(chunks) <= rows:
            raise ValueError("Unbounded parent chunk map")
        total = 0
        for index, item in enumerate(chunks):
            if type(item.get("rows")) is not int or not 1 <= item["rows"] <= rows:
                raise ValueError("Unbounded parent chunk rows")
            total += item["rows"]
            prefix = "payloads/common_dev/" if role == "common_dev" else "payloads/full_minus_holdout/"
            if not isinstance(item.get("file"), str) or not item["file"].startswith(prefix):
                raise ValueError("ROLE_BLOCKED: invalid selected physical payload")
            # Size is checked before IndexedRoleDataset's eager chunk decoding.
            source = _local(parent, item["file"])
            if not stat.S_ISREG(source.stat().st_mode) or source.stat().st_size > max_chunk_bytes:
                raise ValueError("Source chunk exceeds declared byte budget")
            offsets = item.get("offsets")
            if offsets is not None and (type(offsets) is not list or len(offsets) != item["rows"]):
                raise ValueError("Unbounded parent offsets")
            for key in (*LOGICAL_FIELDS, "sha256"):
                digest(item[key], key)
            allowed.update(_chunk_paths(role, index))
        logical = canonical_fingerprint({"members": m["source_index_sha256"],
            "chunks": [{k: c[k] for k in LOGICAL_FIELDS} for c in chunks]})
        if total != rows or logical != m["logical_content_sha256"] or logical != binding.get("logical_content_sha256"):
            raise ValueError("Parent logical identity/count mismatch")
        manifests[role] = m
        allowed.update({relative, f"members/{role}.i64le"})
    request = {"format": EXPORT_FORMAT, "private_required": True, "roles": list(roles),
        "parent_acceptance_sha256": parent_acceptance_sha256,
        "permits": {r: asdict(permits[r]) for r in roles}, "max_chunk_bytes": max_chunk_bytes,
        "max_rows": dict(max_rows), "scope": scope, "status": status}
    _inventory(output, allowed | {"failure.json"})
    if (output / "failure.json").exists():
        raise ValueError("Export failed previously; preserved evidence requires a separate output root")
    return parent, output, roles, manifests, request, allowed


def _empty_census():
    return {"rows": 0, "node_histogram": {}, "directed_edge_histogram": {},
        "isolated_atom_count": 0, "zero_edge_graph_count": 0, "max_nodes": 0,
        "max_directed_edges": 0, "exceeding_caps_source_ids": {name: [] for name in CAPS}}


def _observe(census, graph):
    import torch
    nodes, edges = int(graph.num_nodes), int(graph.edge_index.shape[1])
    endpoints = graph.edge_index.reshape(-1)
    if endpoints.numel() and (int(endpoints.min()) < 0 or int(endpoints.max()) >= nodes):
        raise ValueError("Graph endpoint outside node range")
    census["rows"] += 1
    for field, count in (("node_histogram", nodes), ("directed_edge_histogram", edges)):
        key = str(count)
        census[field][key] = census[field].get(key, 0) + 1
    census["isolated_atom_count"] += nodes - int(torch.unique(endpoints).numel())
    census["zero_edge_graph_count"] += int(edges == 0)
    census["max_nodes"] = max(census["max_nodes"], nodes)
    census["max_directed_edges"] = max(census["max_directed_edges"], edges)
    for name, (node_cap, edge_cap) in CAPS.items():
        if nodes > node_cap or edges > edge_cap:
            census["exceeding_caps_source_ids"][name].append(int(graph.source_idx.item()))


def _merge_census(chunks):
    result = _empty_census()
    for chunk in chunks:
        for key in ("rows", "isolated_atom_count", "zero_edge_graph_count"):
            result[key] += chunk[key]
        for key in ("max_nodes", "max_directed_edges"):
            result[key] = max(result[key], chunk[key])
        for key in ("node_histogram", "directed_edge_histogram"):
            for value, count in chunk[key].items():
                result[key][value] = result[key].get(value, 0) + count
        for name in CAPS:
            result["exceeding_caps_source_ids"][name].extend(chunk["exceeding_caps_source_ids"][name])
    return result


def _graph_schema(graph):
    import torch
    from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
    if set(graph.keys()) - GRAPH_FIELDS - {"num_nodes"} or not GRAPH_FIELDS <= set(graph.keys()):
        raise ValueError("Unknown/missing graph fields in selected payload")
    if any(not torch.is_tensor(graph[k]) or graph[k].device.type != "cpu" for k in GRAPH_FIELDS):
        raise ValueError("CPU graph tensors required")
    if (graph.x.ndim != 2 or graph.x.shape[1] != 9 or graph.x.dtype != torch.int64 or not len(graph.x) or
            graph.edge_attr.dtype != torch.int64 or graph.edge_index.ndim != 2 or
            graph.edge_index.shape[0] != 2 or graph.edge_index.dtype != torch.int64 or
            graph.edge_attr.shape != (graph.edge_index.shape[1], 3) or
            graph.random_walk_pe.shape != (graph.num_nodes, 16) or
            graph.source_idx.dtype != torch.int64 or graph.source_idx.numel() != 1 or
            graph.y.numel() != 1 or not graph.y.is_floating_point() or
            not bool(torch.isfinite(graph.y).all()) or not bool(torch.isfinite(graph.random_walk_pe).all())):
        raise ValueError("Selected graph schema differs from accepted ABI")
    if bool(((graph.x < 0) | (graph.x >= graph.x.new_tensor(ATOM_FEATURE_DIMS))).any()) or bool(
            ((graph.edge_attr < 0) | (graph.edge_attr >= graph.edge_attr.new_tensor(BOND_FEATURE_DIMS))).any()):
        raise ValueError("Selected categorical features are outside the accepted ABI")


def _facts(payload, rows):
    data, slices = payload
    if set(data.keys()) - GRAPH_FIELDS - {"num_nodes"}:
        raise ValueError("Unknown graph fields in packed export")
    ids = data.source_idx.reshape(-1).tolist()
    values = data.y.reshape(-1).tolist()
    if len(ids) != rows or len(values) != rows:
        raise ValueError("Decoded selected row count mismatch")
    topology = {k: v for k, v in data.to_dict().items() if k not in {"source_idx", "y"}}
    return {"rows": rows, "row_sha256": index_digest(ids), "target_sha256": target_digest(values),
        "topology_sha256": state_digest({"data": topology,
            "slices": {k: v for k, v in (slices or {}).items() if k not in {"source_idx", "y"}}})}


def _match(facts, parent_chunk):
    if any(facts[k] != parent_chunk[k] for k in ("rows", *LOGICAL_FIELDS)):
        raise ValueError("Decoded export differs from parent row/target/topology pins")


def _decode(path, parent_chunk, budget):
    if path.stat().st_size > budget:
        raise ValueError("Export chunk exceeds declared byte budget")
    shard = PackedGraphDataset(path, mmap=True)
    if len(shard) != parent_chunk["rows"]:
        raise ValueError("Export contains unselected graphs")
    facts = _facts((shard._data, shard.slices), len(shard))
    _match(facts, parent_chunk)
    census = _empty_census()
    for offset in range(len(shard)):
        graph = shard.get(offset)
        _graph_schema(graph)
        _observe(census, graph)
        shard._data_list = None
    return facts, census


def _base_receipt(request, role, index, item):
    file, _ = _chunk_paths(role, index)
    return {"format": EXPORT_FORMAT + "-chunk", "status": "CONTENT_VERIFIED",
        "request_identity": canonical_fingerprint(request), "role": role, "parent_chunk_index": index,
        "parent_manifest_sha256": request["permits"][role]["manifest_sha256"],
        "parent_chunk_identity": canonical_fingerprint(item), "file": file,
        "offsets": None, "rows": item["rows"], **{k: item[k] for k in LOGICAL_FIELDS}}


def _check_chunk(output, request, role, index, item):
    file, receipt_file = _chunk_paths(role, index)
    receipt = _json(_local(output, receipt_file))
    base = _base_receipt(request, role, index, item)
    if set(receipt) != set(base) | {"sha256", "bytes", "census"} or any(receipt[k] != v for k, v in base.items()):
        raise ValueError("Export chunk receipt differs from parent binding")
    path = _local(output, file)
    if file_digest(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("Export physical chunk identity changed")
    _, census = _decode(path, item, request["max_chunk_bytes"])
    if census != receipt["census"]:
        raise ValueError("Export census changed")
    return receipt


def _role_manifest(parent, request, role, receipts):
    # Do not copy parent checks, held-out metadata, identity DB paths or statistics.
    keys = ("format", "status", "version_identity", "role", "rows", "source_index_sha256",
        "ordering", "source_index_file", "source_index_file_sha256", "feature_schema", "pure_2d",
        "protected", "access", "label_statistics_published", "logical_content_sha256")
    m = {k: parent[k] for k in keys}
    m.update(parent_acceptance_sha256=request["parent_acceptance_sha256"],
        parent_manifest_sha256=request["permits"][role]["manifest_sha256"],
        graph_chunks=[{k: c[k] for k in ("file", "sha256", "bytes", "rows", "offsets", *LOGICAL_FIELDS)} for c in receipts])
    return m


def _completion(output, roles, manifests, request, receipts, allowed):
    bindings = {}
    for role in roles:
        path = _local(output, f"roles/{role}.json")
        expected = _role_manifest(manifests[role], request, role, receipts[role])
        if path.read_bytes() != json_bytes(expected):
            raise ValueError("Export role manifest differs from pinned parent")
        member = _local(output, expected["source_index_file"])
        if member.stat().st_size != expected["rows"] * 8 or file_digest(member) != expected["source_index_file_sha256"]:
            raise ValueError("Export member identity changed")
        bindings[role] = {"path": f"roles/{role}.json", "sha256": file_digest(path),
            **{k: expected[k] for k in ("rows", "source_index_sha256", "logical_content_sha256")}}
    census = {"format": EXPORT_FORMAT + "-census", "label_free": True,
        "request_identity": canonical_fingerprint(request), "caps": {k: {"nodes": v[0], "directed_edges": v[1]} for k, v in CAPS.items()},
        "roles": {r: _merge_census([c["census"] for c in receipts[r]]) for r in roles}}
    return bindings, census


def _acceptance(output, request, bindings, allowed):
    inventory = {name: {"sha256": file_digest(_local(output, name)),
        "bytes": _local(output, name).stat().st_size} for name in sorted(allowed - {"acceptance.json"})}
    return {"format": EXPORT_FORMAT + "-acceptance", "status": request["status"], "scope": request["scope"],
        "rule_identity": request["permits"][request["roles"][0]]["version_identity"],
        "parent_acceptance_sha256": request["parent_acceptance_sha256"], "private_required": True,
        "request_identity": canonical_fingerprint(request), "roles": bindings, "inventory": inventory,
        "official_validation_read": False, "test_roles_read": False, "graphs_rebuilt": False,
        "audit_label_statistics_exposed": False, "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING",
        "publication_status": "NOT_PUBLISHED"}


def verify_indexed_export(*, export_acceptance_sha256, **kwargs):
    """Independently verify physical bytes AND decoded equivalence to parent pins.

    Supply the retained export acceptance SHA after retrieval. Repinning output
    metadata cannot authorize a different row, target, topology or source role.
    """
    digest(export_acceptance_sha256, "externally supplied export acceptance")
    _, output, roles, manifests, request, allowed = _prepare(**kwargs)
    _inventory(output, allowed, complete=True)
    if file_digest(_local(output, "acceptance.json")) != export_acceptance_sha256:
        raise ValueError("Export acceptance bytes changed")
    if _local(output, "export_request.json").read_bytes() != json_bytes(request):
        raise ValueError("Export request changed")
    receipts = {r: [_check_chunk(output, request, r, i, item) for i, item in enumerate(manifests[r]["graph_chunks"])] for r in roles}
    bindings, census = _completion(output, roles, manifests, request, receipts, allowed)
    if _local(output, "census.json").read_bytes() != json_bytes(census):
        raise ValueError("Export aggregate census changed")
    expected = _acceptance(output, request, bindings, allowed)
    if _local(output, "acceptance.json").read_bytes() != json_bytes(expected):
        raise ValueError("Export inventory/acceptance changed")
    return expected


def export_indexed_roles(*, parent_root, output_root, roles, permits, authorities,
                         role_histories, parent_acceptance_sha256, max_chunk_bytes, max_rows):
    """Export exactly (train_100k OR train_500k) + common_dev, privately.

    Inputs are externally pinned permits and their explicit authority/history
    mappings. max_rows is a per-selected-role budget. Source and output chunk
    bytes are bounded by max_chunk_bytes. Return the loader-compatible acceptance
    record; pin its on-disk SHA externally before moving the export. census.json
    and each chunk receipt contain label-free coverage counts only. Neither this
    export nor its census qualifies a model or authorizes scientific training.
    """
    kwargs = dict(parent_root=parent_root, output_root=output_root, roles=roles, permits=permits,
        authorities=authorities, role_histories=role_histories,
        parent_acceptance_sha256=parent_acceptance_sha256, max_chunk_bytes=max_chunk_bytes, max_rows=max_rows)
    parent, output, roles, manifests, request, allowed = _prepare(**kwargs)
    if (output / "acceptance.json").exists():
        # A completed export is checked without repairing or overwriting it.
        return verify_indexed_export(export_acceptance_sha256=file_digest(output / "acceptance.json"), **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(output, 0o700)
    found = _inventory(output, allowed)
    if found and "export_request.json" not in found:
        raise ValueError("Unbound export destination")
    _publish(output, "export_request.json", request)
    receipts = {}
    role, index = None, None
    try:
        for role in roles:
            dataset = IndexedRoleDataset(parent, permits[role], authority=authorities[role], role_history=role_histories[role])
            if dataset.manifest != manifests[role]:
                raise ValueError("Parent manifest changed during export")
            receipts[role] = []
            for index, item in enumerate(manifests[role]["graph_chunks"]):
                file, receipt_file = _chunk_paths(role, index)
                path, receipt_path = _local(output, file), _local(output, receipt_file)
                if path.exists():
                    if not receipt_path.exists():
                        raise ValueError("Unbound physical export chunk")
                    receipts[role].append(_check_chunk(output, request, role, index, item))
                    continue
                import torch
                from torch_geometric.data import InMemoryDataset
                shard = dataset._load(index)
                graphs, census = [], _empty_census()
                for offset in dataset._maps[index]:
                    # clone compacts selected storages; serializing views can leak
                    # unselected backing tensors. Bypass runtime role_name tags.
                    graph = shard.get(offset).clone()
                    _graph_schema(graph)
                    _observe(census, graph)
                    graphs.append(graph)
                    shard._data_list = None
                payload = InMemoryDataset.collate(graphs)
                facts = _facts(payload, len(graphs))
                try:
                    _match(facts, item)
                except ValueError:
                    _publish(output, "failure.json", {"format": EXPORT_FORMAT + "-failure", "role": role,
                        "chunk": index, "expected": {k: item[k] for k in ("rows", *LOGICAL_FIELDS)}, "observed": facts})
                    raise
                del graphs, shard
                _safe_local(output.parent)
                with tempfile.TemporaryDirectory(prefix="molgap_selected_export_", dir=output.parent) as scratch:
                    scratch_path = Path(scratch)
                    _safe_local(scratch_path)
                    if scratch_path.resolve().parent != output.parent.resolve():
                        raise ValueError("Scratch directory escaped the export parent")
                    temporary = scratch_path / "selected.pt"
                    # A stream keeps torch's archive name stable for receipt-first
                    # retries, independently of the temporary directory name.
                    with temporary.open("wb") as stream:
                        torch.save(payload, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    del payload
                    if temporary.stat().st_size > max_chunk_bytes:
                        raise ValueError("Serialized export chunk exceeds declared byte budget")
                    decoded, decoded_census = _decode(temporary, item, max_chunk_bytes)
                    if decoded != facts or decoded_census != census:
                        raise ValueError("Serialized selected tensors/census changed")
                    receipt = {**_base_receipt(request, role, index, item),
                        "sha256": file_digest(temporary), "bytes": temporary.stat().st_size, "census": census}
                    # If interrupted here, regeneration must match this immutable
                    # physical pin before the missing payload can be published.
                    _publish(output, receipt_file, receipt)
                    copy_immutable_file(temporary, path, expected_sha256=receipt["sha256"])
                    sync_directory(path.parent)
                receipts[role].append(receipt)
            member = _local(parent, manifests[role]["source_index_file"])
            copy_immutable_file(member, _local(output, manifests[role]["source_index_file"]),
                expected_sha256=manifests[role]["source_index_file_sha256"])
            _publish(output, f"roles/{role}.json", _role_manifest(manifests[role], request, role, receipts[role]))
            del dataset
        bindings, census = _completion(output, roles, manifests, request, receipts, allowed)
        _publish(output, "census.json", census)
        _inventory(output, allowed - {"acceptance.json"}, complete=True)
        # Recheck supplied parent metadata pins immediately before committing.
        _prepare(**kwargs)
        acceptance = _acceptance(output, request, bindings, allowed)
        _publish(output, "acceptance.json", acceptance)
        return acceptance
    except ValueError as error:
        if not (output / "failure.json").exists():
            _publish(output, "failure.json", {"format": EXPORT_FORMAT + "-failure", "role": role,
                "chunk": index, "reason": type(error).__name__, "parent_acceptance_sha256": parent_acceptance_sha256})
        raise


__all__ = ["export_indexed_roles", "verify_indexed_export"]
