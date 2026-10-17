"""CPU-qualified prototype of the accepted V2 indexed-role read path.

Reuses immutable role maps and PackedGraphDataset. No member selection, graph
building, train loop, role promotion, upload or real/native release lives here.
Only generated acceptance fixtures can be opened in this implementation stage.
"""
from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
import struct

from ..common.io import _safe_local
from ..common.wire import canonical_fingerprint
from ..cross_scale_plan import digest
from ..cross_scale_inputs import index_digest,target_digest
from ..cross_scale_checkpoint import state_digest
from ..pcqm_unified_roles import FORMAT,GRAPH_FIELDS,authorize_role,_pin
from ..research_memory.paths import repo_local_path
from ..research_memory.trace import file_digest


def _local(root, relative):
    # Check the unresolved components first: resolve() would hide an internal
    # junction into another role even if it remained within the dataset root.
    _safe_local(root / relative)
    return repo_local_path(root, relative)


@dataclass(frozen=True)
class IndexedInputPermit:
    role: str
    acceptance_sha256: str
    manifest_sha256: str
    version_identity: str
    authority_sha256: str
    role_history_sha256: str
    scope: str = "CPU_SYNTHETIC_ONLY"

    def __post_init__(self):
        if self.scope != "CPU_SYNTHETIC_ONLY" or self.role not in {"train_100k","train_500k","common_dev"}:
            raise ValueError("ROLE_BLOCKED: CPU fixture profile only; real/audit/full roles unreleased")
        for key in ("acceptance_sha256","manifest_sha256","version_identity","authority_sha256","role_history_sha256"):
            digest(getattr(self,key),key)

    def verify(self,authority,history):
        if (canonical_fingerprint(authority) != self.authority_sha256 or
                canonical_fingerprint(history) != self.role_history_sha256 or
                authority != {"scope":self.scope,"role":self.role,"manifest_sha256":self.manifest_sha256,
                    "operations":["prediction_input","labels_read"],"protected":False} or
                history != {"role":self.role,"manifest_sha256":self.manifest_sha256,
                    "usage":"training" if self.role.startswith("train_") else "selection-used"}):
            raise ValueError("ROLE_BLOCKED: indexed role authority/history mismatch")


@dataclass(frozen=True)
class AcceptedIndexedInputPermit:
    """Explicit accepted-data binding; possession is not training authority.

    Pins must be supplied by the owning accepted-data record, never discovered
    from an untrusted mount. Only train100K/500K and Common-Dev are admissible.
    The synthetic permit above deliberately retains its original restrictions.
    """
    role: str
    acceptance_sha256: str
    manifest_sha256: str
    version_identity: str
    authority_sha256: str
    role_history_sha256: str
    max_chunk_bytes: int
    scope: str = "ACCEPTED_INDEXED_INPUT_V1"

    def __post_init__(self):
        if self.scope != "ACCEPTED_INDEXED_INPUT_V1" or self.role not in {"train_100k", "train_500k", "common_dev"}:
            raise ValueError("ROLE_BLOCKED: accepted indexed train/Common-Dev only")
        for key in ("acceptance_sha256", "manifest_sha256", "version_identity", "authority_sha256", "role_history_sha256"):
            digest(getattr(self, key), key)
        if type(self.max_chunk_bytes) is not int or not 1 <= self.max_chunk_bytes <= 512 * 1024**2:
            raise ValueError("An explicit bounded CPU chunk budget is required")

    def verify(self, authority, history):
        if (canonical_fingerprint(authority) != self.authority_sha256 or
                canonical_fingerprint(history) != self.role_history_sha256 or
                authority != {"scope": self.scope, "role": self.role,
                    "acceptance_sha256": self.acceptance_sha256, "manifest_sha256": self.manifest_sha256,
                    "version_identity": self.version_identity, "operations": ["prediction_input", "labels_read"],
                    "protected": False, "max_chunk_bytes": self.max_chunk_bytes} or
                history != {"role": self.role, "manifest_sha256": self.manifest_sha256,
                    "usage": "training" if self.role.startswith("train_") else "selection-used"}):
            raise ValueError("ROLE_BLOCKED: accepted indexed authority/history mismatch")


class IndexedRoleDataset:
    """One retained packed chunk at a time, explicit offsets, immutable copies."""
    def __init__(self,root,permit,*,authority,role_history):
        if type(permit) not in (IndexedInputPermit, AcceptedIndexedInputPermit):
            raise TypeError("Reviewed indexed input permit required")
        permit.verify(authority,role_history)  # Before opening members or payloads.
        self.permit = permit
        real = type(permit) is AcceptedIndexedInputPermit
        self.root=Path(root).absolute();_safe_local(self.root)
        receipt_path=_local(self.root,"acceptance.json")
        if file_digest(receipt_path) != permit.acceptance_sha256:
            raise ValueError("Indexed acceptance bytes changed")
        receipt=json.loads(receipt_path.read_bytes())
        if (receipt.get("scope") != ("OFFICIAL_TRAIN_CPU_PREPARATION_ONLY" if real else "CPU_SYNTHETIC_ONLY") or
                receipt.get("status") != ("DATA_ROLE_ACCEPTED" if real else "SYNTHETIC_DATA_ROLES_VERIFIED") or
                receipt.get("rule_identity") != permit.version_identity):
            raise ValueError("Real indexed V2/native loading is not released")
        if real and any(receipt.get(key) is not False for key in (
                "official_validation_read", "test_roles_read", "graphs_rebuilt", "audit_label_statistics_exposed")):
            raise ValueError("Accepted indexed receipt lacks confined data provenance")
        binding=receipt.get("roles",{}).get(permit.role,{})
        expected_path="roles/"+permit.role+".json"
        if binding.get("path") != expected_path or binding.get("sha256") != permit.manifest_sha256:
            raise ValueError("Indexed manifest differs from acceptance")
        manifest_path=_local(self.root,expected_path)
        if file_digest(manifest_path) != permit.manifest_sha256:
            raise ValueError("Indexed role manifest bytes changed")
        self.manifest=json.loads(manifest_path.read_bytes())
        m=self.manifest
        if (m.get("role") != permit.role or m.get("version_identity") != permit.version_identity or
                m.get("format") != FORMAT or m.get("feature_schema") != "ogb-atom9-bond3-rwse16-v1" or
                m.get("pure_2d") is not True or m.get("ordering") != "ascending-source-index" or
                type(m.get("rows")) is not int or not 1 <= m["rows"] <= (500000 if real else 512)):
            raise ValueError("Unknown/unbounded CPU indexed role ABI")
        if real and m["rows"] != {"train_100k": 100000, "train_500k": 500000, "common_dev": 50000}[permit.role]:
            raise ValueError("Accepted indexed role count differs from the frozen scale")
        for operation in ("prediction_input","labels_read"):
            authorize_role(m,operation=operation)
        member=_local(self.root,m["source_index_file"])
        if m["source_index_file"] != "members/"+permit.role+".i64le" or member.stat().st_size != m["rows"]*8:
            raise ValueError("Indexed member path/count mismatch")
        if file_digest(member) != m["source_index_file_sha256"]:
            raise ValueError("Indexed member bytes changed")
        self.source_indices=tuple(v[0] for v in struct.iter_unpack("<q",member.read_bytes()))
        if list(self.source_indices) != sorted(set(self.source_indices)) or index_digest(list(self.source_indices)) != m["source_index_sha256"]:
            raise ValueError("Indexed member identity/order mismatch")
        logical=canonical_fingerprint({"members":m["source_index_sha256"],"chunks":[
            {k:c[k] for k in ("row_sha256","target_sha256","topology_sha256")} for c in m["graph_chunks"]]})
        if logical != m["logical_content_sha256"] or logical != binding.get("logical_content_sha256"):
            raise ValueError("Indexed logical content mismatch")
        self._loaded_index,self._loaded,self._maps=None,None,[]
        self._loaded_file_identity=None
        self._ends=[];decoded=[]
        for index,item in enumerate(m["graph_chunks"]):
            shard=self._load(index)
            offsets=list(range(len(shard))) if item["offsets"] is None else item["offsets"]
            if (type(offsets) is not list or any(type(i) is not int or not 0 <= i < len(shard) for i in offsets) or
                    offsets != sorted(set(offsets)) or len(offsets) != item["rows"]):
                raise ValueError("Invalid non-contiguous indexed offsets")
            if real:
                # The independently accepted, byte-pinned shard already binds
                # topology. Do not duplicate a whole collated chunk in RAM.
                import torch
                if set(shard._data.keys()) - GRAPH_FIELDS - {"num_nodes"}:
                    raise ValueError("Unknown/3D indexed graph field")
                source, target = shard._data.source_idx.reshape(-1), shard._data.y.reshape(-1)
                if source.dtype != torch.int64 or len(source) != len(shard) or len(target) != len(shard):
                    raise ValueError("Invalid decoded indexed identity/target schema")
                chosen = torch.tensor(offsets, dtype=torch.int64)
                ids = source[chosen].tolist()
                values = target[chosen]
                if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
                    raise ValueError("Decoded indexed targets are nonfinite")
                values = values.tolist()
            else:
                graphs=[shard.get(i) for i in offsets]
                if any(set(g.keys())-GRAPH_FIELDS-{"num_nodes"} for g in graphs):
                    raise ValueError("Unknown/3D indexed graph field")
                ids=[int(g.source_idx.item()) for g in graphs]
                values=[float(g.y.item()) for g in graphs]
                from torch_geometric.data import InMemoryDataset
                data,slices=InMemoryDataset.collate(graphs)
                topology={k:v for k,v in data.to_dict().items() if k not in {"source_idx","y"}}
                if state_digest({"data":topology,"slices":{k:v for k,v in (slices or {}).items() if k not in {"source_idx","y"}}}) != item["topology_sha256"]:
                    raise ValueError("Decoded indexed topology identity changed")
            if index_digest(ids) != item["row_sha256"] or target_digest(values) != item["target_sha256"]:
                raise ValueError("Decoded indexed row/target identity changed")
            decoded.extend(ids);self._maps.append(tuple(offsets));self._ends.append(len(decoded))
        if tuple(decoded) != self.source_indices:
            raise ValueError("Decoded chunks differ from explicit indexed members")
        self.permit=permit

    def _load(self,index):
        item=self.manifest["graph_chunks"][index]
        role=self.manifest["role"]
        prefix="payloads/common_dev/" if role == "common_dev" else "payloads/full_minus_holdout/"
        if not item["file"].startswith(prefix):
            raise ValueError("ROLE_BLOCKED: wrong/audit physical payload")
        path=_local(self.root,item["file"])
        info=path.stat()
        limit = self.permit.max_chunk_bytes if type(self.permit) is AcceptedIndexedInputPermit else 64*1024**2
        if info.st_size > limit:
            raise ValueError("CPU indexed chunk safety cap exceeded")
        identity=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,item["sha256"])
        if self._loaded_index != index or self._loaded_file_identity != identity:
            path=_pin(self.root,item)
            from .packed_graph import PackedGraphDataset
            loaded=PackedGraphDataset(path,mmap=True)
            after=path.stat()
            if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,item["sha256"]) != identity:
                raise ValueError("Indexed shard changed while opening")
            self._loaded,self._loaded_index=loaded,index
            self._loaded_file_identity=identity
        return self._loaded

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self,index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Indexed role row out of range")
        chunk=bisect_right(self._ends,index)
        local=index-(self._ends[chunk-1] if chunk else 0)
        graph = self._load(chunk).get(self._maps[chunk][local]).clone()
        if int(graph.source_idx.item()) != self.source_indices[index]:
            raise ValueError("Decoded row identity changed during indexed access")
        # Clearing PyG's per-graph memoization keeps the one-chunk memory bound.
        self._loaded._data_list = None
        graph.role_name = "CPU_SYNTHETIC_ONLY" if type(self.permit) is IndexedInputPermit else self.permit.role
        return graph

    def selected_targets(self):
        """Train-only statistics callers need labels, not cloned graph lists."""
        import torch
        for chunk, offsets in enumerate(self._maps):
            shard = self._load(chunk)
            yield shard._data.y.reshape(-1)[torch.tensor(offsets, dtype=torch.int64)].double().clone()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_loaded=None, _loaded_index=None, _loaded_file_identity=None)
        return state


def mirror_prerequisites(acceptance):
    """Metadata-only private export plan, never an uploader or publication receipt."""
    if acceptance.get("status") != "DATA_ROLE_ACCEPTED":
        raise ValueError("Accepted IMS master required for a real mirror draft")
    roles={r:acceptance["roles"][r] for r in ("train_100k","train_500k","common_dev")}
    return {"status":"NOT_READY_FOR_PUBLICATION","private_required":True,"roles":roles,
        "excluded":["independent_audit","private_identity.sqlite","original_full_shards"],
        "missing":["authorized export transaction","exact safe payload inventory","decoded equivalence if repackaged",
                   "private account/dataset binding","owning platform validation/publication receipt","native loader qualification"],
        "platforms":{"kaggle1":"NOT_PUBLISHED","kaggle2":"NOT_PUBLISHED"},
        "automatic_training_action":False}


# Retain serialized type/function references without duplicating any definitions.
_local.__module__ = "molgap.pcqm_indexed_inputs"
IndexedInputPermit.__module__ = "molgap.pcqm_indexed_inputs"
IndexedRoleDataset.__module__ = "molgap.pcqm_indexed_inputs"
mirror_prerequisites.__module__ = "molgap.pcqm_indexed_inputs"
