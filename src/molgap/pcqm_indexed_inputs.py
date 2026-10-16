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

from .common.io import _safe_local
from .common.wire import canonical_fingerprint
from .cross_scale_plan import digest
from .cross_scale_inputs import index_digest,target_digest
from .cross_scale_checkpoint import state_digest
from .pcqm_unified_roles import FORMAT,GRAPH_FIELDS,authorize_role,_pin
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest


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


class IndexedRoleDataset:
    """One retained packed chunk at a time, explicit offsets, immutable copies."""
    def __init__(self,root,permit,*,authority,role_history):
        if type(permit) is not IndexedInputPermit:
            raise TypeError("Reviewed indexed input permit required")
        permit.verify(authority,role_history)  # Before opening members or payloads.
        self.root=Path(root).absolute();_safe_local(self.root)
        receipt_path=repo_local_path(self.root,"acceptance.json")
        _safe_local(receipt_path)
        if file_digest(receipt_path) != permit.acceptance_sha256:
            raise ValueError("Indexed acceptance bytes changed")
        receipt=json.loads(receipt_path.read_bytes())
        if (receipt.get("scope") != "CPU_SYNTHETIC_ONLY" or receipt.get("status") != "SYNTHETIC_DATA_ROLES_VERIFIED" or
                receipt.get("rule_identity") != permit.version_identity):
            raise ValueError("Real indexed V2/native loading is not released")
        binding=receipt.get("roles",{}).get(permit.role,{})
        expected_path="roles/"+permit.role+".json"
        if binding.get("path") != expected_path or binding.get("sha256") != permit.manifest_sha256:
            raise ValueError("Indexed manifest differs from acceptance")
        manifest_path=repo_local_path(self.root,expected_path)
        _safe_local(manifest_path)
        if file_digest(manifest_path) != permit.manifest_sha256:
            raise ValueError("Indexed role manifest bytes changed")
        self.manifest=json.loads(manifest_path.read_bytes())
        m=self.manifest
        if (m.get("role") != permit.role or m.get("version_identity") != permit.version_identity or
                m.get("format") != FORMAT or m.get("feature_schema") != "ogb-atom9-bond3-rwse16-v1" or
                m.get("pure_2d") is not True or m.get("ordering") != "ascending-source-index" or
                type(m.get("rows")) is not int or not 1 <= m["rows"] <= 512):
            raise ValueError("Unknown/unbounded CPU indexed role ABI")
        for operation in ("prediction_input","labels_read"):
            authorize_role(m,operation=operation)
        member=repo_local_path(self.root,m["source_index_file"])
        _safe_local(member)
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
        self._ends=[];decoded=[]
        for index,item in enumerate(m["graph_chunks"]):
            shard=self._load(index)
            offsets=list(range(len(shard))) if item["offsets"] is None else item["offsets"]
            if (type(offsets) is not list or any(type(i) is not int or not 0 <= i < len(shard) for i in offsets) or
                    offsets != sorted(set(offsets)) or len(offsets) != item["rows"]):
                raise ValueError("Invalid non-contiguous indexed offsets")
            graphs=[shard.get(i) for i in offsets]
            if any(set(g.keys())-GRAPH_FIELDS-{"num_nodes"} for g in graphs):
                raise ValueError("Unknown/3D indexed graph field")
            ids=[int(g.source_idx.item()) for g in graphs]
            if index_digest(ids) != item["row_sha256"] or target_digest([float(g.y.item()) for g in graphs]) != item["target_sha256"]:
                raise ValueError("Decoded indexed row/target identity changed")
            from torch_geometric.data import InMemoryDataset
            data,slices=InMemoryDataset.collate(graphs)
            topology={k:v for k,v in data.to_dict().items() if k not in {"source_idx","y"}}
            if state_digest({"data":topology,"slices":{k:v for k,v in (slices or {}).items() if k not in {"source_idx","y"}}}) != item["topology_sha256"]:
                raise ValueError("Decoded indexed topology identity changed")
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
        path=repo_local_path(self.root,item["file"])
        _safe_local(path)
        if path.stat().st_size > 64*1024**2:
            raise ValueError("CPU indexed chunk safety cap exceeded")
        path=_pin(self.root,item)
        if self._loaded_index != index:
            from .packed_graph_dataset import PackedGraphDataset
            self._loaded=PackedGraphDataset(path,mmap=True);self._loaded_index=index
        return self._loaded

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self,index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Indexed role row out of range")
        chunk=bisect_right(self._ends,index)
        local=index-(self._ends[chunk-1] if chunk else 0)
        return self._load(chunk).get(self._maps[chunk][local]).clone()


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
