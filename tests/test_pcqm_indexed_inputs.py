"""Synthetic accepted V2 maps only; no real role or platform access."""
from copy import deepcopy
import json
import pytest

from molgap.pcqm_indexed_inputs import IndexedInputPermit,IndexedRoleDataset,mirror_prerequisites
from molgap.pcqm_unified_roles import prepare_unified_roles
from molgap.common.wire import canonical_fingerprint
from molgap.research_memory.trace import file_digest,json_bytes
from test_pcqm_unified_roles import source_fixture


def fixture(root,role="common_dev"):
    fixed,rule=source_fixture(root);out=root/"unified"
    acceptance=prepare_unified_roles(boundary=root,fixed_root=fixed,output_root=out,rule=rule)
    manifest=out/acceptance["roles"][role]["path"]
    authority={"scope":"CPU_SYNTHETIC_ONLY","role":role,"manifest_sha256":file_digest(manifest),
        "operations":["prediction_input","labels_read"],"protected":False}
    history={"role":role,"manifest_sha256":file_digest(manifest),"usage":"training" if role.startswith("train_") else "selection-used"}
    permit=IndexedInputPermit(role,file_digest(out/"acceptance.json"),file_digest(manifest),acceptance["rule_identity"],
        canonical_fingerprint(authority),canonical_fingerprint(history))
    return out,permit,authority,history


@pytest.mark.parametrize("role",["train_100k","train_500k","common_dev"])
def test_existing_payloads_noncontiguous_maps_exact_identity_and_no_rebuild(tmp_path,role):
    root,p,a,h=fixture(tmp_path,role)
    before={str(f.relative_to(root)):file_digest(f) for f in root.rglob('*') if f.is_file()}
    data=IndexedRoleDataset(root,p,authority=a,role_history=h)
    assert [int(data[i].source_idx.item()) for i in range(len(data))]==list(data.source_indices)
    original=data[0].x.clone();data[0].x.zero_();assert (data[0].x==original).all()
    assert before=={str(f.relative_to(root)):file_digest(f) for f in root.rglob('*') if f.is_file()}


@pytest.mark.parametrize("kind",["authority","history","acceptance","manifest","members","payload"])
def test_tampered_indexed_evidence_fails_closed(tmp_path,kind):
    root,p,a,h=fixture(tmp_path)
    if kind=="authority": a["protected"]=True
    elif kind=="history": h["usage"]="unseen"
    elif kind=="acceptance": (root/"acceptance.json").write_bytes(b'{}')
    elif kind=="manifest": (root/"roles/common_dev.json").write_bytes(b'{}')
    elif kind=="members": (root/"members/common_dev.i64le").write_bytes(b'bad')
    else:
        m=json.loads((root/"roles/common_dev.json").read_bytes())
        (root/m["graph_chunks"][0]["file"]).write_bytes(b'bad')
    with pytest.raises(ValueError): IndexedRoleDataset(root,p,authority=a,role_history=h)


@pytest.mark.parametrize("role",["independent_audit","audit","official_validation","test_dev","train_1m","full_minus_holdout"])
def test_protected_or_unreleased_roles_rejected_before_io(role):
    with pytest.raises(ValueError,match="ROLE_BLOCKED"):
        IndexedInputPermit(role,*(["a"*64]*5))


def test_real_scope_denied_before_packed_payload_open(tmp_path,monkeypatch):
    from dataclasses import replace
    from molgap import packed_graph_dataset
    root,p,a,h=fixture(tmp_path)
    acceptance=json.loads((root/"acceptance.json").read_bytes());acceptance.update(scope="OFFICIAL_TRAIN_CPU_PREPARATION_ONLY",status="DATA_ROLE_ACCEPTED")
    (root/"acceptance.json").write_bytes(json_bytes(acceptance))
    def forbidden(*a,**k): raise AssertionError("Real payload was opened")
    monkeypatch.setattr(packed_graph_dataset,"PackedGraphDataset",forbidden)
    with pytest.raises(ValueError,match="not released"):
        IndexedRoleDataset(root,replace(p,acceptance_sha256=file_digest(root/"acceptance.json")),authority=a,role_history=h)
    draft=mirror_prerequisites(acceptance)
    assert draft["private_required"] and draft["status"]=="NOT_READY_FOR_PUBLICATION"
    assert "independent_audit" not in draft["roles"] and not draft["automatic_training_action"]


@pytest.mark.parametrize("kind", ["audit_path","outside_path","junction_path","offset_bounds","offset_duplicate","offset_order","feature_schema","pure_2d"])
def test_repinned_semantically_invalid_maps_are_not_authorized(tmp_path,monkeypatch,kind):
    from dataclasses import replace
    from molgap import packed_graph_dataset
    root,p,a,h=fixture(tmp_path,"train_100k")
    manifest_path=root/"roles/train_100k.json"
    m=json.loads(manifest_path.read_bytes());item=m["graph_chunks"][0]
    if kind=="audit_path":item["file"]="payloads/independent_audit/must-not-open.pt"
    elif kind=="outside_path":item["file"]="payloads/full_minus_holdout/../../outside.pt"
    elif kind=="junction_path":
        from pathlib import Path
        from types import SimpleNamespace
        import stat
        item["file"]="payloads/full_minus_holdout/junction/must-not-open.pt"
        original=Path.lstat
        def fake_reparse(path,*args,**kwargs):
            if path.name=="junction":return SimpleNamespace(st_mode=stat.S_IFDIR,st_file_attributes=0x400)
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,"lstat",fake_reparse)
    elif kind=="offset_bounds":item["offsets"]=[999]*item["rows"]
    elif kind=="offset_duplicate":item["offsets"]=[0]*item["rows"]
    elif kind=="offset_order":item["offsets"]=list(reversed(item["offsets"]))
    elif kind=="feature_schema":m["feature_schema"]="geometry-v1"
    else:m["pure_2d"]=False
    manifest_path.write_bytes(json_bytes(m));manifest_sha=file_digest(manifest_path)
    receipt_path=root/"acceptance.json";receipt=json.loads(receipt_path.read_bytes())
    receipt["roles"][p.role]["sha256"]=manifest_sha;receipt_path.write_bytes(json_bytes(receipt))
    a["manifest_sha256"]=h["manifest_sha256"]=manifest_sha
    p=replace(p,manifest_sha256=manifest_sha,acceptance_sha256=file_digest(receipt_path),
        authority_sha256=canonical_fingerprint(a),role_history_sha256=canonical_fingerprint(h))
    if kind in {"audit_path","outside_path","junction_path","feature_schema","pure_2d"}:
        def forbidden(*args,**kwargs):raise AssertionError("Forbidden physical payload was opened")
        monkeypatch.setattr(packed_graph_dataset,"PackedGraphDataset",forbidden)
    with pytest.raises(ValueError):IndexedRoleDataset(root,p,authority=a,role_history=h)
